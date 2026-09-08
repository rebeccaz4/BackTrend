"""Setting 4 — Signal-level LLM Judgment.

For each predicted signal, ask the LLM: does it match any GT signal? (0/1)
For each GT signal, ask the LLM: does it match any predicted signal? (0/1)

  Precision = sum(precision_matches) / n_pred
  Recall    = sum(recall_matches)    / n_gt

Batches candidates in groups of batch_size to reduce API calls.
Runs N times and returns mean ± std.

Prompt templates live in evaluation/prompt.py; metric aggregation comes from
evaluation/metrics.py.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

from openai import AsyncOpenAI

# Shared prompts live in evaluation/prompt.py (SYSTEM_PROMPT identical to the
# set-level judge); metric aggregation in evaluation/metrics.py.
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from prompt import SYSTEM_PROMPT, build_pairwise_prompt                          # noqa: E402
from metrics import compute_metrics                                              # noqa: E402

DEFAULT_BATCH_SIZE = 5
DEFAULT_N_WORKERS  = 8
JUDGE_MAX_TOKENS   = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))


def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


async def _judge_batch(
    client: AsyncOpenAI,
    model: str,
    reference: list[str],
    candidates: list[str],
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
) -> list[int]:
    """Ask LLM whether each candidate matches any reference. Returns 0/1 list."""
    async with semaphore:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    # The reply is a short JSON array of 0/1, one per candidate
                    # in the batch. Without a cap the provider reserves credit
                    # for the model's full output window (65k tokens), which is
                    # enough to exhaust an API balance mid-run.
                    max_tokens=JUDGE_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_pairwise_prompt(reference, candidates)},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("Empty response")
                payload = _safe_json(text)
                if payload is None:
                    raise ValueError("Could not parse JSON")
                matches = payload.get("matches")
                if not isinstance(matches, list) or len(matches) != len(candidates):
                    raise ValueError(f"Expected {len(candidates)} matches, got {matches}")
                result = [int(v) for v in matches]
                if not all(v in (0, 1) for v in result):
                    raise ValueError(f"Non-binary values: {result}")
                return result

            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(f"Judge batch failed after {attempt} attempts: {exc}") from exc
                import asyncio as _aio
                await _aio.sleep(retry_backoff * attempt)


async def _run_direction(
    client: AsyncOpenAI,
    model: str,
    reference: list[str],
    candidates: list[str],
    temperature: float,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> list[int]:
    """Run batched matching: does each candidate match any reference?"""
    batches = [
        candidates[i * batch_size : (i + 1) * batch_size]
        for i in range(math.ceil(len(candidates) / batch_size))
    ]
    results = await asyncio.gather(*[
        _judge_batch(client, model, reference, batch, temperature, semaphore)
        for batch in batches
    ])
    return [v for batch in results for v in batch]


async def _run_once(
    client: AsyncOpenAI,
    model: str,
    gt: list[str],
    pred: list[str],
    temperature: float,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """One evaluation run: precision pass + recall pass in parallel."""
    prec_task = _run_direction(client, model, gt,   pred, temperature, batch_size, semaphore)
    rec_task  = _run_direction(client, model, pred, gt,   temperature, batch_size, semaphore)
    prec_matches, rec_matches = await asyncio.gather(prec_task, rec_task)
    return compute_metrics(prec_matches, rec_matches, n_gt=len(gt), n_ext=len(pred))


async def _run_all(
    gt: list[str],
    pred: list[str],
    client: AsyncOpenAI,
    judge_model: str,
    n_runs: int,
    temperature: float,
    batch_size: int,
    n_workers: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(n_workers)
    tasks = [
        _run_once(client, judge_model, gt, pred, temperature, batch_size, semaphore)
        for _ in range(n_runs)
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


def eval_signal_llm(
    gt: list[str],
    pred: list[str],
    api_key: str,
    base_url: str,
    judge_model: str,
    user_agent: str = "Mozilla/5.0",
    n_runs: int = 5,
    temperature: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_workers: int = DEFAULT_N_WORKERS,
) -> dict:
    """Setting 4: signal-level LLM judgment. Returns mean/std over n_runs."""

    async def _main() -> list[dict]:
        async with AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"User-Agent": user_agent},
        ) as client:
            runs = await _run_all(gt, pred, client, judge_model, n_runs, temperature, batch_size, n_workers)
        return runs

    runs = asyncio.run(_main())

    for i, r in enumerate(runs, 1):
        print(f"    run {i}/{n_runs}: P={r['precision']} R={r['recall']} F1={r['f1']}")

    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        mean = round(statistics.mean(vals), 4)
        std  = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        return mean, std

    p_mean, p_std = _agg("precision")
    r_mean, r_std = _agg("recall")
    f_mean, f_std = _agg("f1")

    return {
        "setting": "signal_llm",
        "precision": p_mean, "precision_std": p_std,
        "recall":    r_mean, "recall_std":    r_std,
        "f1":        f_mean, "f1_std":        f_std,
        "n_runs": n_runs,
        "n_pred": len(pred),
        "n_gt":   len(gt),
    }
