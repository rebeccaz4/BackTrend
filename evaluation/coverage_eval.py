"""Setting 5 — Coverage@K (a.k.a. Recall@K / Subtopic-Recall@K).

For each ground-truth signal, ask the judge whether it is covered by any of the
model's **top-K** predictions.  The prediction list is the model's own ranking
(the ranked-10 prompt asks for "most to least confident, strongest first"), so
top-K is simply the first K elements.

    Coverage@K = |{g in GT : g matches some p in top_K(pred)}| / |GT|

This is the lenient metric the reviewer asked for: it drops the precision term
but keeps the *same* strict matching criterion, the same judge, the same
prompt, and the same number of runs as the signal-level judge (setting 4).
The leniency comes from not penalising extra predictions, not from relaxing
what counts as a match.

Also reports Coverage@inf (no truncation) so the effect of the top-K cut is
visible.  When a model produced <= K signals the two are identical by
construction and the second judging pass is skipped.

Reuses `_run_direction` from llm_signal_eval, i.e. exactly the recall side of
setting 4, so the two metrics are directly comparable.
"""
from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

from openai import AsyncOpenAI

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from llm_signal_eval import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_WORKERS,
    _run_direction,
)

DEFAULT_K = 10


async def _cover_once(
    client: AsyncOpenAI,
    model: str,
    gt: list[str],
    pred_subset: list[str],
    temperature: float,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> list[int]:
    """One judging pass: for each GT signal, is it covered by `pred_subset`?

    Note the argument order — `reference` is the prediction list and
    `candidates` is the GT list, which is exactly how setting 4 computes its
    recall side.
    """
    return await _run_direction(
        client, model, pred_subset, gt, temperature, batch_size, semaphore
    )


async def _run_all(
    gt: list[str],
    pred: list[str],
    k: int,
    client: AsyncOpenAI,
    judge_model: str,
    n_runs: int,
    temperature: float,
    batch_size: int,
    n_workers: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (topk_flag_runs, full_flag_runs); the second reuses the first
    when the model produced <= k predictions."""
    semaphore = asyncio.Semaphore(n_workers)
    top_k = pred[:k]
    truncated = len(pred) > k

    topk_tasks = [
        _cover_once(client, judge_model, gt, top_k, temperature, batch_size, semaphore)
        for _ in range(n_runs)
    ]
    full_tasks = [
        _cover_once(client, judge_model, gt, pred, temperature, batch_size, semaphore)
        for _ in range(n_runs)
    ] if truncated else []

    topk_runs = await asyncio.gather(*topk_tasks)
    full_runs = await asyncio.gather(*full_tasks) if truncated else list(topk_runs)
    return list(topk_runs), list(full_runs)


def _agg(flag_runs: list[list[int]]) -> tuple[float, float]:
    """Mean/std of per-run coverage ratios."""
    ratios = [
        (sum(flags) / len(flags)) if flags else 0.0
        for flags in flag_runs
    ]
    mean = round(statistics.mean(ratios), 4)
    std = round(statistics.stdev(ratios) if len(ratios) > 1 else 0.0, 4)
    return mean, std


def eval_coverage_at_k(
    gt: list[str],
    pred: list[str],
    api_key: str,
    base_url: str,
    judge_model: str,
    user_agent: str = "Mozilla/5.0",
    n_runs: int = 3,
    k: int = DEFAULT_K,
    temperature: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_workers: int = DEFAULT_N_WORKERS,
) -> dict:
    """Setting 5: Coverage@K. Returns mean/std over n_runs."""

    async def _main():
        async with AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"User-Agent": user_agent},
        ) as client:
            return await _run_all(
                gt, pred, k, client, judge_model, n_runs,
                temperature, batch_size, n_workers,
            )

    topk_runs, full_runs = asyncio.run(_main())

    cov_mean, cov_std = _agg(topk_runs)
    full_mean, full_std = _agg(full_runs)

    for i, flags in enumerate(topk_runs, 1):
        covered = sum(flags)
        print(f"    run {i}/{n_runs}: covered {covered}/{len(flags)} "
              f"= {covered / len(flags) if flags else 0.0:.3f}")

    # Per-GT hit count across runs, for auditing which references stay missed.
    n_gt = len(gt)
    gt_hits = [sum(run[j] for run in topk_runs) for j in range(n_gt)] if n_gt else []

    return {
        "setting": f"coverage_at_{k}",
        "coverage": cov_mean, "coverage_std": cov_std,
        "coverage_full": full_mean, "coverage_full_std": full_std,
        "k": k,
        "truncated": len(pred) > k,
        "n_runs": n_runs,
        "n_pred": len(pred),
        "n_gt": n_gt,
        "gt_hits_across_runs": gt_hits,
    }
