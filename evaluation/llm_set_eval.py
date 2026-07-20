"""Setting 2 — Set-level LLM Judgment.

Give both GT and predicted signal sets to an LLM in one prompt. The LLM returns
the matched pairs (which GT signal matched which predicted signal) plus its own
precision/recall.

Precision/recall are computed from the matched PAIRS, not from the LLM's self-
reported numbers:

    precision = manual_p = (# predicted topics appearing in some pair) / n_pred
    recall    = manual_r = (# GT topics appearing in some pair)        / n_gt
    f1        = harmonic mean of the two

Both keys are emitted (``precision``==``manual_p``, ``recall``==``manual_r``) so
the format stays comparable across runs and the score is always backed by the
pairs. The LLM's own precision/recall are still parsed and range-validated, but
only as a response-validity gate — they never affect the score.

Runs N times and returns mean ± std, keeping every run's pairs.
"""
from __future__ import annotations

import json
import re
import statistics
import time

import sys
from pathlib import Path

from openai import OpenAI

# Shared prompts live in evaluation/prompt.py — SYSTEM_PROMPT is identical to
# the signal-level judge; only the user message differs.
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from prompt import SYSTEM_PROMPT, build_set_prompt  # noqa: E402


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


def _normalize_pairs(raw_pairs, gt: list[str], pred: list[str]) -> list[dict]:
    """Turn raw {gt_index, pred_index} objects into readable evidence pairs.

    Out-of-range or malformed entries are skipped rather than fatal — evidence
    should never crash a run.
    """
    pairs: list[dict] = []
    if not isinstance(raw_pairs, list):
        return pairs
    for entry in raw_pairs:
        if not isinstance(entry, dict):
            continue
        gi = entry.get("gt_index")
        pi = entry.get("pred_index")
        if not isinstance(gi, int) or not isinstance(pi, int):
            continue
        if not (0 <= gi < len(gt)) or not (0 <= pi < len(pred)):
            continue
        pairs.append(
            {
                "gt_index": gi,
                "pred_index": pi,
                "ground_truth": gt[gi],
                "predicted": pred[pi],
            }
        )
    return pairs


def _run_once(
    client: OpenAI,
    model: str,
    gt: list[str],
    pred: list[str],
    temperature: float,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
) -> dict:
    """One LLM call. Returns validated P/R/F1 plus matched-pair evidence."""
    prompt = build_set_prompt(gt, pred)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
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
            precision_raw = payload.get("precision")
            recall_raw = payload.get("recall")
            if not isinstance(precision_raw, (int, float)):
                raise ValueError(f"precision is not numeric: {precision_raw!r}")
            if not isinstance(recall_raw, (int, float)):
                raise ValueError(f"recall is not numeric: {recall_raw!r}")

            llm_precision = float(precision_raw)
            llm_recall = float(recall_raw)
            if not 0.0 <= llm_precision <= 1.0:
                raise ValueError(f"precision out of range [0,1]: {llm_precision}")
            if not 0.0 <= llm_recall <= 1.0:
                raise ValueError(f"recall out of range [0,1]: {llm_recall}")

            matched_pairs = _normalize_pairs(payload.get("matched_pairs"), gt, pred)

            # Score from the matched pairs (the evidence), NOT the LLM's self-
            # reported precision/recall. manual_p = fraction of predicted topics
            # that appear in some pair; manual_r = fraction of GT topics that do.
            # The LLM's own precision/recall are still parsed/validated above only
            # as a response-validity gate (a malformed answer triggers a retry).
            n_pred, n_gt = len(pred), len(gt)
            matched_preds = {p["pred_index"] for p in matched_pairs}
            matched_gts = {p["gt_index"] for p in matched_pairs}
            manual_p = len(matched_preds) / n_pred if n_pred else 0.0
            manual_r = len(matched_gts) / n_gt if n_gt else 0.0
            denom = manual_p + manual_r
            f1 = 2 * manual_p * manual_r / denom if denom > 0 else 0.0

            manual_p, manual_r = round(manual_p, 4), round(manual_r, 4)
            return {
                "precision": manual_p,
                "manual_p": manual_p,
                "recall": manual_r,
                "manual_r": manual_r,
                "f1": round(f1, 4),
                "matched_pairs": matched_pairs,
            }

        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"LLM set judge failed after {attempt} attempts: {exc}") from exc
            sleep_s = retry_backoff * attempt
            print(f"  [warn] attempt {attempt}: {exc}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def eval_set_llm(
    gt: list[str],
    pred: list[str],
    client: OpenAI,
    judge_model: str,
    n_runs: int = 5,
    temperature: float = 1.0,
) -> dict:
    """Setting 2: set-level LLM judgment. Returns mean/std over n_runs.

    Each run's matched-pair evidence is preserved under ``runs`` so the reported
    precision/recall can be manually verified against the pairs the judge used.
    """
    runs = []
    for i in range(n_runs):
        result = _run_once(client, judge_model, gt, pred, temperature)
        runs.append(result)
        print(
            f"    run {i+1}/{n_runs}: P={result['precision']} R={result['recall']} "
            f"F1={result['f1']} pairs={len(result['matched_pairs'])}"
        )

    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        mean = round(statistics.mean(vals), 4)
        std  = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        return mean, std

    p_mean, p_std = _agg("precision")
    r_mean, r_std = _agg("recall")
    f_mean, f_std = _agg("f1")

    return {
        "setting": "set_llm",
        "precision": p_mean, "manual_p": p_mean, "precision_std": p_std,
        "recall":    r_mean, "manual_r": r_mean, "recall_std":    r_std,
        "f1":        f_mean, "f1_std":        f_std,
        "n_runs": n_runs,
        "n_pred": len(pred),
        "n_gt":   len(gt),
        "runs": runs,
    }
