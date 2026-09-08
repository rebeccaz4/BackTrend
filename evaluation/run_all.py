#!/usr/bin/env python3
"""Main evaluation runner.

For every (model, topic, direction) that a model produced predictions for,
run the selected evaluation settings against the validated GT in
construction/weak_signal.json.

Output layout (one folder per model, one JSON per algorithm):

    evaluation/outputs/<model>/set_bertscore.json     # Setting 1
    evaluation/outputs/<model>/set_llm.json           # Setting 2
    evaluation/outputs/<model>/signal_bertscore.json  # Setting 3
    evaluation/outputs/<model>/signal_llm.json        # Setting 4
    evaluation/outputs/<model>/coverage_at_10.json    # Setting 5

Each JSON holds every (topic, direction) result. Combinations with no
validated GT are recorded with status "no_gt" and skipped (not scored).

LLM settings (2, 4 & 5) use an OpenRouter endpoint with a neutral judge model
(none of the evaluated systems are Claude).

Usage:
    python evaluation/run_all.py --settings 1 3             # BERTScore only (no API)
    python evaluation/run_all.py --settings 2 4             # LLM only
    python evaluation/run_all.py --settings 1 2 3 4         # the four default settings
    python evaluation/run_all.py --settings 5               # Coverage@10 (opt-in)
    python evaluation/run_all.py --models gpt5.4 tongyi --settings 1 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Ensure evaluation/ is importable
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from loaders import PREDICTION_ROOT, iter_eval_items, slugify  # noqa: E402

REPO_ROOT = _EVAL_DIR.parent
OUTPUTS_DIR = _EVAL_DIR / "outputs"
DIRECTIONS = ("problem", "solution")

SETTING_NAMES = {1: "set_bertscore", 2: "set_llm", 3: "signal_bertscore", 4: "signal_llm",
                 5: "coverage_at_10"}
LLM_SETTINGS = {2, 4, 5}

# LLM judge endpoint (OpenAI-compatible chat completions).
# Claude Opus 4.8 over OpenRouter is a neutral judge: none of the evaluated
# systems are Claude.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.8"


# ---------------------------------------------------------------------------
# Environment / provider
# ---------------------------------------------------------------------------

def _load_eval_env() -> None:
    env_candidates = [
        REPO_ROOT / "construction" / ".env",
        REPO_ROOT / ".env",
        REPO_ROOT / "prediction" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _first_env(names: list[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve_provider(api_key_override: str | None,
                     base_url_override: str | None, judge_model_override: str | None) -> dict:
    """Resolve (api_key, base_url, judge_model) for the LLM judge."""
    return {
        "api_key": api_key_override or _first_env(["OPENROUTER_API_KEY"]),
        "base_url": base_url_override or _first_env(["OPENROUTER_BASE_URL"]) or OPENROUTER_BASE_URL,
        "judge_model": judge_model_override or DEFAULT_JUDGE_MODEL,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all evaluation settings.")
    p.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4],
                   help="1=set-BERTScore, 2=set-LLM, 3=signal-BERTScore, 4=signal-LLM, "
                        "5=Coverage@K (opt-in; not part of the default set)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to evaluate (default: all in prediction/outputs/)")
    p.add_argument("--topics", nargs="*", default=None,
                   help="Filter topics by slug or display name, e.g. large_language_models or 'large language models'")
    p.add_argument("--directions", nargs="*", default=["problem", "solution"],
                   choices=["problem", "solution"])
    p.add_argument("--judge-model", default=None,
                   help=f"LLM model for settings 2, 4 & 5 (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--api-key", default=None,
                   help="Override the judge API key (default: $OPENROUTER_API_KEY)")
    p.add_argument("--base-url", default=None,
                   help="Override the judge base URL (default: the OpenRouter endpoint)")
    p.add_argument("--user-agent", default="Mozilla/5.0")
    p.add_argument("--prediction-root", default=None,
                   help="Root of the prediction tree to evaluate "
                        "(default: prediction/outputs). Point at a variant dir like "
                        "prediction/outputs_b/b0 to score a sweep variant.")
    p.add_argument("--outputs-dir", default=None,
                   help="Directory to write per-model result JSONs into "
                        "(default: evaluation/outputs). Use a distinct dir per variant so "
                        "same-named models don't overwrite each other.")
    p.add_argument("--coverage-k", type=int, default=10,
                   help="K for setting 5 (Coverage@K / Recall@K); default 10")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Number of LLM judge runs for settings 2, 4 & 5 (default: 3)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                   help="Reuse already-scored (topic, direction) entries from existing output JSON (default: enabled)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Recompute every entry even if already present in the output JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _output_path(model: str, setting_name: str) -> Path:
    return OUTPUTS_DIR / model / f"{setting_name}.json"


def _load_existing_results(path: Path) -> dict[tuple, dict]:
    """Map (topic_slug, direction) -> prior result record from an output JSON."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[tuple, dict] = {}
    for rec in data.get("results", []):
        key = (rec.get("topic_slug"), rec.get("direction"))
        out[key] = rec
    return out


def _summarize(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("status") == "scored"]
    summary = {
        "n_total": len(results),
        "n_scored": len(scored),
        "n_no_gt": sum(1 for r in results if r.get("status") == "no_gt"),
    }
    for metric in ("precision", "recall", "f1", "coverage", "coverage_full"):
        vals = [r[metric] for r in scored if isinstance(r.get(metric), (int, float))]
        if vals:
            summary[f"{metric}_mean"] = round(statistics.mean(vals), 4)
            summary[f"{metric}_std"] = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        elif metric in ("precision", "recall", "f1"):
            # Coverage keys only appear for setting 5; P/R/F1 keep their
            # existing "always present, possibly null" shape.
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
    return summary


def _write_output(path: Path, model: str, setting_name: str, judge_model: str | None,
                  ts: str, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "setting": setting_name,
        "generated_at": ts,
        "summary": _summarize(results),
        "results": results,
    }
    if judge_model is not None:
        payload["judge_model"] = judge_model
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _empty_pred_metrics(setting_name: str) -> dict:
    """Zero metrics when GT exists but the model produced no predictions."""
    if setting_name == "set_llm":
        # Mirror eval_set_llm's key layout (precision==manual_p, recall==manual_r).
        return {"setting": setting_name,
                "precision": 0.0, "manual_p": 0.0, "recall": 0.0, "manual_r": 0.0,
                "f1": 0.0, "note": "empty_pred", "matched_pairs": []}
    if setting_name.startswith("coverage_at_"):
        return {"setting": setting_name, "coverage": 0.0, "coverage_full": 0.0,
                "note": "empty_pred"}
    return {"setting": setting_name, "precision": 0.0, "recall": 0.0, "f1": 0.0, "note": "empty_pred"}


def score_item(setting: int, item, ctx: dict) -> dict:
    """Compute the metric dict for one scored (GT-present) item."""
    setting_name = SETTING_NAMES[setting]
    if not item.has_pred:
        return _empty_pred_metrics(setting_name)

    gt, pred = item.gt_signals, item.pred_signals
    if setting == 1:
        return ctx["eval_set_bertscore"](gt, pred)
    if setting == 2:
        return ctx["eval_set_llm"](
            gt, pred, client=ctx["llm_client"], judge_model=ctx["judge_model"], n_runs=ctx["n_runs"]
        )
    if setting == 3:
        return ctx["eval_signal_bertscore"](gt, pred)
    if setting == 4:
        return ctx["eval_signal_llm"](
            gt, pred,
            api_key=ctx["api_key"], base_url=ctx["base_url"], judge_model=ctx["judge_model"],
            user_agent=ctx["user_agent"], n_runs=ctx["n_runs"],
        )
    if setting == 5:
        return ctx["eval_coverage_at_k"](
            gt, pred,
            api_key=ctx["api_key"], base_url=ctx["base_url"], judge_model=ctx["judge_model"],
            user_agent=ctx["user_agent"], n_runs=ctx["n_runs"], k=ctx["coverage_k"],
        )
    raise ValueError(f"Unknown setting: {setting}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_eval_env()
    args = parse_args()

    requested_topics = {slugify(t) for t in args.topics} if args.topics else None

    need_llm = bool(LLM_SETTINGS & set(args.settings))
    prov = resolve_provider(args.api_key, args.base_url, args.judge_model)
    api_key = prov["api_key"]
    base_url = prov["base_url"]
    judge_model = prov["judge_model"]
    if need_llm and not api_key:
        raise SystemExit(
            "Error: no LLM judge API key. "
            "Set OPENROUTER_API_KEY (or pass --api-key) for settings 2, 4 and 5."
        )

    ctx: dict = {
        "judge_model": judge_model,
        "n_runs": args.n_runs,
        "api_key": api_key,
        "base_url": base_url,
        "user_agent": args.user_agent,
        "coverage_k": args.coverage_k,
    }

    # Lazy imports — only load heavy libs when needed
    if 1 in args.settings or 3 in args.settings:
        from bertscore_eval import eval_set_bertscore, eval_signal_bertscore
        ctx["eval_set_bertscore"] = eval_set_bertscore
        ctx["eval_signal_bertscore"] = eval_signal_bertscore
    if 2 in args.settings:
        from llm_set_eval import eval_set_llm
        from openai import OpenAI
        ctx["eval_set_llm"] = eval_set_llm
        ctx["llm_client"] = OpenAI(
            api_key=api_key, base_url=base_url,
            default_headers={"User-Agent": args.user_agent},
        )
    if 4 in args.settings:
        from llm_signal_eval import eval_signal_llm
        ctx["eval_signal_llm"] = eval_signal_llm
    if 5 in args.settings:
        from coverage_eval import eval_coverage_at_k
        SETTING_NAMES[5] = f"coverage_at_{args.coverage_k}"
        ctx["eval_coverage_at_k"] = eval_coverage_at_k

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Optional overrides: evaluate a different prediction tree (e.g. a sweep
    # variant) and/or write results to a different directory. Defaults keep the
    # standard prediction/outputs -> evaluation/outputs behavior.
    prediction_root = Path(args.prediction_root).resolve() if args.prediction_root else PREDICTION_ROOT
    global OUTPUTS_DIR
    if args.outputs_dir:
        OUTPUTS_DIR = Path(args.outputs_dir).resolve()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Prediction root: {prediction_root}")
    print(f"Outputs dir:     {OUTPUTS_DIR}")

    # Group prediction items by model (output is one folder per model).
    items_by_model: dict[str, list] = defaultdict(list)
    for item in iter_eval_items(models=args.models, prediction_root=prediction_root):
        if item.direction not in args.directions:
            continue
        if requested_topics and item.topic_slug not in requested_topics:
            continue
        items_by_model[item.model].append(item)

    if not items_by_model:
        print("No prediction items found for the requested models/topics.")
        return

    print(
        f"Judge model: {judge_model} | base_url: {base_url}"
        if need_llm else "BERTScore only (no LLM)."
    )
    print(f"Skip-existing: {'on' if args.skip_existing else 'off'}")

    total_scored, total_skipped, total_no_gt, errors = 0, 0, 0, 0

    for model in sorted(items_by_model):
        items = items_by_model[model]
        print(f"\n{'='*64}\nMODEL: {model}  ({len(items)} topic/direction combos)")

        for s in args.settings:
            sname = SETTING_NAMES[s]
            out_path = _output_path(model, sname)
            existing = _load_existing_results(out_path) if args.skip_existing else {}
            results: list[dict] = []

            print(f"  [{sname}]")
            for item in items:
                base_rec = {
                    "topic": item.topic_display,
                    "topic_slug": item.topic_slug,
                    "direction": item.direction,
                }
                key = (item.topic_slug, item.direction)

                if not item.has_gt:
                    results.append({**base_rec, "status": "no_gt", "n_pred": len(item.pred_signals)})
                    total_no_gt += 1
                    continue

                prior = existing.get(key)
                if args.skip_existing and prior and prior.get("status") == "scored":
                    results.append(prior)
                    total_skipped += 1
                    continue

                try:
                    metrics = score_item(s, item, ctx)
                    record = {
                        **base_rec,
                        "status": "scored",
                        "n_gt": len(item.gt_signals),
                        "n_pred": len(item.pred_signals),
                        **metrics,
                    }
                    results.append(record)
                    total_scored += 1
                    if "coverage" in metrics:
                        print(f"    {item.topic_slug}/{item.direction}: "
                              f"Coverage@{metrics.get('k')}={metrics.get('coverage')} "
                              f"Coverage@inf={metrics.get('coverage_full')}")
                    else:
                        print(f"    {item.topic_slug}/{item.direction}: "
                              f"P={metrics.get('precision')} R={metrics.get('recall')} F1={metrics.get('f1')}")
                except Exception as exc:
                    errors += 1
                    print(f"    {item.topic_slug}/{item.direction}: ERROR: {exc}")
                    traceback.print_exc()
                    # keep prior result if we had one, else record the error
                    if prior is not None:
                        results.append(prior)
                    else:
                        results.append({**base_rec, "status": "error", "error": str(exc)})

            judge = judge_model if s in LLM_SETTINGS else None
            _write_output(out_path, model, sname, judge, ts, results)
            summ = _summarize(results)
            try:
                shown_path = out_path.relative_to(REPO_ROOT)
            except ValueError:
                shown_path = out_path  # outputs dir outside the repo
            print(f"    -> {shown_path}  "
                  f"(scored={summ['n_scored']}, no_gt={summ['n_no_gt']}, "
                  f"P={summ.get('precision_mean')} R={summ.get('recall_mean')} "
                  f"F1={summ.get('f1_mean')})")

    print(f"\n{'='*64}")
    print(f"Done. Scored: {total_scored}  Reused: {total_skipped}  no_gt: {total_no_gt}  Errors: {errors}")
    try:
        shown_root = OUTPUTS_DIR.relative_to(REPO_ROOT)
    except ValueError:
        shown_root = OUTPUTS_DIR  # outputs dir outside the repo
    print(f"Outputs under: {shown_root}/<model>/<setting>.json")


if __name__ == "__main__":
    main()
