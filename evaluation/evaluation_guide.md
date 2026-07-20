# Evaluation Guide

## Overview

This folder evaluates model-predicted weak signals against the human-validated ground-truth weak signals across 4 settings:

| Setting | Name | Method |
|---------|------|--------|
| 1 | Set-level BERTScore | Concatenate all signals per set → BERTScore on two strings |
| 2 | Set-level LLM | Give both sets to an LLM judge → P/R recomputed from the matched pairs it returns → F1 |
| 3 | Signal-level BERTScore | Pairwise BERTScore F1 matrix → greedy max aggregation |
| 4 | Signal-level LLM | Per-signal binary LLM judgment, run in both directions → count-based P/R |

All evaluation is computed within a single `(topic, direction)` pair.

Two details that matter when comparing numbers:

- **Setting 2** asks the judge for `precision`, `recall` and `matched_pairs`, but only uses the returned floats as a sanity check; the reported precision and recall are recomputed from `matched_pairs` (matched ground-truth / predicted items over set size), so they are always consistent with the evidence.
- **Setting 4** issues two judgments per run — candidates against references for precision, and references against candidates for recall.

**Aggregation.** F1 is computed per `(topic, direction)` pair, and the reported `f1_mean` is the plain mean of those per-pair F1 values — it is *not* recomputed from the mean precision and mean recall, so `f1_mean != 2PR/(P+R)` in general. For LLM settings the per-pair value is itself the mean over `--n-runs` judge runs. `*_std` is the sample standard deviation across pairs.

---

## File Structure

```
evaluation/
├── loaders.py           # Load GT + predicted signals from disk
├── bertscore_eval.py    # Settings 1 and 3
├── llm_set_eval.py      # Setting 2
├── llm_signal_eval.py   # Setting 4
├── prompt.py            # Shared LLM-judge prompt templates
├── metrics.py           # Metric aggregation helpers
├── run_all.py           # Main runner (use this)
├── outputs/             # Per-model result JSONs written here
└── evaluation_guide.md  # This file
```

---

## Setup

Install dependencies from the repo root (this includes `bert-score`):

```bash
pip install -r requirements.txt
```

Quick check that BERTScore is importable:

```bash
python -c "import bert_score; print('ok')"
```

Settings 2 & 4 call an LLM judge over an OpenAI-compatible endpoint. Set the API key for your provider in a `.env` (loaded automatically from `construction/.env`, the repo root `.env`, or `prediction/.env`):

```bash
# default provider: openrouter (neutral judge, e.g. anthropic/claude-opus-4.8)
OPENROUTER_API_KEY=sk-...
```

`OPENROUTER_BASE_URL` overrides the default OpenRouter endpoint.

`.env` is git-ignored — never commit keys.

---

## Running Evaluations

Run from the repo root:

```bash
cd /path/to/BackTrend   # the directory you cloned this repo into
```

Evaluation reads:
- **Ground truth** from `construction/weak_signal.json` (produced by the construction pipeline + human validation). The file ships with the repo; it can be regenerated from the released Hugging Face dataset with `python construction/scripts/build_weak_signal_json.py --download`.
- **Predictions** from `prediction/outputs/`.

The set of models and topics is discovered automatically from the prediction tree — every model directory under `prediction/outputs/` is evaluated against the GT.

### Settings 1 & 3 only (BERTScore, no API needed)
```bash
python evaluation/run_all.py --settings 1 3
```

### Settings 2 & 4 only (LLM judge)
```bash
python evaluation/run_all.py --settings 2 4
```

### All 4 settings
```bash
python evaluation/run_all.py --settings 1 2 3 4
```

### Specific models only
```bash
python evaluation/run_all.py --settings 1 3 --models gpt5.4 tongyi
```

### Specific topics only
```bash
# by slug or display name
python evaluation/run_all.py --settings 1 3 --topics large_language_models "explainable ai"
```

### One direction only
```bash
python evaluation/run_all.py --settings 1 3 --directions problem
```

### Incremental continuation (default)
```bash
python evaluation/run_all.py --settings 1 2 3 4
```
`--skip-existing` is **on by default**: already-scored `(topic, direction)` entries in an existing output JSON are reused. Pass `--no-skip-existing` to recompute everything.

### Configuring the LLM judge ```bash python evaluation/run_all.py --settings 2
4 \ --judge-model anthropic/claude-opus-4.8 \ --n-runs 5 ``` The judge runs over OpenRouter with `anthropic/claude-opus-4.8` by default, reading `OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`). Override any of these with `--judge-model`, `--api-key`, `--base-url`.

### Scoring a prediction variant
```bash
python evaluation/run_all.py --settings 1 3 \
    --prediction-root prediction/outputs_b/b0 \
    --outputs-dir evaluation/outputs_b/b0
```

---

## Output Format

Results are written as one JSON per model per setting:

```
evaluation/outputs/<model>/set_bertscore.json     # Setting 1
evaluation/outputs/<model>/set_llm.json           # Setting 2
evaluation/outputs/<model>/signal_bertscore.json  # Setting 3
evaluation/outputs/<model>/signal_llm.json        # Setting 4
```

Each file holds every `(topic, direction)` result:

```json
{
  "model": "gpt5.4",
  "setting": "signal_bertscore",
  "generated_at": "20260101T000000Z",
  "summary": { "n_total": 50, "n_scored": 42, "n_no_gt": 8,
               "precision_mean": 0.41, "recall_mean": 0.38, "f1_mean": 0.39, "...": "..." },
  "results": [
    { "topic": "large language models", "topic_slug": "large_language_models",
      "direction": "problem", "status": "scored",
      "n_gt": 6, "n_pred": 10, "precision": 0.5, "recall": 0.33, "f1": 0.4 }
  ]
}
```

- `status` is `scored`, `no_gt` (no validated GT for that topic/direction — not scored), or `error`. With the shipped ground truth, 30 of the 50 `(topic, direction)` pairs are `scored` and the other 20 are `no_gt`.
- A pair that has ground truth but for which the model predicted nothing is still `scored`, with all-zero metrics and an extra `"note": "empty_pred"` field, and it *is* included in the means.
- For LLM settings (2 & 4), metrics are averaged over `--n-runs` judge runs.
- Mean P/R/F1 is printed after each (model, setting) completes, followed by a final counts summary.

---

## Data Sources

- **Ground truth**: `construction/weak_signal.json`
  - Shape: `{ "<topic display name>": { "problem": [...], "solution": [...] } }`
  - Only human-validated signals are present; a missing `(topic, direction)` is treated as "no GT".
- **Predictions**: `prediction/outputs/<model>/<domain_slug>/<topic_slug>/<direction>/2019_2023/signals_latest.json`
  - Extracts `data["signals"]`.

Model names are simply the directory names under `prediction/outputs/` (e.g. `gpt5.4`, `tongyi`, `deepseek_r1_0528`, `qwen3.5_397b`, `qwen3_8b_rag`, `qwen3_30b_awq_rag`, `dr_tulu`).

---

## Notes

- **BERTScore** uses `roberta-large` with `rescale_with_baseline=False`. The first run downloads the model (~500MB) and caches it automatically.
- **LLM settings** use the OpenRouter endpoint (`OPENROUTER_API_KEY`). Requests use `User-Agent: Mozilla/5.0`.
- Default `--n-runs` is `3` for LLM settings (use more for lower-variance publication numbers). The judge is called with `temperature=1.0`, which is hardcoded and not exposed as a flag — this is the source of the run-to-run variance that `--n-runs` averages over.
- Evaluation reads predictions from `prediction/outputs/` by default (`--prediction-root`). Run the prediction stage first; the directory must exist or the run fails immediately.
- Settings 1 and 3 need no API key; the judge key is only required when setting 2 or 4 is requested.
- `run_all.py` auto-loads `.env` from `construction/`, the repo root, and `prediction/`.
- Each output JSON is written when its (model, setting) pass completes, so finished model×setting files are preserved if a longer run stops midway and are reused on the next run (`--skip-existing`, on by default).
