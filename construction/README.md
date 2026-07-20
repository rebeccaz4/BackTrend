# Construction pipeline

This folder contains the weak-signal construction pipeline. The example below runs the pipeline for the target topic `large language models`.

Target topics are configured in [`topics.json`](topics.json) (25 topics, each with a list of `paraphrase` query variants). Omitting `--topic` makes steps 1, 2, 3 and 5 process all 25 topics; step 4 requires `--topic` and handles a single topic per invocation. The paraphrases are used by `fetch_papers.py` as additional Semantic Scholar search queries; later stages use only the topic name.

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The scripts read credentials from environment variables or `construction/.env`:

```bash
SEMANTIC_SCHOLAR_API_KEY=...   # paper retrieval + reference-adoption matching
OPENAI_API_KEY=...             # candidate extraction (gpt-5.4)
OPENROUTER_API_KEY=...         # clustering embeddings (step 3 as invoked below)
```

`SEMANTIC_SCHOLAR_API_KEY` is used for paper retrieval and reference-adoption matching; `S2_API_KEY` is accepted as an alias. Candidate extraction (`extract_candidate.py`) uses the OpenAI API (`OPENAI_API_KEY`) with `gpt-5.4`; pass `--model` to use a different OpenAI model.

Semantic clustering (`dedupe_candidates.py`) **defaults to a local sentence-transformers model**, which requires the optional `sentence-transformers` dependency. The pipeline and the step-3 command below instead pass OpenRouter embeddings, which is the configuration used to build the released benchmark and needs `OPENROUTER_API_KEY`:

| | `--cluster-provider` | `--cluster-embed-model` |
| --- | --- | --- |
| Script default | `local` | `BAAI/bge-large-en-v1.5` |
| Released benchmark | `openrouter` | `openai/text-embedding-3-large` |

## Pipeline

Run all commands from the repository root.

### 1. Fetch target-topic papers

Fetch the 2019-2024 paper pools for the target topic:

```bash
python construction/scripts/fetch_papers.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023 2024
```

Outputs:

```text
construction/papers/large-language-models/papers_large-language-models_{year}.parquet
```

### 2. Extract candidate topics

Extract problem-space and solution-space candidate topics from the 2019-2023 papers:

```bash
python construction/scripts/extract_candidate.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023
```

Outputs:

```text
construction/candidate_topics/large-language-models/candidate_topics_large-language-models_{year}.jsonl
```

### 3. Deduplicate and cluster candidates

Deduplicate candidate labels and cluster semantically similar candidates:

```bash
python construction/scripts/dedupe_candidates.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023 \
  --use-clustering \
  --cluster-provider openrouter \
  --cluster-embed-model openai/text-embedding-3-large \
  --cluster-threshold 0.85 \
  --output-suffix _cluster_t0.85
```

Outputs:

```text
construction/candidate_dedup/large-language-models/candidate_index_large-language-models_cluster_t0.85.json
construction/candidate_dedup/large-language-models/candidate_clusters_large-language-models_cluster_t0.85.json
```

### 4. Match later adoption through references

Fetch references from 2024 target-topic papers and match them to candidate source papers:

```bash
python construction/scripts/match_candidate_reference_adoption.py \
  --topic "large language models" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --output-suffix _cluster_t0.85 \
  --batch-size 500
```

Outputs:

```text
construction/candidate_matching/large-language-models/reference_match_large-language-models_cluster_t0.85.parquet
construction/candidate_matching/large-language-models/reference_match_large-language-models_cluster_t0.85.json
```

(`--output-suffix` must match step 5's `--matching-suffix` so that `compute_final_results.py` finds these files.)

### 5. Compute final onset scores, gates, and weak signals

Compute source-grounded early frequencies, 2024 reference-adoption frequencies, trend scores, and final onset/impact scores, then apply the hard gates and write the final results:

```bash
python construction/scripts/compute_final_results.py \
  --topic "large language models" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --matching-suffix _cluster_t0.85 \
  --validation-lift-threshold 1.2
```

Outputs, per topic and space (`problem` / `solution`):

```text
construction/final_results/large-language-models/<space>/all_scored.csv
construction/final_results/large-language-models/<space>/top30_by_score.csv
construction/final_results/large-language-models/<space>/passed_all_gates/weak_signals.csv
construction/final_results/large-language-models/<space>/passed_all_gates/weak_signals.md
construction/final_results/large-language-models/<space>/gate{1,2,3,4}_*/          # per-gate audit CSVs + plots
```

`passed_all_gates/weak_signals.{csv,md}` holds the candidates that survive every hard gate — the per-topic weak-signal candidates.

> **Ground truth.** These candidates are the input to human validation. The final, human-validated weak signals used as ground truth by `evaluation/` live in `construction/weak_signal.json` (topic → `problem` / `solution` → validated signals). That file is produced by the annotation step, not by this script. It ships with this repository and is also distributed with the released dataset; regenerate it with `python construction/scripts/build_weak_signal_json.py --download`.

## Metrics

All counts exclude survey papers (title matching the word `survey`, case-insensitively).

For each candidate topic and early year `y` in `2019-2023`:

```text
n_y = number of unique non-survey source papers in year y
N_y = total number of non-survey papers in the target-topic paper pool in year y
f_y = n_y / N_y
```

Later-period adoption:

```text
n_later = number of unique non-survey 2024 target-topic papers that cite the candidate's early source papers
N_later = total number of non-survey 2024 target-topic papers
ref_f_2024 = n_later / N_later
```

### Onset scoring

For each candidate and each onset year `o` in `2019-2022` (`--onset-years`), a log-linear trend `log(f + eps) ~ year` is fitted to the onset window `[f_o, ..., f_2023]`, giving `slope` and `R^2`. Window statistics:

```text
growth          = f_2023 / (first nonzero f in the window)      # a ratio, not a difference
positive_share  = fraction of year-over-year deltas that are positive
end_strength    = f_2023 / max(f in the window)
terminal        = min(1, f_2023 / f_2022)
pre_onset_penalty = min(1, f_o / (pre-onset peak f))   # 1 if no pre-onset activity
```

The raw score is `0` when `slope <= 0`; otherwise:

```text
problem space:
raw = R^2^2 * log(growth) * positive_share * end_strength * pre_onset_penalty
      (log(growth) term is 0 unless growth > 1)

solution space:
raw = R^2^3 * log1p(growth - 1) * positive_share * end_strength * terminal
      * nonzero_penalty * pre_onset_penalty^2
      (nonzero_penalty = min(1, nonzero_years / 3)^2)
```

The onset year with the highest `raw` is kept, and the final `score` is `raw` min-max normalized within each (topic, space).

### Hard gates

A candidate becomes a weak-signal candidate only if it passes all four gates (defaults shown are the pipeline's values):

1. **Signal presence** — `score > 0`.
2. **2024 validation peak** — `ref_f_2024 >= 1.2 * max(f_2019..f_2023)` (`--validation-lift-threshold`, default `1.2`).
3. **Trajectory shape** — problem space: 2023 retention, `f_2023 >= 0.8 * (onset-to-2023 peak)`; solution space: smooth growth, every year-over-year step retains `>= 0.8x` the previous value (`--retention-tolerance 0.8`), where steps from years with `<= 1` paper may be skipped (`--gate3-skip-current-n`).
4. **Pre-onset not too high** — the pre-onset peak must not exceed `max(f_onset, 0.6 * f_2023)` (`--pre-peak-tolerance 0.6`).
