#!/usr/bin/env bash
set -euo pipefail

TOPIC=""
SPACE="all"
CONDA_ENV=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  bash construction/scripts/run_final_pipeline.sh --topic "large language models" [--space all|problem-space|solution-space]

Options:
  --topic      Target topic name from construction/topics.json. Required.
  --space      Candidate space to process: all, problem-space, or solution-space. Default: all.
  --conda-env  Optional conda environment name. Default: use the active environment.
USAGE
}

step() {
  echo
  echo "==> $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      TOPIC="${2:-}"
      shift 2
      ;;
    --space)
      SPACE="${2:-all}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TOPIC}" ]]; then
  echo "--topic is required." >&2
  usage >&2
  exit 2
fi

if [[ "${SPACE}" != "all" && "${SPACE}" != "problem-space" && "${SPACE}" != "solution-space" ]]; then
  echo "--space must be one of: all, problem-space, solution-space." >&2
  exit 2
fi

cd "${ROOT_DIR}"

run_py() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "${CONDA_ENV}" python "$@"
  else
    python "$@"
  fi
}

step "Step 1/5: Fetch target-topic papers for 2019-2024"
run_py construction/scripts/fetch_papers.py \
  --topic "${TOPIC}"

step "Step 2/5: Extract problem-space and solution-space candidate topics from 2019-2023 papers"
run_py construction/scripts/extract_candidate.py \
  --topic "${TOPIC}"

step "Step 3/5: Simple dedup candidates, then cluster them with threshold 0.85"
run_py construction/scripts/dedupe_candidates.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --use-clustering \
  --cluster-threshold 0.85 \
  --cluster-provider openrouter \
  --cluster-embed-model openai/text-embedding-3-large \
  --output-suffix _cluster_t0.85

step "Step 4/5: Match 2024 reference adoption and write reference_match outputs"
run_py construction/scripts/match_candidate_reference_adoption.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --output-suffix _cluster_t0.85

step "Step 5/5: Compute final onset scores, apply hard gates, and write final_results"
run_py construction/scripts/compute_final_results.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --matching-suffix _cluster_t0.85 \
  --validation-lift-threshold 1.2

echo
echo "Done. Results are under construction/final_results/."
