# Weak Signal Prediction Guide

This directory contains prediction-only scripts for generating weak-signal predictions with the shared prompt templates in `prediction_prompts.py`.

Current benchmark settings:

- Topic source: `../construction/topics.json`
- Topic count: 25 topics
- Default domain wrapper: `Artificial intelligence and machine learning`
- Prediction window: `2019-2023`
- Output year slug: `2019_2023`
- Default spaces: `problem solution`

When `--domain` is omitted, every script runs the single default domain and all 25 topics from `construction/topics.json`. In normal use, omit `--domain`.

## Scripts

| Script | Model / method | Local GPU |
|---|---|---:|
| `gpt5.4.py` | GPT-5.4 via the OpenAI API | No |
| `deepseek_r1_0528.py` | DeepSeek-R1-0528 / `deepseek-reasoner` API | No |
| `qwen3_5_397b.py` | Qwen3.5-397B-A17B through DashScope API | No |
| `qwen3_8B_rag.py` | Qwen3-8B + Semantic Scholar + LlamaIndex RAG + vLLM | Yes |
| `qwen3_30B_rag.py` | Qwen3-30B-A3B AWQ + Semantic Scholar + LlamaIndex RAG + vLLM | Yes |
| `DR_Tulu_eval.py` | DR-Tulu deep research agent + MCP + vLLM | Yes |
| `Tongyi_eval.py` | Tongyi DeepResearch ReAct agent + vLLM | Yes |

## Common Usage

Run both spaces for all 25 topics:

```bash
cd prediction
python3 qwen3_5_397b.py --output-dir ./outputs
```

Run only one space:

```bash
python3 qwen3_5_397b.py --spaces problem --output-dir ./outputs
python3 qwen3_5_397b.py --spaces solution --output-dir ./outputs
```

Run only the default domain explicitly, if needed:

```bash
python3 qwen3_5_397b.py \
    --domain "Artificial intelligence and machine learning" \
    --output-dir ./outputs
```

Re-running the same command resumes a partial sweep, but the scripts differ in what counts as "already done":

| Script | Skip rule on re-run |
| --- | --- |
| `gpt5.4.py`, `deepseek_r1_0528.py`, `qwen3_5_397b.py`, `DR_Tulu_eval.py` | Skips whenever the result directory exists, **even if the run inside it crashed**. Delete the directory to retry it. |
| `qwen3_8B_rag.py`, `qwen3_30B_rag.py` | Skips existing directories; pass `--rerun-incomplete` to redo the ones missing a final result, or `--overwrite` to redo everything. |
| `Tongyi_eval.py` | Skips only completed results and automatically reruns incomplete ones. |

Use a new `--output-dir` if you want a fresh run without deleting old outputs.

## Output Layout

Each run writes:

```text
{output-dir}/
  {model_name}/
    artificial_intelligence_and_machine_learning/
      {topic_slug}/
        problem/
          2019_2023/
            response_latest.txt
            response_{timestamp}.txt
            signals_latest.json
            signals_{timestamp}.json
        solution/
          2019_2023/
            response_latest.txt
            response_{timestamp}.txt
            signals_latest.json
            signals_{timestamp}.json
```

`signals_latest.json` stores only the extracted signal names plus metadata. The full raw model output, including `what_it_was` and `why_weak_signal`, is stored in `response_latest.txt`.

Example `signals_latest.json`:

```json
{
  "domain": "Artificial intelligence and machine learning",
  "space": "problem",
  "mainframe_topic": "trustworthy AI",
  "year_range": "2019-2023",
  "timestamp": "20260623T120000Z",
  "signals": ["signal name 1", "signal name 2"]
}
```

## Environment

Install the core dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Extra dependencies for RAG scripts:

```bash
pip install vllm llama-index llama-index-embeddings-huggingface
```

Extra dependencies for the agent scripts: `vllm` (both), `qwen-agent` (Tongyi only), plus whatever the upstream DR-Tulu / Tongyi-DeepResearch repositories require.

The two agent repositories are not vendored in this repo. Clone them and pass the paths via `--dr-tulu-dir` / `--tongyi-dir`:

```bash
git clone https://github.com/rlresearch/dr-tulu.git dr-tulu-main
git clone https://github.com/Alibaba-NLP/DeepResearch.git Tongyi-DeepResearch-main
```

`DR_Tulu_eval.py` requires the clone's `agent/` directory (default `--dr-tulu-dir ./dr-tulu-main`); `Tongyi_eval.py` requires the clone's `inference/` directory and `--tongyi-dir` is mandatory. Both scripts rewrite a few files inside your clone to enforce the 2023 retrieval cutoff, so use a dedicated checkout.

Useful `.env` keys in this directory:

```bash
# OpenAI / GPT-5.4
OPENAI_API_KEY=...

# DeepSeek
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-reasoner

# DashScope / Qwen3.5
DASHSCOPE_API_KEY=...
QWEN_MODEL=qwen3.5-397b-a17b

# Semantic Scholar, used by RAG / DR-Tulu / Tongyi
SEMANTIC_SCHOLAR_API_KEY=...
# or
S2_API_KEY=...
```

Optional overrides read directly from the environment (each also has a CLI flag unless noted):

```bash
RAG_TOP_K=50              # default --rag-top-k for both RAG scripts
QWEN_MODEL_PATH=...       # default --model  for qwen3_8B_rag.py (not QWEN_MODEL)
QWEN_VLLM_HOST=127.0.0.1  # default --host   for the RAG scripts
QWEN_VLLM_PORT=...        # default --port   for the RAG scripts
TONGYI_MODEL_PATH=...     # default --model  for Tongyi_eval.py
TONGYI_VLLM_PORT=...      # default --port   for Tongyi_eval.py
```

> `QWEN_MODEL` (used by `qwen3_5_397b.py` for the DashScope API model name) and `QWEN_MODEL_PATH` (used by `qwen3_8B_rag.py` for the local vLLM checkpoint) are two different variables.

## API Models

### GPT-5.4 through the OpenAI API

```bash
python3 gpt5.4.py --output-dir ./outputs
```

Problem only:

```bash
python3 gpt5.4.py --spaces problem --output-dir ./outputs
```

Main options:

| Argument | Default |
|---|---|
| `--model` / `--openai-model` | `gpt-5.4` |
| `--api-key` / `--openai-api-key` | env `OPENAI_API_KEY` |
| `--temperature` | `1.0` |
| `--max-tokens` | `32768` |
| `--max-retries` | `4` |

### DeepSeek-R1-0528

```bash
python3 deepseek_r1_0528.py --output-dir ./outputs
```

Solution only:

```bash
python3 deepseek_r1_0528.py --spaces solution --output-dir ./outputs
```

Main options:

| Argument | Default |
|---|---|
| `--model` | env `DEEPSEEK_MODEL` or `deepseek-reasoner` |
| `--api-key` | env `DEEPSEEK_API_KEY` |
| `--base-url` | `https://api.deepseek.com/v1` |
| `--temperature` | `1.0` |
| `--max-tokens` | `32768` |
| `--max-retries` | `6` |

### Qwen3.5-397B-A17B through DashScope

```bash
python3 qwen3_5_397b.py --output-dir ./outputs
```

Problem only:

```bash
python3 qwen3_5_397b.py --spaces problem --output-dir ./outputs
```

Main options:

| Argument | Default |
|---|---|
| `--model` | env `QWEN_MODEL` or `qwen3.5-397b-a17b` |
| `--api-key` | env `DASHSCOPE_API_KEY` |
| `--base-url` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `--temperature` | `0.7` |
| `--max-tokens` | `32768` |
| `--max-retries` | `6` |

## RAG Models

Both RAG scripts build their candidate pool with the prediction-window cutoff: for `YEAR_RANGE = 2019-2023`, only papers published `<= 2023` are used. The pool comes from one of two sources, in this order:

1. **Local construction corpus (preferred).** If `construction/papers/<topic-slug>/papers_<topic-slug>_<year>.parquet` exists, those pre-fetched pools are used directly (`--papers-dir`, default `../construction/papers`). This is the path used to produce the released results, and it is what running the construction stage first gives you.
2. **Live Semantic Scholar fallback.** Only if that directory is missing, the scripts query Semantic Scholar at run time. Here `--retrieval-queries` matters: when omitted, `qwen3_8B_rag.py` queries with the topic name plus its paraphrases from `construction/topics.json` — the same query set the construction stage uses — while `qwen3_30B_rag.py` queries with the topic name only, giving it a narrower pool than the local corpus would.

Either way the pool is then embedded with `BAAI/bge-large-en-v1.5`, ranked by cosine similarity against the target topic, and truncated to `--rag-top-k` (default `50`, also settable via the `RAG_TOP_K` environment variable).

### Qwen3-8B RAG

Auto-start vLLM:

```bash
python3 qwen3_8B_rag.py --output-dir ./outputs
```

Reuse an already running vLLM server:

```bash
python3 qwen3_8B_rag.py --output-dir ./outputs --skip-vllm-start
```

Custom retrieval queries:

```bash
python3 qwen3_8B_rag.py \
    --output-dir ./outputs \
    --retrieval-queries "trustworthy AI" "AI safety" "model robustness"
```

Main options:

| Argument | Default |
|---|---|
| `--qwen-model` | env `QWEN_MODEL_PATH` or `Qwen/Qwen3-8B` |
| `--vllm-port` | env `QWEN_VLLM_PORT` or `6003` |
| `--tensor-parallel` | `1` |
| `--max-model-len` | `32768` |
| `--gpu-memory-utilization` | `0.7` |
| `--rag-top-k` | env `RAG_TOP_K` or `50` |
| `--s2-max-total` | `10000` |

### Qwen3-30B AWQ RAG

Auto-start vLLM:

```bash
python3 qwen3_30B_rag.py --output-dir ./outputs
```

Reuse an already running vLLM server:

```bash
python3 qwen3_30B_rag.py --output-dir ./outputs --skip-vllm-start
```

Run vLLM and prediction in separate terminals:

Use this when one terminal should keep the Qwen3-30B server alive and another terminal should run the prediction script against that server. Both terminals must be on the same GPU node or Slurm allocation because `127.0.0.1` is node-local.

Terminal 1: start the Qwen3-30B vLLM server:

```bash
cd prediction

python -m vllm.entrypoints.openai.api_server \
    --model stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --host 127.0.0.1 \
    --port 6004 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
```

Optional readiness check:

```bash
curl http://127.0.0.1:6004/v1/models
```

Terminal 2: run the prediction script and reuse the existing server:

```bash
cd prediction

python qwen3_30B_rag.py \
    --output-dir ./outputs_topk/k30 \
    --skip-vllm-start \
    --vllm-port 6004 \
    --rag-top-k 30
```

If using a Slurm interactive allocation, attach both terminals to the same job before running these commands:

```bash
srun --jobid <JOBID> --pty bash -l
```

Main options:

| Argument | Default |
|---|---|
| `--qwen-model` | env `QWEN_MODEL_PATH` or `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` |
| `--vllm-port` | env `QWEN_VLLM_PORT` or `6004` |
| `--tensor-parallel` | `1` |
| `--max-model-len` | `32768` |
| `--gpu-memory-utilization` | `0.9` |
| `--rag-top-k` | env `RAG_TOP_K` or `50` |
| `--s2-max-total` | `10000` |

## Agent Models

> **Note:** both agent scripts patch a few files inside your local clone of the upstream repository at startup (to enforce the retrieval-year cutoff and the prediction prompt). The originals are saved next to each patched file as `*.py.bak`. Use a dedicated clone for these runs.

### DR-Tulu

Requires a local clone of the DR-Tulu repository containing `agent/`.

```bash
python3 DR_Tulu_eval.py \
    --output-dir ./outputs \
    --dr-tulu-dir /path/to/dr-tulu-main
```

Problem only:

```bash
python3 DR_Tulu_eval.py \
    --spaces problem \
    --output-dir ./outputs \
    --dr-tulu-dir /path/to/dr-tulu-main
```

Split-terminal launch inside the same Slurm allocation:

Terminal 1 starts the vLLM model server:

```bash
srun --jobid <JOBID> --overlap --pty bash -l

cd <path-to-dr-tulu>/agent

python -m vllm.entrypoints.openai.api_server \
    --model rl-research/DR-Tulu-8B \
    --host 127.0.0.1 \
    --port 30001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.7
```

Terminal 2 runs the prediction script after the model server is ready:

```bash
srun --jobid <JOBID> --overlap --pty bash -l

cd prediction

until curl -fsS http://127.0.0.1:30001/v1/models >/dev/null; do
    sleep 5
done

python DR_Tulu_eval.py \
    --output-dir ./outputs \
    --dr-tulu-dir <path-to-dr-tulu> \
    --vllm-port 30001
```

Main options:

| Argument | Default |
|---|---|
| `--dr-tulu-model` | `rl-research/DR-Tulu-8B` |
| `--dr-tulu-port` | `8080` |
| `--vllm-port` | `30001` |
| `--mcp-port` | `8000` |
| `--request-timeout` | `1800` |
| `--gpu-memory-utilization` | `0.7` |
| `--max-model-len` | `32768` |

Signal extraction includes a fallback for DR-Tulu's occasional curly-quote / broken-JSON responses.

### Tongyi DeepResearch

Requires a local clone of Tongyi-DeepResearch containing `inference/`.

```bash
python3 Tongyi_eval.py \
    --output-dir ./outputs \
    --tongyi-dir /path/to/Tongyi-DeepResearch-main
```

Reuse an already running vLLM server:

```bash
 GOOGLE_SCHOLAR_TOOL_BUDGET=3 python Tongyi_eval.py \
    --output-dir ./outputs \
    --tongyi-dir <path-to-tongyi-deepresearch> \
    --skip-vllm-start
```

Split-terminal launch inside the same Slurm allocation:

Use this when one terminal should keep the Tongyi vLLM server alive and another terminal should run the prediction script against that server. Both terminals must be on the same GPU node or Slurm allocation because `127.0.0.1` is node-local.

Terminal 1 starts the Tongyi vLLM model server:

```bash
cd prediction

python -m vllm.entrypoints.openai.api_server \
    --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    --host 127.0.0.1 \
    --port 6001 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
```

Optional readiness check:

```bash
curl http://127.0.0.1:6001/v1/models
```

Terminal 2 runs the prediction script after the model server is ready:

```bash
cd prediction

GOOGLE_SCHOLAR_TOOL_BUDGET=8 python Tongyi_eval.py \
      --output-dir ./outputs_b/b8 \
      --tongyi-dir <path-to-tongyi-deepresearch> \
      --skip-vllm-start \
      --vllm-port 6001
```

If using a Slurm interactive allocation, attach both terminals to the same job before running these commands:

```bash
srun --jobid <JOBID> --overlap --pty bash -l
```

Main options:

| Argument | Default |
|---|---|
| `--tongyi-model` | env `TONGYI_MODEL_PATH` or `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` |
| `--vllm-port` | env `TONGYI_VLLM_PORT` or `6001` |
| `--tensor-parallel` | `2` |
| `--max-model-len` | `32768` |
| `--gpu-memory-utilization` | `0.9` |
| `--temperature` | `0.7` |
| `--top-p` | `0.95` |
| `--presence-penalty` | `1.1` |

## Quick Checklist Before Running

1. Confirm `.env` contains the API keys required by the selected script.
2. Omit `--domain` unless you have a specific reason to pass the default domain name.
3. Use `--spaces problem` or `--spaces solution` for a smaller run.
4. Use a fresh `--output-dir` when you want to rerun predictions that already exist.
5. For RAG / agent scripts, confirm `SEMANTIC_SCHOLAR_API_KEY` or `S2_API_KEY` is set.
