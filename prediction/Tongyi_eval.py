#!/usr/bin/env python
# coding: utf-8
"""
Tongyi DeepResearch -- weak-signal prediction (prediction only, no evaluation).

Uses the Tongyi-DeepResearch inference framework (ReAct agent with Semantic
Scholar tool) via a local vLLM server.

Requires: the Tongyi-DeepResearch repo cloned locally (--tongyi-dir).

Usage example:
    python Tongyi_eval.py \
        --spaces problem solution \
        --domain "Artificial intelligence and machine learning" \
        --output-dir ./outputs \
        --tongyi-dir /path/to/Tongyi-DeepResearch-main
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from prediction_prompts import (
    YEAR_RANGE,
    YEAR_SLUG,
    build_prompt,
    extract_candidate_signals,
    load_weak_signal_json,
    make_topic_slug,
)


def year_range_to_cutoff(year_range: str) -> int:
    """Use the end of the prediction window as the S2 evidence cutoff."""
    return int(year_range.split("-")[-1])


# ---------------------------------------------------------------------------
# Tongyi repo patching
# ---------------------------------------------------------------------------

def patch_tongyi_repo(tongyi_dir: Path) -> None:
    """Patch Tongyi inference files to use Semantic Scholar with year cutoff."""
    infer_dir = tongyi_dir / "inference"

    # --- 1) Patch tool_scholar.py ---
    tool_file = infer_dir / "tool_scholar.py"
    if tool_file.exists():
        src = tool_file.read_text(encoding="utf-8")
        if "STEPB6_S2_PATCH" in src:
            print("tool_scholar.py: already patched.")
        else:
            if not tool_file.with_suffix(".py.bak").exists():
                tool_file.with_suffix(".py.bak").write_text(src, encoding="utf-8")
            patched = r'''# STEPB6_S2_PATCH
import os
import json
import requests
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor
from qwen_agent.tools.base import BaseTool, register_tool

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_ENDPOINT = f"{S2_BASE_URL}/paper/search"

@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Semantic Scholar search with hard year cutoff. Accepts multiple queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries.",
            },
        },
        "required": ["query"],
    }

    def _search_one(self, query: str) -> str:
        if not S2_API_KEY:
            return "Semantic Scholar API key missing. Set SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY."
        cutoff_year = int(os.getenv("S2_CUTOFF_YEAR", "9999"))
        headers = {"x-api-key": S2_API_KEY}
        params = {
            "query": query,
            "limit": 10,
            "fields": "title,year,abstract,url,venue,authors,citationCount",
        }
        try:
            resp = requests.get(S2_ENDPOINT, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            return f"S2 request failed for query '{query}': {e}"

        papers = payload.get("data", []) or []
        filtered = [p for p in papers if isinstance(p.get("year"), int) and p["year"] <= cutoff_year]
        if not filtered:
            return f"S2 search for '{query}' found no papers at or before {cutoff_year}."

        lines = []
        for i, p in enumerate(filtered, start=1):
            abstract = (p.get("abstract") or "").strip()
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            lines.append(
                f"{i}. Title: {p.get('title', 'Untitled')}\n"
                f"   Year: {p.get('year')}\n"
                f"   Venue: {p.get('venue', 'N/A')}\n"
                f"   Citations: {p.get('citationCount')}\n"
                f"   URL: {p.get('url', 'N/A')}\n"
                f"   Abstract: {abstract}"
            )
        return (
            f"Semantic Scholar results for '{query}' "
            f"(year <= {cutoff_year}, returned {len(filtered)} items):\n\n"
            + "\n\n".join(lines)
        )

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except Exception:
            return '[google_scholar] Invalid request format: expected {"query": [...]}'
        if isinstance(query, str):
            return self._search_one(query)
        assert isinstance(query, List)
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self._search_one, query))
        return "\n=======\n".join(results)
'''
            tool_file.write_text(patched, encoding="utf-8")
            print("tool_scholar.py: patched to Semantic Scholar + cutoff.")
    else:
        print(f"[warn] tool_scholar.py not found at {tool_file}")

    # --- 2) Patch prompt.py ---
    prompt_file = infer_dir / "prompt.py"
    if prompt_file.exists():
        p_src = prompt_file.read_text(encoding="utf-8")
        if "STEPB6_S2_ONLY_PROMPT_PATCH" not in p_src:
            if not prompt_file.with_suffix(".py.bak").exists():
                prompt_file.with_suffix(".py.bak").write_text(p_src, encoding="utf-8")
            new_system_prompt = '''SYSTEM_PROMPT = """You are a deep research assistant.
You must use ONLY the provided tools. For retrieval, use google_scholar (mapped to Semantic Scholar API with year cutoff).
Do not use generic web search tools.

When you have gathered sufficient information and are ready to provide the definitive response, enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "google_scholar", "description": "Semantic Scholar search with hard year cutoff. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "parse_file", "description": "Parse user uploaded local files.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "description": "Uploaded files to parse."}}, "required": ["files"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """
# STEPB6_S2_ONLY_PROMPT_PATCH
'''
            start = p_src.find('SYSTEM_PROMPT = """')
            end = p_src.find('EXTRACTOR_PROMPT = """')
            if start != -1 and end != -1 and end > start:
                p_src = p_src[:start] + new_system_prompt + "\n\n" + p_src[end:]
                prompt_file.write_text(p_src, encoding="utf-8")
                print("prompt.py: patched.")
            else:
                print("[warn] prompt.py: could not locate boundaries for patching.")
        else:
            print("prompt.py: already patched.")

    # --- 3) Patch react_agent.py ---
    agent_file = infer_dir / "react_agent.py"
    if agent_file.exists():
        a_src = agent_file.read_text(encoding="utf-8")
        if "STEPB6_S2_ONLY_AGENT_PATCH" not in a_src:
            if not agent_file.with_suffix(".py.bak").exists():
                agent_file.with_suffix(".py.bak").write_text(a_src, encoding="utf-8")
            pattern = r"TOOL_CLASS\s*=\s*\[[\s\S]*?\]\s*TOOL_MAP\s*=\s*\{tool\.name:\s*tool\s*for\s*tool\s*in\s*TOOL_CLASS\}"
            replacement = (
                "TOOL_CLASS = [\n    FileParser(),\n    Scholar(),\n    PythonInterpreter(),\n]\n"
                "# STEPB6_S2_ONLY_AGENT_PATCH\n"
                "TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}"
            )
            new_a_src, n = re.subn(pattern, replacement, a_src, count=1)
            if n == 1:
                agent_file.write_text(new_a_src, encoding="utf-8")
                print("react_agent.py: patched.")
            else:
                print("[warn] react_agent.py: TOOL_CLASS/TOOL_MAP block not found.")
        else:
            print("react_agent.py: already patched.")

        a_src = agent_file.read_text(encoding="utf-8")
        if "STEPB6_OPTIONAL_MAX_OUTPUT_TOKENS_PATCH" not in a_src:
            patched = a_src
            old_budget = (
                "GOOGLE_SCHOLAR_TOOL_BUDGET = int(os.getenv('GOOGLE_SCHOLAR_TOOL_BUDGET', '-1'))\n"
                "# STEPB6_GS_BUDGET_PATCH"
            )
            new_budget = (
                "GOOGLE_SCHOLAR_TOOL_BUDGET = int(os.getenv('GOOGLE_SCHOLAR_TOOL_BUDGET', '-1'))\n"
                "\n"
                "def _optional_positive_int_env(name):\n"
                "    raw = os.getenv(name)\n"
                "    if raw is None or raw.strip() == '':\n"
                "        return None\n"
                "    value = int(raw)\n"
                "    if value <= 0:\n"
                "        raise ValueError(f'{name} must be a positive integer when set')\n"
                "    return value\n"
                "\n"
                "TONGYI_MAX_OUTPUT_TOKENS = _optional_positive_int_env('TONGYI_MAX_OUTPUT_TOKENS')\n"
                "TONGYI_CONTEXT_LIMIT = int(os.getenv('TONGYI_CONTEXT_LIMIT', '32768'))\n"
                "# STEPB6_OPTIONAL_MAX_OUTPUT_TOKENS_PATCH\n"
                "# STEPB6_GS_BUDGET_PATCH"
            )
            if old_budget in patched:
                patched = patched.replace(old_budget, new_budget, 1)
            else:
                print("[warn] react_agent.py: could not locate budget block for max-token patch.")

            old_call = """                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )"""
            new_call = """                request_kwargs = {
                    "model": self.model,
                    "messages": msgs,
                    "stop": ["\\n<tool_response>", "<tool_response>"],
                    "temperature": self.llm_generate_cfg.get('temperature', 0.6),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
                    "logprobs": True,
                    "presence_penalty": self.llm_generate_cfg.get('presence_penalty', 1.1),
                }
                if TONGYI_MAX_OUTPUT_TOKENS is not None:
                    request_kwargs["max_tokens"] = TONGYI_MAX_OUTPUT_TOKENS
                chat_response = client.chat.completions.create(**request_kwargs)"""
            if old_call in patched:
                patched = patched.replace(old_call, new_call, 1)
            else:
                print("[warn] react_agent.py: could not locate chat completion block for max-token patch.")

            old_limit = """            max_tokens = 32768
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")"""
            new_limit = """            context_threshold = TONGYI_CONTEXT_LIMIT
            if TONGYI_MAX_OUTPUT_TOKENS is not None:
                context_threshold = max(1, TONGYI_CONTEXT_LIMIT - TONGYI_MAX_OUTPUT_TOKENS)
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > context_threshold:
                print(f"Token quantity exceeds the limit: {token_count} > {context_threshold}")"""
            if old_limit in patched:
                patched = patched.replace(old_limit, new_limit, 1)
            else:
                print("[warn] react_agent.py: could not locate context threshold block for max-token patch.")

            if patched != a_src:
                agent_file.write_text(patched, encoding="utf-8")
                print("react_agent.py: patched optional max output tokens.")
        else:
            print("react_agent.py: optional max output tokens already patched.")

    # --- 4) Patch run_multi_react.py ---
    runner_file = infer_dir / "run_multi_react.py"
    if runner_file.exists():
        r_src = runner_file.read_text(encoding="utf-8")
        if "STEPB6_S2_ONLY_RUNNER_PATCH" not in r_src:
            if not runner_file.with_suffix(".py.bak").exists():
                runner_file.with_suffix(".py.bak").write_text(r_src, encoding="utf-8")
            old = 'function_list=["search", "visit", "google_scholar", "PythonInterpreter"]'
            new = 'function_list=["google_scholar", "PythonInterpreter"]  # STEPB6_S2_ONLY_RUNNER_PATCH'
            if old in r_src:
                r_src = r_src.replace(old, new, 1)
                runner_file.write_text(r_src, encoding="utf-8")
                print("run_multi_react.py: patched.")
            else:
                regex_old = r'function_list\s*=\s*\[[^\]]*google_scholar[^\]]*\]'
                new_r_src, n = re.subn(
                    regex_old,
                    'function_list=["google_scholar", "PythonInterpreter"]  # STEPB6_S2_ONLY_RUNNER_PATCH',
                    r_src, count=1,
                )
                if n == 1:
                    runner_file.write_text(new_r_src, encoding="utf-8")
                    print("run_multi_react.py: patched via regex.")
                else:
                    print("[warn] run_multi_react.py: function_list pattern not found.")
        else:
            print("run_multi_react.py: already patched.")


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_server(
    model: str,
    host: str,
    port: int,
    tensor_parallel: int = 2,
    max_model_len: int = 32768,
    gpu_mem: float = 0.9,
) -> subprocess.Popen | None:
    """Start vLLM for Tongyi model if not already running."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            print(f"vLLM already running on {host}:{port}. Reusing.")
            return None

    log_path = Path(f"tongyi_vllm_server_{port}.log")
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
    ]

    print(f"Starting vLLM on {host}:{port} with model {model} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"Spawned vLLM pid={proc.pid}, log={log_path}")

    # Wait for readiness (up to 20 min)
    import requests as http_requests
    max_wait, interval, waited = 1200, 5, 0
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        try:
            r = http_requests.get(f"http://{host}:{port}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"vLLM ready after {waited}s")
                return proc
        except Exception:
            print(f"  Waiting for vLLM ... ({waited}s elapsed)", flush=True)

    print(f"[warn] vLLM may not be ready after {max_wait}s. Check {log_path}")
    return proc


# ---------------------------------------------------------------------------
# Tongyi inference (via run_multi_react.py subprocess)
# ---------------------------------------------------------------------------

def run_tongyi_inference(
    infer_dir: Path,
    model: str,
    prompt: str,
    output_base: Path,
    cutoff_year: int,
    s2_api_key: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    presence_penalty: float = 1.1,
    max_output_tokens: int | None = None,
) -> str:
    """Run Tongyi ReAct inference and return the prediction text."""
    output_base = Path(output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_name = f"tongyi_input_{run_tag}.json"
    dataset_path = output_base / dataset_name

    dataset_items = [{"question": prompt, "answer": ""}]
    dataset_path.write_text(
        json.dumps(dataset_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    env = os.environ.copy()
    env["S2_CUTOFF_YEAR"] = str(cutoff_year)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("TONGYI_MAX_OUTPUT_TOKENS", None)
    if max_output_tokens is not None:
        env["TONGYI_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    if s2_api_key:
        env["SEMANTIC_SCHOLAR_API_KEY"] = s2_api_key
        env["S2_API_KEY"] = s2_api_key

    runner_output = output_base / "tongyi_runner_outputs"
    runner_output.mkdir(parents=True, exist_ok=True)

    run_script = infer_dir / "run_multi_react.py"
    cmd = [
        sys.executable, "-B", str(run_script),
        "--model", model,
        "--output", str(runner_output),
        "--dataset", dataset_name,
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--presence_penalty", str(presence_penalty),
        "--max_workers", "1",
        "--roll_out_count", "1",
        "--total_splits", "1",
        "--worker_split", "1",
    ]

    print(f"  Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(output_base), env=env, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"  stdout (last 40 lines):\n{chr(10).join(res.stdout.splitlines()[-40:])}")
        print(f"  stderr (last 40 lines):\n{chr(10).join(res.stderr.splitlines()[-40:])}")
        raise RuntimeError("run_multi_react.py failed. See logs above.")

    # Find iter1.jsonl output
    model_leaf = os.path.basename(model.rstrip("/"))
    iter1_path = runner_output / f"{model_leaf}_sglang" / dataset_name / "iter1.jsonl"
    if not iter1_path.exists():
        candidates = sorted(
            runner_output.glob("**/iter1.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("iter1.jsonl not found under runner output dir.")
        iter1_path = candidates[0]

    rows = [ln for ln in iter1_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {iter1_path}")

    last = json.loads(rows[-1])
    return (last.get("prediction") or "").strip()


# ---------------------------------------------------------------------------
# Completion checks
# ---------------------------------------------------------------------------

def _has_unclosed_tag(text: str, tag: str) -> bool:
    opens = len(re.findall(fr"<{tag}\b[^>]*>", text, flags=re.IGNORECASE))
    closes = len(re.findall(fr"</{tag}\s*>", text, flags=re.IGNORECASE))
    return opens > closes


def validate_response(response_text: str, signals: List[str] | None = None) -> List[str]:
    """Return reasons this response should not be treated as complete."""
    reasons: List[str] = []
    if not response_text or not response_text.strip():
        return ["empty_response"]
    for tag in ("think", "tool_call"):
        if _has_unclosed_tag(response_text, tag):
            reasons.append(f"unclosed_{tag}")
    payload = load_weak_signal_json(response_text)
    if payload is None:
        reasons.append("missing_final_weak_signals_json")
    elif not extract_candidate_signals(response_text):
        reasons.append("empty_final_weak_signals_json")
    if signals is not None and not signals:
        reasons.append("empty_extracted_signals")
    return reasons


def load_saved_signals(signals_path: Path) -> List[str] | None:
    try:
        data = json.loads(signals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    signals = data.get("signals")
    if not isinstance(signals, list):
        return None
    return [str(s).strip() for s in signals if str(s).strip()]


def result_is_complete(result_dir: Path) -> bool:
    response_path = result_dir / "response_latest.txt"
    signals_path = result_dir / "signals_latest.json"
    if not response_path.exists() or not signals_path.exists():
        return False
    try:
        response_text = response_path.read_text(encoding="utf-8")
    except OSError:
        return False
    saved_signals = load_saved_signals(signals_path)
    if not saved_signals:
        return False
    return not validate_response(response_text, saved_signals)


def _normalized_path_text(path: Path) -> str:
    text = path.as_posix().rstrip("/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _target_prefix_matches(entry_prefix: str, output_dir: Path) -> bool:
    entry = entry_prefix.strip().rstrip("/")
    while entry.startswith("./"):
        entry = entry[2:]
    current = _normalized_path_text(output_dir / "tongyi")
    return current == entry or current.endswith(f"/{entry}") or entry.endswith(f"/{current}")


def load_target_filter(target_file: Path | None, output_dir: Path) -> set[tuple[str, str]] | None:
    """Load topic/space pairs for this output-dir from a rerun_targets.tsv file."""
    if target_file is None:
        return None
    targets: set[tuple[str, str]] = set()
    lines = target_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        return targets
    header = lines[0].split("\t")
    required = {"prefix", "topic", "space"}
    if not required.issubset(header):
        raise ValueError(f"{target_file} must contain columns: {sorted(required)}")
    col = {name: header.index(name) for name in required}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) <= max(col.values()):
            continue
        if not _target_prefix_matches(parts[col["prefix"]], output_dir):
            continue
        targets.add((parts[col["topic"]], parts[col["space"]]))
    return targets


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    space: str,
    domain: str,
    topic: str,
    response_text: str,
    signals: List[str],
) -> Path:
    from mainframe_topics import make_domain_slug

    topic_slug = make_topic_slug(topic)
    domain_slug = make_domain_slug(domain)
    result_dir = output_dir / "tongyi" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")

    payload = {
        "space": space,
        "domain": domain,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "signals": signals,
    }
    (result_dir / f"signals_{timestamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "signals_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_dir


def save_incomplete_result(
    output_dir: Path,
    space: str,
    domain: str,
    topic: str,
    response_text: str,
    signals: List[str],
    reasons: List[str],
) -> Path:
    from mainframe_topics import make_domain_slug

    topic_slug = make_topic_slug(topic)
    domain_slug = make_domain_slug(domain)
    result_dir = output_dir / "tongyi" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (result_dir / f"incomplete_response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "incomplete_response_latest.txt").write_text(response_text, encoding="utf-8")

    payload = {
        "space": space,
        "domain": domain,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "signals": signals,
        "incomplete_reasons": reasons,
    }
    (result_dir / f"incomplete_{timestamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "incomplete_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_dir


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tongyi DeepResearch weak-signal prediction (no evaluation)."
    )
    p.add_argument("--spaces", nargs="+", default=["problem", "solution"],
                   choices=["problem", "solution"])
    p.add_argument("--domain", nargs="+", default=None,
                   help="Domains to predict. If omitted, uses all domains.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tongyi-dir", required=True, type=Path,
                   help="Path to Tongyi-DeepResearch repo (contains inference/ directory).")
    p.add_argument("--target-file", type=Path, default=None,
                   help="Optional rerun_targets.tsv; only matching topic/space rows for this output-dir are run.")

    p.add_argument("--tongyi-model",
                   default=os.getenv("TONGYI_MODEL_PATH", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"))
    p.add_argument("--vllm-host", default="127.0.0.1")
    p.add_argument("--vllm-port", type=int, default=int(os.getenv("TONGYI_VLLM_PORT", "6001")))
    p.add_argument("--tensor-parallel", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--skip-vllm-start", action="store_true")

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--presence-penalty", type=float, default=1.1)
    p.add_argument("--max-output-tokens", "--max-tokens", dest="max_output_tokens",
                   type=int, default=None,
                   help="Optional per-call max_tokens for Tongyi/vLLM. If omitted, no max_tokens is sent.")

    p.add_argument("--s2-api-key",
                   default=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", ""))

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from mainframe_topics import TOPICS_BY_DOMAIN, ALL_DOMAINS, make_domain_slug

    args = parse_args()

    domains = args.domain if args.domain else ALL_DOMAINS
    random.seed(args.seed)
    target_filter = load_target_filter(args.target_file, args.output_dir)

    tongyi_dir = Path(args.tongyi_dir)
    infer_dir = tongyi_dir / "inference"

    print("=" * 60)
    print(f"Spaces:      {args.spaces}")
    print(f"Domains:     {domains}")
    print(f"Year range:  {YEAR_RANGE}")
    print(f"Model:       {args.tongyi_model}")
    print(f"Tongyi dir:  {tongyi_dir}")
    print(f"Output dir:  {args.output_dir}")
    if target_filter is not None:
        print(f"Target file: {args.target_file} ({len(target_filter)} matching topic/space pairs)")
    print("=" * 60)

    # Patch Tongyi repo
    print("\nPatching Tongyi repo ...")
    patch_tongyi_repo(tongyi_dir)

    # Start vLLM
    if not args.skip_vllm_start:
        start_vllm_server(
            model=args.tongyi_model,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_mem=args.gpu_memory_utilization,
        )

    for domain in domains:
        domain_slug = make_domain_slug(domain)
        topics = TOPICS_BY_DOMAIN[domain]
        for topic in topics:
            for space in args.spaces:
                print(f"\n{'─' * 60}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'─' * 60}")

                topic_slug = make_topic_slug(topic)
                if target_filter is not None and (topic_slug, space) not in target_filter:
                    continue
                result_dir = args.output_dir / "tongyi" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_is_complete(result_dir):
                    print(f"[skip] Complete result exists: {result_dir}")
                    continue
                if result_dir.exists():
                    print(f"[rerun] Incomplete existing result: {result_dir}")

                cutoff_year = year_range_to_cutoff(YEAR_RANGE)
                print(f"S2 cutoff year: {cutoff_year}")

                prompt = build_prompt(space, domain, topic)
                print(f"Prompt length: {len(prompt)} chars")

                # Run Tongyi inference
                run_output_base = (
                    args.output_dir / "tongyi" / domain_slug / topic_slug / space / YEAR_SLUG / "tongyi_runs"
                )

                print("Running Tongyi ReAct inference ...")
                response_text = run_tongyi_inference(
                    infer_dir=infer_dir,
                    model=args.tongyi_model,
                    prompt=prompt,
                    output_base=run_output_base,
                    cutoff_year=cutoff_year,
                    s2_api_key=args.s2_api_key,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    max_output_tokens=args.max_output_tokens,
                )
                print(f"Response length: {len(response_text)} chars")

                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                invalid_reasons = validate_response(response_text, signals)
                if invalid_reasons:
                    reason_text = ", ".join(invalid_reasons)
                    result_dir = save_incomplete_result(
                        args.output_dir, space, domain, topic,
                        response_text, signals, invalid_reasons,
                    )
                    print(f"[incomplete] Not saving normal result: {reason_text}")
                    print(f"Incomplete artifacts saved to: {result_dir}")
                    continue

                result_dir = save_results(
                    args.output_dir, space, domain, topic,
                    response_text, signals,
                )
                print(f"Results saved to: {result_dir}")

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
