#!/usr/bin/env python
# coding: utf-8
"""
DR-Tulu – weak-signal prediction (prediction only, no evaluation).

Usage example:
    python DR_Tulu_eval.py \
        --spaces problem solution \
        --domain "Artificial intelligence and machine learning" \
        --output-dir ./outputs \
        --dr-tulu-dir /path/to/dr-tulu-main
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DR_TULU_DIR = REPO_ROOT / "dr-tulu-main"

from prediction_prompts import (
    YEAR_RANGE,
    YEAR_SLUG,
    build_prompt,
    extract_candidate_signals,
    make_topic_slug,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clean_signal(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,.;:")


def _fallback_extract_signals(response_text: str) -> List[str]:
    """Extract signal names from DR-Tulu's invalid-JSON style responses.

    prediction_prompts.extract_candidate_signals handles normal JSON; DR-Tulu
    sometimes emits curly quotes or otherwise broken JSON. This fallback is
    deliberately narrow: it only targets fields named "signal" and stops at
    the next quote.
    """
    signals: List[str] = []
    patterns = [
        r'["“”]signal["“”]\s*:\s*["“”]([^"“”]+)["“”]',
        r"'signal'\s*:\s*'([^']+)'",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, response_text, flags=re.IGNORECASE):
            signal = _clean_signal(match.group(1))
            if signal:
                signals.append(signal)

    deduped: List[str] = []
    seen: set[str] = set()
    for signal in signals:
        if signal not in seen:
            seen.add(signal)
            deduped.append(signal)
    return deduped


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether a TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 300, interval: int = 5) -> None:
    """Block until a TCP port is accepting connections, or raise after timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if port_open(port, host):
            return
        print(f"  Waiting for port {port} ... ({int(time.time() - start)}s elapsed)")
        time.sleep(interval)
    raise RuntimeError(f"Port {port} not ready after {timeout}s. Check the service logs.")


def clean_python_env() -> dict[str, str]:
    """Return subprocess env without inherited Python import-path pollution."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def year_range_to_cutoff(year_range: str) -> int:
    """Derive the S2 cutoff year from a year range like '2020-2022'.

    The cutoff is set to the end of the prediction window, so that DR-Tulu
    can only see papers published at or before the window end year.
    """
    return int(year_range.split("-")[-1])



# ---------------------------------------------------------------------------
# Semantic Scholar cutoff patch
# ---------------------------------------------------------------------------

def patch_semantic_scholar_cutoff(dr_tulu_dir: Path) -> Path:
    """Patch semantic_scholar_apis.py to enforce S2_CUTOFF_YEAR env var."""
    s2_file = (
        dr_tulu_dir / "agent" / "dr_agent" / "mcp_backend"
        / "apis" / "semantic_scholar_apis.py"
    )
    if not s2_file.exists():
        raise FileNotFoundError(
            f"semantic_scholar_apis.py not found at {s2_file}; "
            "refusing to start unpatched DR-Tulu service"
        )

    backup = s2_file.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(s2_file.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {backup}")

    src = s2_file.read_text(encoding="utf-8")

    if "S2_CUTOFF_YEAR" in src and 'query_params.year = f"-' in src:
        print("S2 cutoff already patched. Skipping.")
        return s2_file

    marker = "    params = query_params.model_dump(exclude_none=True)\n"
    inject = (
        "    # Enforce cutoff year if set\n"
        "    s2_cutoff_year = os.getenv(\"S2_CUTOFF_YEAR\")\n"
        "    if s2_cutoff_year:\n"
        "        try:\n"
        "            _y = int(s2_cutoff_year)\n"
        "            query_params.year = f\"-{_y}\"\n"
        "        except Exception:\n"
        "            pass\n"
    )

    if marker in src:
        src = src.replace(marker, inject + marker)
        if "import os" not in src:
            src = src.replace(
                "from pydantic import", "import os\nfrom pydantic import"
            )
        s2_file.write_text(src, encoding="utf-8")
        print("S2 cutoff patch applied.")
        return s2_file
    else:
        raise RuntimeError(
            f"Patch marker not found in {s2_file}; "
            "refusing to start unpatched DR-Tulu service"
        )


# ---------------------------------------------------------------------------
# Service management
# ---------------------------------------------------------------------------

def start_mcp_server(
    agent_dir: Path, mcp_port: int, cutoff_year: int
) -> subprocess.Popen | None:
    """Start the MCP (Semantic Scholar) server if not already running."""
    if port_open(mcp_port):
        print(f"MCP server already running on port {mcp_port}. Reusing.")
        return None

    env = clean_python_env()
    env["S2_API_KEY"] = (
        os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or ""
    )
    env["S2_CUTOFF_YEAR"] = str(cutoff_year)

    proc = subprocess.Popen(
        [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(mcp_port)],
        cwd=str(agent_dir),
        env=env,
        stdout=open("mcp_server.log", "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"MCP server launching (pid={proc.pid}), waiting for port {mcp_port} ...")
    wait_for_port(mcp_port, timeout=120)
    print(f"MCP server ready on port {mcp_port}")
    return proc


def start_vllm_server(
    agent_dir: Path,
    model: str,
    vllm_port: int,
    gpu_mem: float,
    max_model_len: int,
) -> subprocess.Popen | None:
    """Start the vLLM OpenAI-compatible server if not already running."""
    if port_open(vllm_port):
        print(f"vLLM server already running on port {vllm_port}. Reusing.")
        return None

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", "127.0.0.1",
            "--port", str(vllm_port),
            "--dtype", "bfloat16",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_mem),
        ],
        cwd=str(Path(__file__).resolve().parent),
        env=clean_python_env(),
        stdout=open("vllm_server.log", "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"vLLM server launching (pid={proc.pid}), waiting for port {vllm_port} ...")
    wait_for_port(vllm_port, timeout=300)
    print(f"vLLM server ready on port {vllm_port}")
    return proc


def start_dr_tulu_service(
    agent_dir: Path,
    dr_tulu_port: int,
    model: str,
    vllm_port: int,
) -> subprocess.Popen | None:
    """Start the DR-Tulu service with config overrides."""
    if port_open(dr_tulu_port):
        print(f"DR-Tulu service already running on port {dr_tulu_port}. Reusing.")
        return None

    config_overrides = ",".join([
        "search_tool_name=s2-only",
        f"search_agent_model_name={model}",
        f"search_agent_base_url=http://127.0.0.1:{vllm_port}/v1",
        "search_agent_api_key=dummy-key",
        "use_browse_agent=false",
    ])

    proc = subprocess.Popen(
        [
            sys.executable, "workflows/auto_search_sft.py",
            "serve",
            "--port", str(dr_tulu_port),
            "--host", "127.0.0.1",
            "--config-overrides", config_overrides,
            "--verbose",
        ],
        cwd=str(agent_dir),
        env=clean_python_env(),
        stdout=open("dr_tulu_service.log", "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"DR-Tulu service launching (pid={proc.pid}), waiting for port {dr_tulu_port} ...")
    wait_for_port(dr_tulu_port, timeout=120)
    print(f"DR-Tulu service ready on port {dr_tulu_port}")
    return proc


def start_all_services(args: argparse.Namespace, cutoff_year: int) -> None:
    """Patch S2 and start MCP + vLLM + DR-Tulu services."""
    dr_tulu_dir = Path(args.dr_tulu_dir)
    agent_dir = dr_tulu_dir / "agent"
    if not agent_dir.is_dir():
        raise FileNotFoundError(
            f"DR-Tulu agent directory not found at {agent_dir}; "
            "pass --dr-tulu-dir pointing to a dr-tulu-main checkout"
        )

    patched_file = patch_semantic_scholar_cutoff(dr_tulu_dir)
    print(f"Using patched DR-Tulu checkout: {dr_tulu_dir}")
    print(f"Patched S2 API file: {patched_file}")

    start_mcp_server(agent_dir, args.mcp_port, cutoff_year)
    start_vllm_server(
        agent_dir, args.dr_tulu_model, args.vllm_port,
        args.gpu_memory_utilization, args.max_model_len,
    )
    start_dr_tulu_service(
        agent_dir, args.dr_tulu_port, args.dr_tulu_model, args.vllm_port,
    )


# ---------------------------------------------------------------------------
# Prediction & saving
# ---------------------------------------------------------------------------

def predict_signals(
    dr_tulu_url: str, prompt: str, timeout: int, max_retries: int = 3,
) -> tuple[str, dict]:
    """Send prompt to DR-Tulu and return (response_text, metadata).

    Retries on timeout or transient HTTP errors up to *max_retries* times.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [attempt {attempt}/{max_retries}] POST {dr_tulu_url}/chat (timeout={timeout}s)")
            resp = requests.post(
                f"{dr_tulu_url}/chat",
                json={"content": prompt},
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            response_text = (result.get("response") or "").strip()
            metadata = result.get("metadata", {})
            return response_text, metadata
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            print(f"  [warn] Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < max_retries:
                print(f"  Retrying in 10s ...")
                time.sleep(10)
    raise RuntimeError(
        f"DR-Tulu request failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


def save_results(
    output_dir: Path,
    space: str,
    domain: str,
    topic: str,
    response_text: str,
    signals: List[str],
    metadata: dict,
) -> Path:
    """Save raw response and extracted signals under a structured directory.

    Directory structure:
        {output_dir}/dr_tulu/{domain_slug}/{topic_slug}/{space}/{YEAR_SLUG}/
    """
    from mainframe_topics import make_domain_slug

    domain_slug = make_domain_slug(domain)
    topic_slug = make_topic_slug(topic)

    result_dir = output_dir / "dr_tulu" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Save raw response text
    (result_dir / f"response_{timestamp}.txt").write_text(
        response_text, encoding="utf-8"
    )
    (result_dir / "response_latest.txt").write_text(
        response_text, encoding="utf-8"
    )

    # Save extracted signals as JSON
    signals_payload = {
        "space": space,
        "domain": domain,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "signals": signals,
        "metadata": metadata,
    }
    (result_dir / f"signals_{timestamp}.json").write_text(
        json.dumps(signals_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "signals_latest.json").write_text(
        json.dumps(signals_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return result_dir


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DR-Tulu weak-signal prediction (no evaluation)."
    )
    parser.add_argument(
        "--spaces", nargs="+", default=["problem", "solution"],
        choices=["problem", "solution"],
        help="Which signal space(s) to predict.",
    )
    parser.add_argument(
        "--domain", nargs="+", default=None,
        help="Domain(s) to predict. If omitted, uses all domains.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--dr-tulu-dir", default=DEFAULT_DR_TULU_DIR, type=Path,
        help=f"Path to the dr-tulu repository (default: {DEFAULT_DR_TULU_DIR}).",
    )
    parser.add_argument(
        "--dr-tulu-model", default="rl-research/DR-Tulu-8B",
        help="DR-Tulu model name for vLLM.",
    )
    parser.add_argument("--dr-tulu-port", type=int, default=8080)
    parser.add_argument("--vllm-port", type=int, default=30001)
    parser.add_argument("--mcp-port", type=int, default=8000)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=32768)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from mainframe_topics import TOPICS_BY_DOMAIN, ALL_DOMAINS, make_domain_slug

    args = parse_args()
    random.seed(args.seed)

    # Determine which domains to run
    domains = args.domain if args.domain is not None else ALL_DOMAINS

    dr_tulu_url = f"http://127.0.0.1:{args.dr_tulu_port}"

    print("=" * 60)
    print(f"Spaces:           {args.spaces}")
    print(f"Domains:          {domains}")
    print(f"Year range:       {YEAR_RANGE}")
    print(f"Output dir:       {args.output_dir}")
    print(f"DR-Tulu dir:      {args.dr_tulu_dir}")
    print(f"DR-Tulu model:    {args.dr_tulu_model}")
    print(f"Seed:             {args.seed}")
    print("=" * 60)

    cutoff_year = year_range_to_cutoff(YEAR_RANGE)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain, [])
        if not topics:
            print(f"\n[warn] No topics found for domain '{domain}'. Skipping.")
            continue

        for topic in topics:
            for space in args.spaces:
                print(f"\n{'─' * 60}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'─' * 60}")

                domain_slug = make_domain_slug(domain)
                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / "dr_tulu" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Already exists: {result_dir}")
                    continue

                try:
                    # Start services with correct cutoff
                    print(f"S2 cutoff year: {cutoff_year}")
                    start_all_services(args, cutoff_year)

                    # Build prompt
                    prompt = build_prompt(space, domain, topic)
                    print(f"Prompt length: {len(prompt)} chars")

                    # Predict
                    print("Sending prompt to DR-Tulu ...")
                    response_text, metadata = predict_signals(
                        dr_tulu_url, prompt, args.request_timeout,
                    )
                    print(f"Response length: {len(response_text)} chars")

                    # Extract signals (fallback handles DR-Tulu's occasional
                    # curly-quote / broken-JSON responses)
                    signals = extract_candidate_signals(response_text)
                    if not signals:
                        signals = _fallback_extract_signals(response_text)
                    print(f"Extracted {len(signals)} candidate signals:")
                    for i, sig in enumerate(signals, 1):
                        print(f"  {i}. {sig}")

                    # Save
                    result_dir = save_results(
                        args.output_dir, space, domain, topic,
                        response_text, signals, metadata,
                    )
                    print(f"Results saved to: {result_dir}")
                except Exception as exc:
                    print(f"[ERROR] Failed for {domain}/{topic}/{space}: {exc}")
                    print("  Skipping to next topic ...")
                    continue

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
