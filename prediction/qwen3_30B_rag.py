#!/usr/bin/env python
# coding: utf-8
"""
Qwen3-30B + LlamaIndex RAG -- weak-signal prediction (prediction only, no evaluation).

Same RAG pipeline as qwen3_8B_rag.py but defaults to the larger Qwen3-30B AWQ
model (tensor parallelism configurable via --tensor-parallel, default 1).

Usage example:
    python qwen3_30B_rag.py \
        --spaces problem solution \
        --domain "Artificial intelligence and machine learning" \
        --output-dir ./outputs \
        --retrieval-queries "model-based reinforcement learning language models"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests as http_requests
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


def year_range_bounds(year_range: str) -> tuple[int, int]:
    parts = year_range.split("-")
    return int(parts[0]), int(parts[-1])


def topic_to_papers_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# Semantic Scholar bulk retrieval (identical to qwen3_8B_rag.py)
# ---------------------------------------------------------------------------

S2_BULK_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CONSTRUCTION_DIR = Path(__file__).resolve().parent / "../construction"
DEFAULT_PAPERS_DIR = CONSTRUCTION_DIR / "papers"


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def _clean_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def deduplicate_papers(papers: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    dedup: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for p in papers:
        key = (
            p.get("paperId")
            or p.get("url")
            or (p.get("title") or "").lower().strip()
        )
        if not key:
            continue
        if key in dedup:
            duplicate_count += 1
            continue
        dedup[key] = p
    return list(dedup.values()), duplicate_count


def load_local_topic_papers(
    topic: str,
    papers_dir: Path,
    year_range: str,
) -> List[Dict[str, Any]]:
    """Load pre-retrieved construction papers for a topic if available."""
    start_year, end_year = year_range_bounds(year_range)
    topic_slug = topic_to_papers_slug(topic)
    topic_dir = papers_dir / topic_slug
    if not topic_dir.exists():
        return []

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to load local construction parquet papers."
        ) from exc

    papers: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        path = topic_dir / f"papers_{topic_slug}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            y = _clean_int(row.get("year"), default=-1)
            if not (start_year <= y <= end_year):
                continue
            papers.append({
                "paperId": _clean_text(row.get("paperId")),
                "title": _clean_text(row.get("title"), "Untitled"),
                "year": y,
                "abstract": _clean_text(row.get("abstract")),
                "url": _clean_text(row.get("url"), "N/A"),
                "citationCount": _clean_int(row.get("citationCount")),
                "venue": _clean_text(row.get("venue")),
                "query": _clean_text(row.get("query_text"), topic),
                "source": "local_construction",
            })

    return papers


def s2_bulk_fetch_with_cutoff(
    query: str,
    year_range: str,
    cutoff_year: int,
    max_total: int,
    page_size: int,
    api_key: str,
    max_retries: int = 5,
    retry_backoff: float = 1.5,
) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError(
            "Missing Semantic Scholar API key. "
            "Set SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY."
        )
    headers = {"x-api-key": api_key}
    fields = "paperId,title,year,abstract,url,citationCount"
    token = None
    scanned = 0
    kept: List[Dict[str, Any]] = []
    start_year, _ = year_range_bounds(year_range)

    while scanned < max_total:
        limit = min(page_size, max_total - scanned)
        params: Dict[str, Any] = {
            "query": query,
            "year": year_range,
            "limit": limit,
            "fields": fields,
        }
        if token:
            params["token"] = token

        payload = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = http_requests.get(
                    S2_BULK_ENDPOINT, headers=headers, params=params, timeout=60,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise http_requests.HTTPError(
                        f"Transient HTTP {resp.status_code}", response=resp,
                    )
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                sleep_s = retry_backoff * attempt
                print(f"  [warn] S2 error '{query}' ({attempt}/{max_retries}): {exc}. Retrying...")
                time.sleep(sleep_s)

        if payload is None:
            raise RuntimeError(f"S2 failed after {max_retries} attempts for '{query}': {last_exc}")

        batch = payload.get("data", []) or []
        if not batch:
            break
        for p in batch:
            scanned += 1
            y = p.get("year")
            if isinstance(y, int) and start_year <= y <= cutoff_year:
                kept.append({
                    "paperId": p.get("paperId") or "",
                    "title": p.get("title") or "Untitled",
                    "year": y,
                    "abstract": (p.get("abstract") or "").strip(),
                    "url": p.get("url") or "N/A",
                    "citationCount": int(p.get("citationCount") or 0),
                    "query": query,
                    "source": "semantic_scholar",
                })
        token = payload.get("token")
        if not token:
            break
    return kept


# ---------------------------------------------------------------------------
# LlamaIndex RAG helpers (identical to qwen3_8B_rag.py)
# ---------------------------------------------------------------------------

def papers_to_documents(papers: List[Dict[str, Any]]):
    from llama_index.core import Document
    docs = []
    for i, p in enumerate(papers, start=1):
        text = (
            f"[Paper {i}]\nTitle: {p.get('title', 'Untitled')}\n"
            f"Abstract: {p.get('abstract', '')}"
        )
        metadata = {
            "paper_id": p.get("paperId", ""),
            "title": p.get("title", "Untitled"),
            "year": p.get("year"),
            "url": p.get("url", "N/A"),
            "citationCount": p.get("citationCount", 0),
            "abstract": p.get("abstract", ""),
            "venue": p.get("venue", ""),
            "query": p.get("query", ""),
            "source": p.get("source", ""),
        }
        metadata_keys = list(metadata.keys())
        docs.append(Document(
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        ))
    return docs


def llamaindex_rank_topk(documents, query: str, top_k: int):
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not documents:
        return []

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)


def build_evidence_block(ranked_nodes) -> tuple[str, List[Dict]]:
    evidence_lines: List[str] = []
    metadata_list: List[Dict] = []
    for rank, nws in enumerate(ranked_nodes, start=1):
        md = nws.node.metadata or {}
        title = md.get("title", "Untitled")
        abstract = (md.get("abstract") or "").strip()
        if len(abstract) > 900:
            abstract = abstract[:900].rstrip() + "..."
        score = float(nws.score) if nws.score is not None else None
        evidence_lines.append(
            f"[{rank}] Title: {title}\n"
            f"Abstract: {abstract}"
        )
        metadata_list.append({
            "rank": rank,
            "title": title,
            "abstract": abstract,
            "score": score,
            "year": md.get("year"),
            "venue": md.get("venue", ""),
            "url": md.get("url", "N/A"),
            "citationCount": md.get("citationCount", 0),
            "query": md.get("query", ""),
            "source": md.get("source", ""),
        })
    return "\n\n".join(evidence_lines), metadata_list


def run_rag_retrieval(
    retrieval_queries, year_range, cutoff_year, papers_dir,
    s2_max_total, s2_page_size,
    s2_api_key, s2_max_retries, s2_retry_backoff, rag_top_k,
    space, mainframe_topic, domain,
) -> tuple[str, List[Dict]]:
    local_papers = load_local_topic_papers(mainframe_topic, papers_dir, year_range)
    if local_papers:
        unique_papers, duplicate_count = deduplicate_papers(local_papers)
        print(
            f"  Local papers: {len(local_papers)} raw -> {len(unique_papers)} unique "
            f"({duplicate_count} duplicates removed, years={year_range})"
        )
    else:
        all_candidates: List[Dict[str, Any]] = []
        for q in retrieval_queries:
            papers = s2_bulk_fetch_with_cutoff(
                q, year_range, cutoff_year, s2_max_total, s2_page_size,
                s2_api_key, s2_max_retries, s2_retry_backoff,
            )
            all_candidates.extend(papers)
            print(f"  Query '{q}': {len(papers)} papers (year={year_range})")

        unique_papers, duplicate_count = deduplicate_papers(all_candidates)
        print(
            f"  S2 candidates: {len(all_candidates)} raw -> {len(unique_papers)} unique "
            f"({duplicate_count} duplicates removed)"
        )

    documents = papers_to_documents(unique_papers)
    ranking_query = mainframe_topic
    print(f"  Ranking query: {ranking_query}")
    ranked_nodes = llamaindex_rank_topk(documents, ranking_query, rag_top_k)
    print(f"  Ranked top-k: {len(ranked_nodes)}")
    return build_evidence_block(ranked_nodes)


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_server(
    model, host, port, tensor_parallel=4, max_model_len=32768, gpu_mem=0.9,
) -> subprocess.Popen | None:
    if port_open(port, host):
        print(f"vLLM already running on {host}:{port}. Reusing.")
        return None
    log_file = f"vllm_server_{port}.log"
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", host, "--port", str(port),
        "--dtype", "auto", "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
    ]
    if tensor_parallel > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
    print(f"[vLLM] Starting server: {' '.join(cmd)}")
    print(f"[vLLM] Log file: {log_file}")
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    import threading
    def _stream_output():
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            print(f"[vLLM] {line_stripped}", flush=True)
            log_fh.write(line)
            log_fh.flush()
        log_fh.close()
    reader_thread = threading.Thread(target=_stream_output, daemon=True)
    reader_thread.start()

    base_url = f"http://{host}:{port}"
    max_wait, interval, waited = 900, 5, 0
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        # Check if process has crashed
        if proc.poll() is not None:
            reader_thread.join(timeout=5)
            print(f"[vLLM] ERROR: Server process exited with code {proc.returncode} after {waited}s")
            raise RuntimeError(f"vLLM server died with exit code {proc.returncode}. Check {log_file}")
        try:
            r = http_requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"[vLLM] Server ready (pid={proc.pid}) on {host}:{port} after {waited}s")
                return proc
        except Exception:
            print(f"[vLLM] Waiting for server... ({waited}s / {max_wait}s)", flush=True)
    print(f"[vLLM] WARNING: Server may not be ready after {max_wait}s. Check {log_file}")
    return proc


def generate_with_qwen(
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    import requests as _req

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    raw = _req.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    if raw.status_code != 200:
        print(f"[vLLM error] status={raw.status_code}")
        print(f"[vLLM error] body={raw.text}")
        raw.raise_for_status()

    data = raw.json()
    choice = data["choices"][0]
    response_text = (choice["message"]["content"] or "").strip()
    return {
        "text": response_text,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "raw_response": data,
    }


def analyze_generation(
    response_text: str,
    signals: List[str],
    generation_result: Dict[str, Any],
    prompt_chars: int,
    estimated_prompt_tokens: int,
    evidence_papers_kept: int,
    max_input_tokens: int,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    weak_signal_payload = load_weak_signal_json(response_text)
    valid_json = weak_signal_payload is not None
    finish_reason = generation_result.get("finish_reason")
    has_think = bool(re.search(r"<think\b[^>]*>", response_text, flags=re.IGNORECASE))
    closed_think = (not has_think) or bool(
        re.search(r"</think\s*>", response_text, flags=re.IGNORECASE)
    )

    incomplete_reasons: List[str] = []
    if finish_reason == "length":
        incomplete_reasons.append("finish_reason_length")
    if not valid_json:
        incomplete_reasons.append("missing_valid_weak_signals_json")
    if has_think and not closed_think:
        incomplete_reasons.append("unclosed_think_block")

    return {
        "finish_reason": finish_reason,
        "usage": generation_result.get("usage"),
        "valid_json": valid_json,
        "empty_signals": len(signals) == 0,
        "incomplete": bool(incomplete_reasons),
        "incomplete_reasons": incomplete_reasons,
        "response_chars": len(response_text),
        "extracted_signal_count": len(signals),
        "has_think_block": has_think,
        "closed_think_block": closed_think,
        "prompt_chars": prompt_chars,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "evidence_papers_kept": evidence_papers_kept,
        "max_input_tokens": max_input_tokens,
        "request_params": request_params,
        "raw_response": generation_result.get("raw_response"),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir,
    space,
    domain,
    topic,
    response_text,
    signals,
    retrieved_metadata,
    generation_meta,
) -> Path:
    from mainframe_topics import make_domain_slug
    topic_slug = make_topic_slug(topic)
    domain_slug = make_domain_slug(domain)
    result_dir = output_dir / "qwen3_30b_awq_rag" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")

    payload = {
        "domain": domain, "space": space, "mainframe_topic": topic,
        "year_range": YEAR_RANGE, "timestamp": timestamp,
        "signals": signals, "retrieved_papers": retrieved_metadata,
    }
    (result_dir / f"signals_{timestamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "signals_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / f"generation_meta_{timestamp}.json").write_text(
        json.dumps(generation_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "generation_meta_latest.json").write_text(
        json.dumps(generation_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_dir


def existing_incomplete_reasons(result_dir: Path) -> List[str]:
    reasons: List[str] = []
    meta_path = result_dir / "generation_meta_latest.json"
    response_path = result_dir / "response_latest.txt"
    signals_path = result_dir / "signals_latest.json"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("incomplete"):
                for reason in meta.get("incomplete_reasons") or ["generation_meta_incomplete"]:
                    if isinstance(reason, str):
                        reasons.append(reason)
            if meta.get("finish_reason") == "length":
                reasons.append("finish_reason_length")
            if meta.get("valid_json") is False:
                reasons.append("missing_valid_weak_signals_json")
        except (json.JSONDecodeError, OSError):
            reasons.append("unreadable_generation_meta")

    if not response_path.exists():
        reasons.append("missing_response_latest")
    else:
        try:
            response_text = response_path.read_text(encoding="utf-8")
            if load_weak_signal_json(response_text) is None:
                reasons.append("missing_valid_weak_signals_json")
            if re.search(r"<think\b[^>]*>", response_text, flags=re.IGNORECASE) and not re.search(
                r"</think\s*>", response_text, flags=re.IGNORECASE
            ):
                reasons.append("unclosed_think_block")
        except OSError:
            reasons.append("unreadable_response_latest")

    if not signals_path.exists():
        reasons.append("missing_signals_latest")

    deduped: List[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-30B + LlamaIndex RAG weak-signal prediction (no evaluation)."
    )
    p.add_argument("--spaces", nargs="+", default=["problem", "solution"],
                   choices=["problem", "solution"],
                   help="Signal spaces to predict.")
    p.add_argument("--domain", nargs="+", default=None,
                   help="Domains to predict. If omitted, uses all domains.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--retrieval-queries", nargs="+", default=None,
                   help="Semantic Scholar queries. If omitted, uses the topic name as query.")

    p.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL_PATH", "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"))
    p.add_argument("--vllm-host", default=os.getenv("QWEN_VLLM_HOST", "127.0.0.1"))
    p.add_argument("--vllm-port", type=int, default=int(os.getenv("QWEN_VLLM_PORT", "6004")))
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Optional vLLM max_tokens. If omitted, uses the server/model default.")
    p.add_argument("--reserve-output-tokens", type=int, default=0,
                   help="Optional prompt-budget reserve. Default 0 preserves current evidence retention.")
    p.add_argument("--prompt-safety-margin", type=int, default=0,
                   help="Extra prompt-budget margin used only when set explicitly.")
    p.add_argument("--skip-vllm-start", action="store_true")
    p.add_argument("--rerun-incomplete", action="store_true",
                   help="Regenerate existing results that are missing valid final JSON or ended by length.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate existing results regardless of completeness.")

    p.add_argument("--rag-top-k", type=int, default=int(os.getenv("RAG_TOP_K", "50")))
    p.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR,
                   help="Local construction/papers directory used before S2 fallback.")
    p.add_argument("--s2-max-total", type=int, default=10000)
    p.add_argument("--s2-page-size", type=int, default=100)
    p.add_argument("--s2-api-key", default=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", ""))
    p.add_argument("--s2-max-retries", type=int, default=5)
    p.add_argument("--s2-retry-backoff", type=float, default=1.5)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    from mainframe_topics import TOPICS_BY_DOMAIN, ALL_DOMAINS, make_domain_slug

    domains = args.domain if args.domain else ALL_DOMAINS
    random.seed(args.seed)
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise RuntimeError("--max-tokens must be positive when provided.")
    if args.reserve_output_tokens < 0:
        raise RuntimeError("--reserve-output-tokens must be >= 0.")
    if args.prompt_safety_margin < 0:
        raise RuntimeError("--prompt-safety-margin must be >= 0.")

    base_url = f"http://{args.vllm_host}:{args.vllm_port}"

    print("=" * 60)
    print(f"Spaces:            {args.spaces}")
    print(f"Domains:           {domains}")
    print(f"Year range:        {YEAR_RANGE}")
    print(f"Model:             {args.qwen_model}")
    print(f"vLLM:              {base_url}")
    print(f"Tensor parallel:   {args.tensor_parallel}")
    print(f"RAG top-k:         {args.rag_top_k}")
    print(f"Max tokens:        {args.max_tokens if args.max_tokens is not None else 'vLLM default'}")
    print(f"Output reserve:    {args.reserve_output_tokens}")
    print(f"Prompt margin:     {args.prompt_safety_margin}")
    print(f"Retrieval queries: {args.retrieval_queries}")
    print(f"Local papers dir:  {args.papers_dir}")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 60)

    if not args.skip_vllm_start:
        start_vllm_server(
            args.qwen_model, args.vllm_host, args.vllm_port,
            args.tensor_parallel, args.max_model_len, args.gpu_memory_utilization,
        )

    cutoff_year = year_range_to_cutoff(YEAR_RANGE)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain, [])
        domain_slug = make_domain_slug(domain)
        for topic in topics:
            for space in args.spaces:
                print(f"\n{'---' * 20}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'---' * 20}")

                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / "qwen3_30b_awq_rag" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    if args.overwrite:
                        print(f"[rerun] --overwrite set for existing result: {result_dir}")
                    else:
                        incomplete_reasons = existing_incomplete_reasons(result_dir)
                        if args.rerun_incomplete and incomplete_reasons:
                            print(
                                "[rerun] Existing result appears incomplete "
                                f"({', '.join(incomplete_reasons)}): {result_dir}"
                            )
                        else:
                            if incomplete_reasons:
                                print(
                                    "[skip] Already exists but appears incomplete; "
                                    f"pass --rerun-incomplete to regenerate: {result_dir}"
                                )
                            else:
                                print(f"[skip] Already exists: {result_dir}")
                            continue

                retrieval_queries = args.retrieval_queries or [topic]
                print(f"Paper year range: {YEAR_RANGE}")
                print(f"Retrieval queries: {retrieval_queries}")

                print("Running RAG retrieval ...")
                evidence_block, retrieved_metadata = run_rag_retrieval(
                    retrieval_queries, YEAR_RANGE, cutoff_year, args.papers_dir,
                    args.s2_max_total, args.s2_page_size,
                    args.s2_api_key, args.s2_max_retries, args.s2_retry_backoff,
                    args.rag_top_k, space, topic, domain,
                )

                base_prompt = build_prompt(space, domain, topic)
                suffix = "\n\nUse this evidence when generating the weak signals."
                evidence_papers = evidence_block.split("\n\n")
                max_input_tokens = (
                    args.max_model_len
                    - args.reserve_output_tokens
                    - args.prompt_safety_margin
                )
                if max_input_tokens <= 0:
                    raise RuntimeError(
                        "Prompt token budget is <= 0. Reduce --reserve-output-tokens "
                        "or --prompt-safety-margin."
                    )

                while True:
                    cur_evidence = "\n\n".join(evidence_papers)
                    augmented_prompt = (
                        f"{base_prompt}\n\nRetrieved paper evidence "
                        f"(already time-cutoff filtered):\n{cur_evidence}"
                        f"{suffix}"
                    )
                    estimated_tokens = len(augmented_prompt) // 3
                    if estimated_tokens <= max_input_tokens or len(evidence_papers) <= 1:
                        break
                    evidence_papers.pop()

                print(f"Augmented prompt length: {len(augmented_prompt)} chars "
                      f"(~{estimated_tokens} tokens, {len(evidence_papers)} papers kept)")

                print("Calling Qwen via vLLM ...")
                generation_result = generate_with_qwen(
                    base_url, args.qwen_model, augmented_prompt,
                    args.temperature, args.max_tokens,
                )
                response_text = generation_result["text"]
                print(f"Response length: {len(response_text)} chars")

                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                generation_meta = analyze_generation(
                    response_text=response_text,
                    signals=signals,
                    generation_result=generation_result,
                    prompt_chars=len(augmented_prompt),
                    estimated_prompt_tokens=estimated_tokens,
                    evidence_papers_kept=len(evidence_papers),
                    max_input_tokens=max_input_tokens,
                    request_params={
                        "model": args.qwen_model,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "reserve_output_tokens": args.reserve_output_tokens,
                        "prompt_safety_margin": args.prompt_safety_margin,
                    },
                )
                result_dir = save_results(
                    args.output_dir, space, domain, topic,
                    response_text, signals, retrieved_metadata, generation_meta,
                )
                print(f"Results saved to: {result_dir}")

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
