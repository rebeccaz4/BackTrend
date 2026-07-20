#!/usr/bin/env python
"""Match candidate topics to later target papers through reference adoption.

This is for backtracking weak signals: early source papers are counted as early
support, and later target papers are counted as later support when they cite
those early source papers.

Run example:
  python construction/scripts/match_candidate_reference_adoption.py \
    --topic "large language models" \
    --candidate-source cluster \
    --dedup-suffix _cluster_t0.85 \
    --output-suffix _cluster_t0.85
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_DEDUP_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_PAPERS_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_matching"
DEFAULT_EARLY_YEARS = [2019, 2020, 2021, 2022, 2023]
DEFAULT_LATER_YEAR = 2024
DEFAULT_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dedupe_candidates import (  # noqa: E402
    load_topics,
    normalize_candidate_topic,
    read_candidate_json,
    slugify,
    topic_candidate_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match candidates to later target papers through references."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--dedup-dir", type=Path, default=DEFAULT_DEDUP_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--papers-suffix", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--early-years", type=int, nargs="+", default=DEFAULT_EARLY_YEARS)
    parser.add_argument("--later-year", type=int, default=DEFAULT_LATER_YEAR)
    parser.add_argument("--candidate-source", choices=["index", "cluster"], default="cluster")
    parser.add_argument("--candidate-topic-type", choices=["all", "problem-space", "solution-space"], default="all")
    parser.add_argument("--dedup-suffix", default="_cluster_t0.85")
    parser.add_argument("--output-suffix", default="_cluster_t0.85")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--retry-backoff-max", type=float, default=60.0)
    parser.add_argument("--throttle-seconds", type=float, default=0.2)
    parser.add_argument("--force-reference-cache", action="store_true")
    return parser.parse_args()


def papers_path(papers_dir: Path, topic: str, year: int, suffix: str = "") -> Path:
    topic_slug = slugify(topic)
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def load_papers(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "paperId" not in df.columns:
        return pd.DataFrame()
    df["paperId"] = df["paperId"].dropna().astype(str)
    return df.reset_index(drop=True)


def output_path(output_dir: Path, topic: str, suffix: str, extension: str) -> Path:
    topic_slug = slugify(topic)
    return output_dir / topic_slug / f"reference_match_{topic_slug}{suffix}.{extension}"


def reference_cache_path(output_dir: Path, topic: str, later_year: int, suffix: str) -> Path:
    topic_slug = slugify(topic)
    safe_suffix = slugify(suffix) if suffix else "default"
    return (
        output_dir
        / topic_slug
        / "reference_cache"
        / f"reference_edges_{topic_slug}_{later_year}_{safe_suffix}.parquet"
    )


def clusters_to_candidates(clusters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in clusters.to_dict(orient="records"):
        candidate_topic = str(record.get("canonical_topic") or "").strip()
        rows.append(
            {
                "candidate_id": str(record.get("cluster_id") or ""),
                "target_topic": record.get("target_topic"),
                "target_topic_slug": record.get("target_topic_slug"),
                "candidate_topic": candidate_topic,
                "candidate_topic_norm": normalize_candidate_topic(candidate_topic),
                "candidate_topic_type": record.get("candidate_topic_type"),
                "source_paper_ids": record.get("source_paper_ids") or [],
                "source_years": record.get("source_years") or [],
                "member_count": record.get("member_count"),
                "source_paper_count": record.get("source_paper_count"),
                "member_candidate_ids": record.get("member_candidate_ids"),
                "member_topics": record.get("member_topics"),
            }
        )
    return pd.DataFrame(rows)


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    topics = load_topics(args.topics_json, args.topic)
    topic = next(iter(topics))
    stem = "candidate_clusters" if args.candidate_source == "cluster" else "candidate_index"
    path = topic_candidate_path(
        args.dedup_dir,
        topic,
        args.dedup_suffix,
        candidate_topic_type=args.candidate_topic_type,
        stem=stem,
    )
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    df = read_candidate_json(path)
    if args.candidate_source == "cluster":
        df = clusters_to_candidates(df)
    return df.reset_index(drop=True)


def list_cell(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        parsed = value.tolist()
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return []


def json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if not isinstance(value, (list, dict)) and pd.isna(value):
                record[key] = None
    return records


def write_json_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(json_safe_records(df), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def sleep_with_jitter(base_delay: float) -> None:
    if base_delay <= 0:
        return
    jitter = base_delay * 0.1
    time.sleep(max(0.0, base_delay + random.uniform(-jitter, jitter)))


def request_batch_references(
    session: requests.Session,
    endpoint: str,
    paper_ids: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any] | None]:
    params = {"fields": "references.paperId,references.year"}
    payload = {"ids": paper_ids}
    attempt = 0
    while True:
        attempt += 1
        if args.throttle_seconds > 0:
            time.sleep(args.throttle_seconds)
        try:
            response = session.post(endpoint, params=params, json=payload, timeout=args.request_timeout)
        except requests.RequestException as exc:
            if attempt > args.max_retries:
                raise RuntimeError(f"Batch reference request failed after retries: {exc}") from exc
            sleep_with_jitter(min(args.retry_backoff * (2 ** (attempt - 1)), args.retry_backoff_max))
            continue
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else []
        if response.status_code in RETRYABLE_STATUS and attempt <= args.max_retries:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = min(float(retry_after), args.retry_backoff_max)
                except ValueError:
                    delay = args.retry_backoff
            else:
                delay = min(args.retry_backoff * (2 ** (attempt - 1)), args.retry_backoff_max)
            sleep_with_jitter(delay)
            continue
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        raise RuntimeError(f"Batch reference request failed: status={response.status_code}, payload={body}")


def fetch_reference_edges(
    *,
    session: requests.Session,
    later_papers: pd.DataFrame,
    cache_path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if cache_path.exists() and not args.force_reference_cache:
        print(f"Loaded reference edge cache: {cache_path}")
        return pd.read_parquet(cache_path)

    paper_ids = later_papers["paperId"].dropna().astype(str).unique().tolist()
    endpoint = f"{args.api_base_url.rstrip('/')}/paper/batch"
    rows = []
    for start in tqdm(range(0, len(paper_ids), args.batch_size), desc="Fetching reference batches", unit="batch"):
        batch_ids = paper_ids[start:start + args.batch_size]
        batch = request_batch_references(session, endpoint, batch_ids, args)
        for source_id, item in zip(batch_ids, batch):
            if not isinstance(item, dict):
                continue
            for ref in item.get("references") or []:
                if not isinstance(ref, dict):
                    continue
                ref_id = ref.get("paperId")
                if not ref_id:
                    continue
                rows.append(
                    {
                        "later_paper_id": str(source_id),
                        "referenced_paper_id": str(ref_id),
                        "referenced_year": ref.get("year"),
                    }
                )

    edges = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(cache_path, index=False)
    print(f"Saved reference edge cache: {cache_path}")
    return edges


def match_row(
    *,
    cand: pd.Series,
    paper_id: str,
    paper_title: str,
    year: int,
    method: str,
    referenced_source_paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": cand["candidate_id"],
        "target_topic": cand["target_topic"],
        "target_topic_slug": cand["target_topic_slug"],
        "candidate_topic": cand["candidate_topic"],
        "candidate_topic_norm": cand["candidate_topic_norm"],
        "candidate_topic_type": cand["candidate_topic_type"],
        "year": year,
        "matched_paper_id": paper_id,
        "matched_paper_title": paper_title,
        "cosine": None,
        "rerank_score": None,
        "match_method": method,
        "candidate_source": "reference_adoption",
        "cluster_member_count": cand.get("member_count"),
        "cluster_source_paper_count": cand.get("source_paper_count"),
        "cluster_member_candidate_ids": cand.get("member_candidate_ids"),
        "cluster_member_topics": cand.get("member_topics"),
        "referenced_source_paper_ids": referenced_source_paper_ids or [],
    }


def build_matches(args: argparse.Namespace) -> pd.DataFrame:
    candidates = load_candidates(args)
    print(f"Loaded candidates: {len(candidates):,}")
    later_papers = load_papers(papers_path(args.papers_dir, args.topic, args.later_year, args.papers_suffix))
    if later_papers.empty:
        raise FileNotFoundError(f"No later papers found for {args.topic} {args.later_year}")
    later_titles = dict(zip(later_papers["paperId"].astype(str), later_papers.get("title", "").fillna("").astype(str)))

    early_papers_by_id: dict[str, tuple[int, str]] = {}
    for year in args.early_years:
        papers = load_papers(papers_path(args.papers_dir, args.topic, year, args.papers_suffix))
        for _, row in papers.iterrows():
            early_papers_by_id[str(row["paperId"])] = (year, str(row.get("title", "") or ""))

    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    ref_to_candidate_indexes: dict[str, list[int]] = {}
    for idx, cand in candidates.iterrows():
        for source_id in list_cell(cand.get("source_paper_ids")):
            source_id = str(source_id)
            ref_to_candidate_indexes.setdefault(source_id, []).append(idx)
            early_info = early_papers_by_id.get(source_id)
            if not early_info:
                continue
            year, title = early_info
            key = (str(cand["candidate_id"]), source_id, year)
            rows_by_key[key] = match_row(
                cand=cand,
                paper_id=source_id,
                paper_title=title,
                year=year,
                method="extraction_source",
                referenced_source_paper_ids=[],
            )

    print(f"Extraction-source pairs: {len(rows_by_key):,}")
    session = requests.Session()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        session.headers.update({"x-api-key": api_key})
    else:
        print("Warning: SEMANTIC_SCHOLAR_API_KEY/S2_API_KEY is not set; rate limits may be low.")
    edges = fetch_reference_edges(
        session=session,
        later_papers=later_papers,
        cache_path=reference_cache_path(args.output_dir, args.topic, args.later_year, args.papers_suffix),
        args=args,
    )
    if edges.empty:
        return pd.DataFrame(list(rows_by_key.values()))

    edges["referenced_paper_id"] = edges["referenced_paper_id"].astype(str)
    matched_edges = edges[edges["referenced_paper_id"].isin(ref_to_candidate_indexes)].copy()
    print(f"Reference edges to candidate source papers: {len(matched_edges):,}")
    adoption_updates = 0
    for _, edge in matched_edges.iterrows():
        later_id = str(edge["later_paper_id"])
        ref_id = str(edge["referenced_paper_id"])
        for cand_idx in ref_to_candidate_indexes.get(ref_id, []):
            cand = candidates.iloc[cand_idx]
            key = (str(cand["candidate_id"]), later_id, args.later_year)
            if key in rows_by_key:
                refs = rows_by_key[key].setdefault("referenced_source_paper_ids", [])
                if ref_id not in refs:
                    refs.append(ref_id)
                continue
            rows_by_key[key] = match_row(
                cand=cand,
                paper_id=later_id,
                paper_title=later_titles.get(later_id, ""),
                year=args.later_year,
                method="reference_adoption",
                referenced_source_paper_ids=[ref_id],
            )
            adoption_updates += 1
    print(f"Reference-adoption pairs added: {adoption_updates:,}")
    return pd.DataFrame(list(rows_by_key.values()))


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    matches = build_matches(args)
    parquet_path = output_path(args.output_dir, args.topic, args.output_suffix, "parquet")
    json_path = output_path(args.output_dir, args.topic, args.output_suffix, "json")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(parquet_path, index=False)
    write_json_records(matches, json_path)
    print(f"Saved {len(matches):,} matched pairs: {parquet_path}")
    print(f"Saved {len(matches):,} matched pairs JSON: {json_path}")


if __name__ == "__main__":
    main()
