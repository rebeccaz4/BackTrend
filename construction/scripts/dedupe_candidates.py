#!/usr/bin/env python
"""Deduplicate extracted candidate topics into a candidate index.

Run examples:
  python construction/scripts/dedupe_candidates.py
  python construction/scripts/dedupe_candidates.py --topic "trustworthy AI"
  python construction/scripts/dedupe_candidates.py --topic "trustworthy AI" --years 2019 2020 2021 2022 2023
  python construction/scripts/dedupe_candidates.py --use-clustering --cluster-threshold 0.9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_CANDIDATE_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_topics"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate extracted candidate topics into candidate_index.json."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", help="Process only one topic from topics.json.")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument(
        "--candidate-topic-type",
        choices=["all", "problem-space", "solution-space"],
        default="all",
    )
    parser.add_argument("--candidate-text", choices=["topic", "topic-evidence"], default="topic")
    parser.add_argument("--input-suffix", default="")
    parser.add_argument("--output-suffix", default="_cluster_t0.85")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--use-clustering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-threshold", type=float, default=0.85)
    parser.add_argument("--cluster-neighbors", type=int, default=50)
    parser.add_argument("--cluster-provider", choices=["local", "openai", "openrouter"], default="local")
    parser.add_argument("--cluster-embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--cluster-batch-size", type=int, default=128)
    parser.add_argument("--openai-cluster-base-url", default=OFFICIAL_OPENAI_BASE_URL)
    parser.add_argument("--openrouter-cluster-base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--openai-cluster-timeout", type=float, default=60)
    parser.add_argument("--openai-cluster-retries", type=int, default=4)
    parser.add_argument("--prefix-absorb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-min-tokens", type=int, default=2)
    parser.add_argument("--cluster-summary", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def normalize_candidate_topic(text: str) -> str:
    norm = text.lower().strip()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"[\s\.,;:]+$", "", norm)
    return norm


def candidate_id(target_topic_slug: str, topic_type: str, candidate_topic_norm: str) -> str:
    raw = f"{target_topic_slug}|{topic_type}|{candidate_topic_norm}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cluster_id(target_topic_slug: str, topic_type: str, member_ids: list[str]) -> str:
    raw = f"{target_topic_slug}|{topic_type}|{'|'.join(sorted(member_ids))}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_topics(path: Path, selected_topic: str | None = None) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        topics = json.load(fp)
    if selected_topic:
        if selected_topic not in topics:
            available = ", ".join(sorted(topics))
            raise SystemExit(
                f"Topic not found in {path}: {selected_topic}\nAvailable topics: {available}"
            )
        return {selected_topic: topics[selected_topic]}
    return topics


def candidate_jsonl_path(candidate_dir: Path, topic: str, year: int, suffix: str = "") -> Path:
    topic_slug = slugify(topic)
    return candidate_dir / topic_slug / f"candidate_topics_{topic_slug}_{year}{suffix}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_candidate_index(
    candidate_dir: Path,
    topics: dict[str, dict[str, Any]],
    years: list[int],
    candidate_text_mode: str,
    input_suffix: str = "",
) -> pd.DataFrame:
    by_id: dict[str, dict[str, Any]] = {}
    raw_mentions = 0
    for target_topic in topics:
        target_slug = slugify(target_topic)
        for year in years:
            for record in read_jsonl(candidate_jsonl_path(candidate_dir, target_topic, year, input_suffix)):
                source_paper = record.get("paper") or {}
                source_paper_id = str(source_paper.get("paperId") or "")
                for item in record.get("candidate_topics") or []:
                    candidate_topic = str(item.get("topic") or "").strip()
                    topic_type = str(item.get("topic_type") or "").strip()
                    if not candidate_topic or topic_type not in {"problem-space", "solution-space"}:
                        continue
                    raw_mentions += 1
                    topic_norm = normalize_candidate_topic(candidate_topic)
                    cid = candidate_id(target_slug, topic_type, topic_norm)
                    evidence = str(item.get("evidence") or "").strip()
                    confidence = str(item.get("confidence") or "").strip()
                    if cid not in by_id:
                        match_text = candidate_topic
                        if candidate_text_mode == "topic-evidence" and evidence:
                            match_text = f"{candidate_topic}. {evidence}"
                        by_id[cid] = {
                            "candidate_id": cid,
                            "target_topic": target_topic,
                            "target_topic_slug": target_slug,
                            "candidate_topic": candidate_topic,
                            "candidate_topic_norm": topic_norm,
                            "candidate_topic_type": topic_type,
                            "match_text": match_text,
                            "source_paper_ids": [],
                            "source_years": [],
                            "source_evidence": [],
                            "source_confidences": [],
                            "raw_mentions": 0,
                        }
                    row = by_id[cid]
                    row["raw_mentions"] += 1
                    if source_paper_id and source_paper_id not in row["source_paper_ids"]:
                        row["source_paper_ids"].append(source_paper_id)
                    if year not in row["source_years"]:
                        row["source_years"].append(year)
                    if evidence and evidence not in row["source_evidence"]:
                        row["source_evidence"].append(evidence)
                    if confidence and confidence not in row["source_confidences"]:
                        row["source_confidences"].append(confidence)

    candidates = pd.DataFrame(by_id.values())
    if candidates.empty:
        print(f"Raw candidate mentions: {raw_mentions:,}")
        return candidates
    candidates["source_count"] = candidates["source_paper_ids"].apply(len)
    candidates.sort_values(["target_topic_slug", "candidate_topic_type", "candidate_topic_norm"], inplace=True)
    candidates = candidates.reset_index(drop=True)
    print(f"Raw candidate mentions: {raw_mentions:,}")
    print(f"Deduplicated candidates: {len(candidates):,}")
    return candidates


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if not isinstance(value, (list, dict)) and pd.isna(value):
                record[key] = None
    return records


def write_candidate_json(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = dataframe_to_json_records(df)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def read_candidate_json(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    return pd.DataFrame(records)


def topic_candidate_path(
    output_dir: Path | str,
    topic: str,
    output_suffix: str = "",
    candidate_topic_type: str = "all",
    stem: str = "candidate_index",
    extension: str = "json",
) -> Path:
    output_dir = Path(output_dir)
    topic_slug = slugify(topic)
    if candidate_topic_type == "all":
        return output_dir / topic_slug / f"{stem}_{topic_slug}{output_suffix}.{extension}"
    return (
        output_dir
        / topic_slug
        / candidate_topic_type
        / f"{stem}_{topic_slug}_{candidate_topic_type}{output_suffix}.{extension}"
    )


def write_topic_candidate_outputs(
    candidates: pd.DataFrame,
    output_dir: Path,
    output_suffix: str = "",
    write_parquet: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for target_topic, topic_df in candidates.groupby("target_topic", sort=True):
        topic_path = topic_candidate_path(output_dir, str(target_topic), output_suffix)
        write_candidate_json(topic_df.reset_index(drop=True), topic_path)
        written.append(topic_path)
        print(f"Saved topic candidate JSON: {topic_path}")
        if write_parquet:
            parquet_path = topic_candidate_path(
                output_dir,
                str(target_topic),
                output_suffix,
                extension="parquet",
            )
            topic_df.reset_index(drop=True).to_parquet(parquet_path, index=False)
            written.append(parquet_path)
            print(f"Saved topic candidate parquet: {parquet_path}")

        for topic_type, type_df in topic_df.groupby("candidate_topic_type", sort=True):
            type_path = topic_candidate_path(
                output_dir,
                str(target_topic),
                output_suffix,
                candidate_topic_type=str(topic_type),
            )
            write_candidate_json(type_df.reset_index(drop=True), type_path)
            written.append(type_path)
            print(f"Saved {topic_type} candidate JSON: {type_path}")
            if write_parquet:
                type_parquet_path = topic_candidate_path(
                    output_dir,
                    str(target_topic),
                    output_suffix,
                    candidate_topic_type=str(topic_type),
                    extension="parquet",
                )
                type_df.reset_index(drop=True).to_parquet(type_parquet_path, index=False)
                written.append(type_parquet_path)
                print(f"Saved {topic_type} candidate parquet: {type_parquet_path}")
    return written


def choose_canonical_topic(group: pd.DataFrame) -> str:
    ranked = group.copy()
    ranked["topic_len"] = ranked["candidate_topic"].astype(str).str.len()
    ranked.sort_values(
        ["source_count", "raw_mentions", "topic_len", "candidate_topic_norm"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    return str(ranked.iloc[0]["candidate_topic"])


def summarize_cluster(member_topics: list[str], canonical_topic: str, enabled: bool) -> str:
    if not enabled:
        return canonical_topic
    unique_topics = []
    for topic in member_topics:
        if topic not in unique_topics:
            unique_topics.append(topic)
    preview = "; ".join(unique_topics[:5])
    return f"{canonical_topic}. Related extracted phrasings: {preview}"


def cluster_embedding_cache_path(output_dir: Path, topic_slug: str, provider: str, model: str, suffix: str) -> Path:
    safe_suffix = slugify(suffix) if suffix else "default"
    return (
        output_dir
        / topic_slug
        / "cluster_embedding_cache"
        / f"candidates_{topic_slug}_{safe_suffix}_{provider}_{slugify(model)}.parquet"
    )


def openai_embed_batch(
    client: OpenAI,
    model: str,
    texts: list[str],
    retries: int,
    provider_label: str,
) -> list[list[float]]:
    kwargs: dict[str, Any] = {"model": model, "input": texts}
    for attempt in range(1, retries + 1):
        try:
            response = client.embeddings.create(**kwargs)
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:
            if attempt >= retries:
                raise
            sleep_seconds = min(2 ** (attempt - 1), 20)
            print(
                f"{provider_label} cluster embedding failed on attempt {attempt}/{retries}: "
                f"{type(exc).__name__}: {str(exc)[:200]}; retrying in {sleep_seconds}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def cluster_api_config(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.cluster_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use --cluster-provider openai.")
        return api_key, args.openai_cluster_base_url, "OpenAI"
    if args.cluster_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY to use --cluster-provider openrouter.")
        return api_key, args.openrouter_cluster_base_url, "OpenRouter"
    raise ValueError(f"Unsupported API cluster provider: {args.cluster_provider}")


def embed_cluster_texts(args: argparse.Namespace, topic_slug: str, texts: list[str]) -> np.ndarray:
    path = cluster_embedding_cache_path(
        args.output_dir,
        topic_slug,
        args.cluster_provider,
        args.cluster_embed_model,
        args.output_suffix,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cache_df = pd.read_parquet(path)
    else:
        cache_df = pd.DataFrame(columns=["text", "embedding"])

    text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
    missing = [text for text in texts if text not in text_to_idx]
    cached_count = len(texts) - len(missing)
    print(
        f"[embedding] topic={topic_slug} cached={cached_count:,} "
        f"missing={len(missing):,} batch_size={args.cluster_batch_size:,}"
    )
    if missing:
        new_rows = []
        if args.cluster_provider in {"openai", "openrouter"}:
            api_key, base_url, provider_label = cluster_api_config(args)
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=args.openai_cluster_timeout,
            )
            for start in tqdm(range(0, len(missing), args.cluster_batch_size), desc="Embedding cluster texts", unit="batch"):
                batch = missing[start:start + args.cluster_batch_size]
                embeddings = openai_embed_batch(
                    client,
                    args.cluster_embed_model,
                    batch,
                    args.openai_cluster_retries,
                    provider_label,
                )
                for text, emb in zip(batch, embeddings):
                    new_rows.append({"text": text, "embedding": np.asarray(emb, dtype=np.float32).tolist()})
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(args.cluster_embed_model)
            for start in tqdm(range(0, len(missing), args.cluster_batch_size), desc="Embedding cluster texts", unit="batch"):
                batch = missing[start:start + args.cluster_batch_size]
                embeddings = model.encode(batch, batch_size=args.cluster_batch_size, show_progress_bar=False)
                for text, emb in zip(batch, embeddings):
                    new_rows.append({"text": text, "embedding": np.asarray(emb, dtype=np.float32).tolist()})
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_df.drop_duplicates(subset="text", keep="last", inplace=True)
        cache_df.reset_index(drop=True, inplace=True)
        cache_df.to_parquet(path, index=False)
        text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
        print(f"Saved cluster embedding cache: {path}")
    else:
        print(f"All cluster embeddings cached: {path}")

    embeddings = cache_df["embedding"].values
    return np.stack([np.asarray(embeddings[text_to_idx[text]], dtype=np.float32) for text in texts])


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_topic_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def is_prefix_extension(short_topic: str, long_topic: str, min_tokens: int) -> bool:
    short_norm = normalize_topic_text(short_topic)
    long_norm = normalize_topic_text(long_topic)
    if short_norm == long_norm:
        return False
    if len(short_norm.split()) < min_tokens:
        return False
    if not long_norm.startswith(short_norm + " "):
        return False
    remainder = long_norm[len(short_norm):].strip()
    if not remainder:
        return False
    if remainder.startswith(("and ", "or ")):
        return False
    return True


def cluster_from_members(members: pd.DataFrame, include_summary: bool) -> dict[str, Any]:
    member_ids = members["candidate_id"].astype(str).tolist()
    canonical_topic = choose_canonical_topic(members)
    member_topics = members["candidate_topic"].astype(str).tolist()
    source_paper_ids = sorted(
        {
            str(paper_id)
            for paper_ids in members["source_paper_ids"]
            for paper_id in (paper_ids or [])
        }
    )
    source_years = sorted(
        {
            int(year)
            for years in members["source_years"]
            for year in (years or [])
        }
    )
    return {
        "cluster_id": cluster_id(
            str(members.iloc[0]["target_topic_slug"]),
            str(members.iloc[0]["candidate_topic_type"]),
            member_ids,
        ),
        "target_topic": members.iloc[0]["target_topic"],
        "target_topic_slug": members.iloc[0]["target_topic_slug"],
        "candidate_topic_type": members.iloc[0]["candidate_topic_type"],
        "canonical_topic": canonical_topic,
        "summary": summarize_cluster(member_topics, canonical_topic, include_summary),
        "member_count": len(members),
        "source_paper_count": len(source_paper_ids),
        "member_candidate_ids": member_ids,
        "member_topics": member_topics,
        "source_paper_ids": source_paper_ids,
        "source_years": source_years,
    }


def knn_union_find_clusters(
    group: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    neighbors: int,
    include_summary: bool,
) -> list[dict[str, Any]]:
    if group.empty:
        return []
    n_neighbors = min(max(neighbors + 1, 2), len(group))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    print(
        f"{group.iloc[0]['target_topic_slug']} / {group.iloc[0]['candidate_topic_type']}: "
        "fitting nearest neighbors..."
    )
    nn.fit(embeddings)
    distances, indexes = nn.kneighbors(embeddings)

    parent = list(range(len(group)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_count = 0
    iterator = tqdm(
        enumerate(zip(distances, indexes)),
        total=len(group),
        desc=f"Union-find {group.iloc[0]['target_topic_slug']}/{group.iloc[0]['candidate_topic_type']}",
        unit="row",
    )
    for row_idx, (row_distances, row_indexes) in iterator:
        for dist, neighbor_idx in zip(row_distances, row_indexes):
            if row_idx == neighbor_idx:
                continue
            if 1.0 - float(dist) >= threshold:
                union(row_idx, int(neighbor_idx))
                edge_count += 1

    components: dict[int, list[int]] = {}
    for idx in range(len(group)):
        components.setdefault(find(idx), []).append(idx)

    clusters = [cluster_from_members(group.iloc[indexes].copy(), include_summary) for indexes in components.values()]
    print(
        f"{group.iloc[0]['target_topic_slug']} / {group.iloc[0]['candidate_topic_type']}: "
        f"candidates={len(group):,}, knn_edges={edge_count:,}, clusters={len(clusters):,}"
    )
    return clusters


def clusters_to_member_frame(clusters: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        for cid, topic in zip(cluster["member_candidate_ids"], cluster["member_topics"]):
            rows.append(
                {
                    "candidate_id": cid,
                    "target_topic": cluster["target_topic"],
                    "target_topic_slug": cluster["target_topic_slug"],
                    "candidate_topic": topic,
                    "candidate_topic_norm": normalize_topic_text(topic),
                    "candidate_topic_type": cluster["candidate_topic_type"],
                    "source_paper_ids": cluster["source_paper_ids"],
                    "source_years": cluster["source_years"],
                    "source_count": cluster["source_paper_count"],
                    "raw_mentions": 0,
                }
            )
    return pd.DataFrame(rows)


def apply_prefix_absorb_to_group(clusters: list[dict[str, Any]], min_tokens: int, include_summary: bool) -> tuple[list[dict[str, Any]], int]:
    if not clusters:
        return clusters, 0

    parent = list(range(len(clusters)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(absorber: int, absorbed: int) -> None:
        absorber_root = find(absorber)
        absorbed_root = find(absorbed)
        if absorber_root != absorbed_root:
            parent[absorbed_root] = absorber_root

    canonical_topics = [str(cluster["canonical_topic"]) for cluster in clusters]
    merge_count = 0
    for short_idx, short_topic in enumerate(canonical_topics):
        for long_idx, long_topic in enumerate(canonical_topics):
            if short_idx == long_idx:
                continue
            if is_prefix_extension(short_topic, long_topic, min_tokens):
                union(short_idx, long_idx)
                merge_count += 1

    components: dict[int, list[int]] = {}
    for idx in range(len(clusters)):
        components.setdefault(find(idx), []).append(idx)

    absorbed = []
    for indexes in components.values():
        if len(indexes) == 1:
            absorbed.append(clusters[indexes[0]])
            continue
        member_frame = clusters_to_member_frame([clusters[idx] for idx in indexes])
        member_frame.drop_duplicates(subset=["candidate_id"], keep="first", inplace=True)
        absorbed.append(cluster_from_members(member_frame.reset_index(drop=True), include_summary))
    return absorbed, merge_count


def apply_prefix_absorb(clusters: pd.DataFrame, min_tokens: int, include_summary: bool) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    absorbed_all: list[dict[str, Any]] = []
    total_merges = 0
    grouped = list(clusters.groupby(["target_topic_slug", "candidate_topic_type"], sort=True))
    for (target_slug, topic_type), group in tqdm(grouped, desc="Prefix absorb groups", unit="group"):
        records = group.to_dict(orient="records")
        absorbed, merges = apply_prefix_absorb_to_group(records, min_tokens, include_summary)
        total_merges += merges
        absorbed_all.extend(absorbed)
        print(f"{target_slug} / {topic_type}: prefix_edges={merges:,}, clusters_after_prefix={len(absorbed):,}")
    print(f"Prefix absorb edges total: {total_merges:,}")
    return pd.DataFrame(absorbed_all)


def semantic_cluster_candidates(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    cluster_rows: list[dict[str, Any]] = []
    for target_slug, topic_candidates in candidates.groupby("target_topic_slug", sort=True):
        topic_candidates = topic_candidates.reset_index(drop=True)
        print(
            f"[cluster] topic={target_slug} candidates={len(topic_candidates):,} "
            f"provider={args.cluster_provider} model={args.cluster_embed_model}"
        )
        texts = topic_candidates["match_text"].fillna(topic_candidates["candidate_topic"]).astype(str).tolist()
        embeddings = normalize_matrix(embed_cluster_texts(args, str(target_slug), texts))
        for (_, _), group in topic_candidates.groupby(["target_topic_slug", "candidate_topic_type"], sort=True):
            group_indexes = group.index.to_numpy()
            cluster_rows.extend(
                knn_union_find_clusters(
                    group.reset_index(drop=True),
                    embeddings[group_indexes],
                    threshold=args.cluster_threshold,
                    neighbors=args.cluster_neighbors,
                    include_summary=args.cluster_summary,
                )
            )

    clusters = pd.DataFrame(cluster_rows)
    if args.prefix_absorb:
        clusters = apply_prefix_absorb(clusters, args.prefix_min_tokens, args.cluster_summary)
    clusters.sort_values(["target_topic_slug", "candidate_topic_type", "canonical_topic"], inplace=True)
    clusters.reset_index(drop=True, inplace=True)
    print(f"Semantic clusters: {len(clusters):,}")
    return clusters


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    topics = load_topics(args.topics_json, args.topic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_index(
        args.candidate_dir,
        topics,
        args.years,
        args.candidate_text,
        args.input_suffix,
    )
    if args.candidate_topic_type != "all" and not candidates.empty:
        candidates = candidates[candidates["candidate_topic_type"] == args.candidate_topic_type].reset_index(drop=True)
    print(f"Candidate rows selected after read/dedup: {len(candidates):,}")
    if candidates.empty:
        raise RuntimeError("No candidate topics found. Run extract_candidate.py first.")
    write_topic_candidate_outputs(
        candidates,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        write_parquet=args.write_parquet,
    )
    if args.use_clustering:
        clusters = semantic_cluster_candidates(candidates, args)
        for target_topic, cluster_df in clusters.groupby("target_topic", sort=True):
            cluster_path = topic_candidate_path(
                args.output_dir,
                str(target_topic),
                args.output_suffix,
                stem="candidate_clusters",
            )
            write_candidate_json(cluster_df.reset_index(drop=True), cluster_path)
            print(f"Saved topic cluster JSON: {cluster_path}")
            if args.write_parquet:
                cluster_parquet_path = topic_candidate_path(
                    args.output_dir,
                    str(target_topic),
                    args.output_suffix,
                    stem="candidate_clusters",
                    extension="parquet",
                )
                cluster_df.reset_index(drop=True).to_parquet(cluster_parquet_path, index=False)
                print(f"Saved topic cluster parquet: {cluster_parquet_path}")
            for topic_type, type_df in cluster_df.groupby("candidate_topic_type", sort=True):
                type_cluster_path = topic_candidate_path(
                    args.output_dir,
                    str(target_topic),
                    args.output_suffix,
                    candidate_topic_type=str(topic_type),
                    stem="candidate_clusters",
                )
                write_candidate_json(type_df.reset_index(drop=True), type_cluster_path)
                print(f"Saved {topic_type} cluster JSON: {type_cluster_path}")
                if args.write_parquet:
                    type_cluster_parquet_path = topic_candidate_path(
                        args.output_dir,
                        str(target_topic),
                        args.output_suffix,
                        candidate_topic_type=str(topic_type),
                        stem="candidate_clusters",
                        extension="parquet",
                    )
                    type_df.reset_index(drop=True).to_parquet(type_cluster_parquet_path, index=False)
                    print(f"Saved {topic_type} cluster parquet: {type_cluster_parquet_path}")


if __name__ == "__main__":
    main()
