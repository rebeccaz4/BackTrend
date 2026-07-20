#!/usr/bin/env python
"""Fetch Semantic Scholar papers for topics and paraphrases.

Default input:
  ../topics.json, relative to this script's construction directory.

Default output:
  ../papers/, relative to this script's construction directory.
  Each topic is stored under ../papers/{topic_slug}/.

Run examples:
  python construction/scripts/fetch_papers.py
  python construction/scripts/fetch_papers.py --topic "trustworthy AI"
  python construction/scripts/fetch_papers.py --year 2023
  python construction/scripts/fetch_papers.py --topic "trustworthy AI" --year 2023
  python construction/scripts/fetch_papers.py --topic "trustworthy AI" --year 2023 --force
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
DEFAULT_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

DEFAULT_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "publicationDate",
    "venue",
    "authors",
    "citationCount",
    "url",
]
OUTPUT_COLUMNS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "publicationDate",
    "venue",
    "authors",
    "citationCount",
    "url",
    "topic",
    "query_text",
    "query_type",
    "query_year",
    "matched_queries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Semantic Scholar papers for topics and paraphrases."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument(
        "--topic",
        help="Fetch only one topic from topics.json. Exact match required.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years to fetch. Default: 2019 2020 2021 2022 2023 2024.",
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Single year to fetch. Can be repeated. Overrides --years.",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--retry-backoff-max", type=float, default=60.0)
    parser.add_argument("--throttle-seconds", type=float, default=0.2)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Semantic Scholar fields to request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Backup existing topic-year outputs and fetch from scratch.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not write raw Semantic Scholar payload JSONL files.",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also add references from source-year target papers into separate deduped outputs.",
    )
    parser.add_argument(
        "--reference-source-years",
        type=int,
        nargs="+",
        default=[2024],
        help="Source paper years whose references are used. Default: 2024.",
    )
    parser.add_argument(
        "--reference-target-years",
        type=int,
        nargs="+",
        help=(
            "Reference years to add into _with_reference outputs. "
            "Default: the years requested by --years/--year."
        ),
    )
    parser.add_argument(
        "--reference-source-suffix",
        default="",
        help="Suffix for source paper files used to read target papers for references.",
    )
    parser.add_argument(
        "--reference-output-suffix",
        default="_with_reference",
        help="Suffix for merged keyword+reference paper outputs.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


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

def topic_queries(topic: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    queries = [{"query_text": topic, "query_type": "topic"}]
    seen = {topic.casefold()}
    for paraphrase in payload.get("paraphrase", []) or []:
        if not isinstance(paraphrase, str):
            continue
        text = paraphrase.strip()
        if not text or text.casefold() in seen:
            continue
        queries.append({"query_text": text, "query_type": "paraphrase"})
        seen.add(text.casefold())
    print(f"Generated {len(queries)} queries for topic: {topic}")
    return queries


def output_path(output_dir: Path, topic: str, year: int, suffix: str = "") -> Path:
    topic_slug = slugify(topic)
    return output_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def raw_path(output_dir: Path, topic: str, year: int, query_text: str) -> Path:
    topic_slug = slugify(topic)
    return (
        output_dir
        / topic_slug
        / "raw"
        / f"semantic_scholar_{topic_slug}_{year}_{slugify(query_text)}.jsonl"
    )


def state_path(output_dir: Path, topic: str, year: int, query_text: str) -> Path:
    topic_slug = slugify(topic)
    return output_dir / topic_slug / "state" / f"{topic_slug}_{year}_{slugify(query_text)}.json"


def summary_path(output_dir: Path) -> Path:
    return output_dir / "paper_fetch_summary.json"


def search_endpoint(api_base_url: str) -> str:
    return f"{api_base_url.rstrip('/')}/paper/search/bulk"


def paper_endpoint(api_base_url: str, paper_id: str) -> str:
    return f"{api_base_url.rstrip('/')}/paper/{paper_id}"


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(f"{path.suffix}.bak.{int(time.time())}")
    shutil.move(str(path), str(backup))
    print(f"Backed up existing file: {backup}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        try:
            return json.load(fp)
        except json.JSONDecodeError:
            return {}


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def clear_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True)
        fp.write("\n")


def sleep_with_jitter(base_delay: float) -> None:
    if base_delay <= 0:
        return
    jitter = base_delay * 0.1
    time.sleep(max(0.0, base_delay + random.uniform(-jitter, jitter)))


def request_with_retry(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    retry_backoff_max: float,
    throttle_seconds: float,
) -> dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)
        try:
            response = session.get(endpoint, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt > max_retries:
                raise RuntimeError(f"Request failed after retries: {exc}") from exc
            delay = min(retry_backoff * (2 ** (attempt - 1)), retry_backoff_max)
            sleep_with_jitter(delay)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRYABLE_STATUS and attempt <= max_retries:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = min(float(retry_after), retry_backoff_max)
                except ValueError:
                    delay = retry_backoff
            else:
                delay = min(retry_backoff * (2 ** (attempt - 1)), retry_backoff_max)
            sleep_with_jitter(delay)
            continue

        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        raise RuntimeError(
            f"Semantic Scholar request failed: status={response.status_code}, payload={payload}"
        )


def normalize_papers(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, max_level=1)
    df.rename(columns={col: col.replace(".", "_") for col in df.columns}, inplace=True)
    return df


def json_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if hasattr(value, "tolist") and not isinstance(value, str):
        return json.dumps(value.tolist(), ensure_ascii=False, sort_keys=True)
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return json_safe(value.tolist())
    try:
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS].copy()


def has_nonempty_abstract(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def filter_empty_abstracts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or "abstract" not in df.columns:
        return df, 0
    keep_mask = df["abstract"].apply(has_nonempty_abstract)
    dropped = int((~keep_mask).sum())
    if dropped:
        return df.loc[keep_mask].copy(), dropped
    return df, 0


def row_has_surrogate(row: pd.Series) -> bool:
    for value in row:
        if isinstance(value, str) and any("\ud800" <= ch <= "\udfff" for ch in value):
            return True
    return False


def drop_bad_unicode_rows(df: pd.DataFrame) -> pd.DataFrame:
    bad_mask = df.apply(row_has_surrogate, axis=1)
    if bad_mask.any():
        print(f"Dropping {bad_mask.sum()} rows with invalid Unicode.")
        return df.loc[~bad_mask].copy()
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except UnicodeEncodeError:
        df = drop_bad_unicode_rows(df)
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise SystemExit(
            "Writing parquet requires pyarrow or fastparquet. Install project dependencies "
            "or run inside the project virtual environment."
        ) from exc


def read_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df, dropped = filter_empty_abstracts(df)
    if dropped:
        print(f"Dropping {dropped} existing rows without abstracts from {path}.")
    records: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        paper_id = row.get("paperId")
        if paper_id:
            records[str(paper_id)] = row
    return records


def matched_queries(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("matched_queries")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return raw
    return []


def add_match(record: dict[str, Any], query_text: str, query_type: str) -> None:
    matches = matched_queries(record)
    key = (query_text, query_type)
    existing = {(item.get("query_text"), item.get("query_type")) for item in matches}
    if key not in existing:
        matches.append({"query_text": query_text, "query_type": query_type})
    record["matched_queries"] = json.dumps(matches, ensure_ascii=False)


def build_params(
    query_text: str,
    year: int,
    fields: list[str],
    per_page: int,
    language: str | None,
    token: str | None,
) -> dict[str, Any]:
    params = {
        "query": query_text,
        "year": year,
        "limit": per_page,
        "fields": ",".join(fields),
    }
    if language:
        params["language"] = language
    if token:
        params["token"] = token
    return params


def records_to_frame(records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame(records.values())
    return ensure_output_columns(df)


def year_mismatch_summary(df: pd.DataFrame, query_year: int) -> dict[str, Any]:
    if df.empty or "year" not in df.columns:
        return {"year_mismatch_count": 0, "year_mismatch_examples": []}
    returned_year = pd.to_numeric(df["year"], errors="coerce")
    mismatch_mask = returned_year.notna() & (returned_year.astype("Int64") != query_year)
    mismatch_df = df.loc[mismatch_mask, ["paperId", "title", "year", "query_year"]].head(10)
    return {
        "year_mismatch_count": int(mismatch_mask.sum()),
        "year_mismatch_examples": json_safe(mismatch_df.to_dict(orient="records")),
    }


def reference_fields(fields: list[str]) -> list[str]:
    nested = [f"references.{field}" for field in fields if field != "matched_queries"]
    return ["references"] + nested


def fetch_paper_references(
    *,
    session: requests.Session,
    api_base_url: str,
    paper_id: str,
    fields: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    payload = request_with_retry(
        session,
        paper_endpoint(api_base_url, paper_id),
        {"fields": ",".join(reference_fields(fields))},
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        retry_backoff_max=args.retry_backoff_max,
        throttle_seconds=args.throttle_seconds,
    )
    references = payload.get("references") or []
    return [ref for ref in references if isinstance(ref, dict)]


def add_reference_record(
    *,
    records: dict[str, dict[str, Any]],
    ref: dict[str, Any],
    topic: str,
    year: int,
    query_text: str,
    args: argparse.Namespace,
) -> bool:
    paper_id = ref.get("paperId")
    if not paper_id:
        return False
    paper_id = str(paper_id)
    if not has_nonempty_abstract(ref.get("abstract")):
        return False
    if paper_id not in records:
        record = {field: ref.get(field) for field in args.fields}
        record["paperId"] = paper_id
        record["authors"] = json_cell(record.get("authors"))
        record["topic"] = topic
        record["query_text"] = query_text
        record["query_type"] = "reference"
        record["query_year"] = year
        record["matched_queries"] = "[]"
        add_match(record, query_text, "reference")
        records[paper_id] = record
        return True
    add_match(records[paper_id], query_text, "reference")
    return False


def fetch_reference_papers_for_topic(
    *,
    session: requests.Session,
    output_dir: Path,
    topic: str,
    years: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records_by_year: dict[int, dict[str, dict[str, Any]]] = {}
    for year in years:
        records = read_existing_records(output_path(output_dir, topic, year))
        existing_reference_path = output_path(output_dir, topic, year, args.reference_output_suffix)
        records.update(read_existing_records(existing_reference_path))
        records_by_year[year] = records

    source_paths = [
        output_path(output_dir, topic, source_year, args.reference_source_suffix)
        for source_year in args.reference_source_years
    ]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Reference source paper files are missing. Fetch source years first: "
            + "; ".join(missing_sources)
        )

    source_frames = [pd.read_parquet(path) for path in source_paths]
    source_df = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    source_ids = sorted(source_df["paperId"].dropna().astype(str).unique()) if "paperId" in source_df else []
    target_years = set(years)
    added_by_year = {year: 0 for year in years}
    seen_reference_ids = {year: set(records_by_year[year]) for year in years}
    query_text = "references_from_" + "_".join(map(str, args.reference_source_years))

    progress = tqdm(source_ids, desc=f"{topic} | references", unit="paper", dynamic_ncols=True)
    failed = 0
    api_references = 0
    eligible_references = 0
    for source_id in progress:
        try:
            refs = fetch_paper_references(
                session=session,
                api_base_url=args.api_base_url,
                paper_id=source_id,
                fields=args.fields,
                args=args,
            )
        except Exception as exc:
            failed += 1
            print(f"{source_id}: reference fetch failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        api_references += len(refs)
        for ref in refs:
            ref_year = ref.get("year")
            try:
                ref_year = int(ref_year)
            except (TypeError, ValueError):
                continue
            if ref_year not in target_years:
                continue
            eligible_references += 1
            paper_id = str(ref.get("paperId") or "")
            was_new = paper_id not in seen_reference_ids[ref_year]
            added = add_reference_record(
                records=records_by_year[ref_year],
                ref=ref,
                topic=topic,
                year=ref_year,
                query_text=query_text,
                args=args,
            )
            if added or was_new:
                seen_reference_ids[ref_year].add(paper_id)
            if added:
                added_by_year[ref_year] += 1

    outputs = {}
    for year, records in records_by_year.items():
        out_file = output_path(output_dir, topic, year, args.reference_output_suffix)
        final_df = records_to_frame(records)
        write_parquet(final_df, out_file)
        check = year_mismatch_summary(final_df, year)
        outputs[str(year)] = {
            "output_path": str(out_file),
            "final_unique_papers": int(final_df["paperId"].nunique()) if not final_df.empty else 0,
            "new_reference_papers": added_by_year[year],
            **check,
        }

    return {
        "status": "references_fetched",
        "source_years": args.reference_source_years,
        "source_papers": len(source_ids),
        "api_references": api_references,
        "eligible_references": eligible_references,
        "failed_source_papers": failed,
        "outputs": outputs,
    }


def fetch_query(
    *,
    session: requests.Session,
    output_dir: Path,
    topic: str,
    query_text: str,
    query_type: str,
    year: int,
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_file = state_path(output_dir, topic, year, query_text)
    state = load_state(state_file)
    token = state.get("next_token")
    api_returned = int(state.get("api_returned", 0) or 0)
    new_unique = int(state.get("new_unique_papers", 0) or 0)
    missing_abstract = int(state.get("missing_abstract_papers", 0) or 0)
    pages = int(state.get("pages", 0) or 0)
    raw_file = raw_path(output_dir, topic, year, query_text)
    out_file = output_path(output_dir, topic, year)

    progress = None
    progress = tqdm(
        desc=f"{year} | {topic} | {query_text}",
        unit="papers",
        dynamic_ncols=True,
    )

    while True:
        params = build_params(
            query_text=query_text,
            year=year,
            fields=args.fields,
            per_page=args.per_page,
            language=args.language.strip() or None,
            token=token,
        )
        payload = request_with_retry(
            session,
            search_endpoint(args.api_base_url),
            params,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            retry_backoff_max=args.retry_backoff_max,
            throttle_seconds=args.throttle_seconds,
        )
        if not args.no_raw:
            append_raw(raw_file, payload)

        rows = payload.get("data", []) or []
        if not rows:
            clear_state(state_file)
            break

        pages += 1
        api_returned += len(rows)
        progress.update(len(rows))

        df = normalize_papers(rows)
        for row in df.to_dict(orient="records"):
            paper_id = row.get("paperId")
            if not paper_id:
                continue
            paper_id = str(paper_id)
            if not has_nonempty_abstract(row.get("abstract")):
                missing_abstract += 1
                continue
            if paper_id not in records:
                record = {field: row.get(field) for field in args.fields}
                record["paperId"] = paper_id
                record["authors"] = json_cell(record.get("authors"))
                record["topic"] = topic
                record["query_text"] = query_text
                record["query_type"] = query_type
                record["query_year"] = year
                record["matched_queries"] = "[]"
                add_match(record, query_text, query_type)
                records[paper_id] = record
                new_unique += 1
            else:
                add_match(records[paper_id], query_text, query_type)

        write_parquet(records_to_frame(records), out_file)

        token = payload.get("token")
        save_state(
            state_file,
            {
                "next_token": token,
                "api_returned": api_returned,
                "new_unique_papers": new_unique,
                "missing_abstract_papers": missing_abstract,
                "pages": pages,
                "updated_at": time.time(),
            },
        )
        if not token:
            clear_state(state_file)
            break

    progress.close()

    return {
        "query_text": query_text,
        "query_type": query_type,
        "api_returned": api_returned,
        "new_unique_papers": new_unique,
        "missing_abstract_papers": missing_abstract,
        "duplicate_or_existing_papers": max(api_returned - new_unique - missing_abstract, 0),
        "pages": pages,
        "raw_path": str(raw_file) if not args.no_raw else None,
    }


def fetch_topic_year(
    *,
    session: requests.Session,
    output_dir: Path,
    topic: str,
    topic_payload: dict[str, Any],
    year: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_file = output_path(output_dir, topic, year)
    query_defs = topic_queries(topic, topic_payload)
    state_files = [state_path(output_dir, topic, year, q["query_text"]) for q in query_defs]
    has_pending_state = any(path.exists() for path in state_files)

    if out_file.exists() and args.force:
        backup_file(out_file)
        for path in state_files:
            clear_state(path)
    elif out_file.exists() and not has_pending_state:
        df = pd.read_parquet(out_file)
        df, dropped = filter_empty_abstracts(df)
        if dropped:
            print(f"Dropping {dropped} rows without abstracts from existing output: {out_file}")
            write_parquet(ensure_output_columns(df), out_file)
        check = year_mismatch_summary(df, year)
        return {
            "status": "skipped_existing",
            "output_path": str(out_file),
            "final_unique_papers": int(df["paperId"].nunique()) if "paperId" in df else len(df),
            "queries": [],
            **check,
        }

    records = read_existing_records(out_file)
    query_summaries = []
    for query_def in query_defs:
        query_summaries.append(
            fetch_query(
                session=session,
                output_dir=output_dir,
                topic=topic,
                query_text=query_def["query_text"],
                query_type=query_def["query_type"],
                year=year,
                records=records,
                args=args,
            )
        )

    final_df = records_to_frame(records)
    write_parquet(final_df, out_file)
    check = year_mismatch_summary(final_df, year)
    return {
        "status": "fetched",
        "output_path": str(out_file),
        "final_unique_papers": int(final_df["paperId"].nunique()) if not final_df.empty else 0,
        "queries": query_summaries,
        **check,
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    path = summary_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(json_safe(summary), fp, ensure_ascii=False, indent=2)


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    requested_years = sorted(set(args.year or args.years))
    reference_target_years = sorted(set(args.reference_target_years or requested_years))
    years = requested_years
    if args.include_references:
        years = sorted(set(requested_years + args.reference_source_years))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topics = load_topics(args.topics_json, args.topic)
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    session = requests.Session()
    if api_key:
        session.headers.update({"x-api-key": api_key})
    else:
        print("Warning: SEMANTIC_SCHOLAR_API_KEY/S2_API_KEY is not set; rate limits may be low.")

    summary: dict[str, Any] = {
        "config": {
            "api_base_url": args.api_base_url,
            "search_endpoint": search_endpoint(args.api_base_url),
            "topics_json": str(args.topics_json),
            "output_dir": str(args.output_dir),
            "requested_years": requested_years,
            "fetch_years": years,
            "topic": args.topic,
            "language": args.language,
            "per_page": args.per_page,
            "fields": args.fields,
            "force": args.force,
            "raw_payloads": not args.no_raw,
            "include_references": args.include_references,
            "reference_source_years": args.reference_source_years,
            "reference_target_years": reference_target_years,
            "reference_source_suffix": args.reference_source_suffix,
            "reference_output_suffix": args.reference_output_suffix,
        },
        "topics": {},
    }

    for topic, topic_payload in topics.items():
        summary["topics"][topic] = {}
        for year in years:
            print("=" * 90)
            print(f"Fetching topic={topic!r} year={year}")
            result = fetch_topic_year(
                session=session,
                output_dir=args.output_dir,
                topic=topic,
                topic_payload=topic_payload,
                year=year,
                args=args,
            )
            summary["topics"][topic][str(year)] = result
            write_summary(args.output_dir, summary)
            print(
                f"{topic} | {year}: {result['status']} | "
                f"unique={result['final_unique_papers']} | "
                f"year_mismatches={result['year_mismatch_count']}"
            )

    if args.include_references:
        summary["reference_topics"] = {}
        for topic in topics:
            print("=" * 90)
            print(f"Fetching reference papers for topic={topic!r}")
            result = fetch_reference_papers_for_topic(
                session=session,
                output_dir=args.output_dir,
                topic=topic,
                years=reference_target_years,
                args=args,
            )
            summary["reference_topics"][topic] = result
            write_summary(args.output_dir, summary)
            for year, year_result in result["outputs"].items():
                print(
                    f"{topic} | {year} | references: unique={year_result['final_unique_papers']} | "
                    f"new_reference={year_result['new_reference_papers']} | "
                    f"year_mismatches={year_result['year_mismatch_count']}"
                )

    write_summary(args.output_dir, summary)
    print(f"Summary written to {summary_path(args.output_dir)}")


if __name__ == "__main__":
    main()
