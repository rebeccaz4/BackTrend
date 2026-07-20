#!/usr/bin/env python
"""Compute final weak-signal onset scores, gates, and plots.

This script is the final post-dedup pipeline step. It reads deduplicated
candidate topics, yearly target-topic papers, and 2024 reference-adoption
matches or reference edges, then writes topic/space separated final results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

_TMP = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_TMP / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP / "fontconfig-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_DEDUP_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_PAPERS_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_MATCHING_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_matching"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "final_results"
DEFAULT_EARLY_YEARS = [2019, 2020, 2021, 2022, 2023]
DEFAULT_LATER_YEAR = 2024
DEFAULT_ONSET_YEARS = [2019, 2020, 2021, 2022]
DEFAULT_DEDUP_SUFFIX = "_cluster_t0.85"
DEFAULT_MATCHING_SUFFIX = "_cluster_t0.85"
EPSILON = 1e-12

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
    parser = argparse.ArgumentParser(description="Compute final weak-signal results.")
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--dedup-dir", type=Path, default=DEFAULT_DEDUP_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--papers-suffix", default="")
    parser.add_argument("--matching-dir", type=Path, default=DEFAULT_MATCHING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", help="Process only one topic from topics.json.")
    parser.add_argument("--candidate-source", choices=["index", "cluster"], default="cluster")
    parser.add_argument("--candidate-topic-type", choices=["all", "problem-space", "solution-space"], default="all")
    parser.add_argument("--dedup-suffix", default=DEFAULT_DEDUP_SUFFIX)
    parser.add_argument("--matching-suffix", default=DEFAULT_MATCHING_SUFFIX)
    parser.add_argument("--early-years", type=int, nargs="+", default=DEFAULT_EARLY_YEARS)
    parser.add_argument("--later-year", type=int, default=DEFAULT_LATER_YEAR)
    parser.add_argument("--onset-years", type=int, nargs="+", default=DEFAULT_ONSET_YEARS)
    parser.add_argument("--top-n-plots", type=int, default=30)
    parser.add_argument("--retention-tolerance", type=float, default=0.8)
    parser.add_argument(
        "--gate3-skip-current-n",
        type=int,
        default=1,
        help=(
            "For solution-space gate3, skip a year-to-year retention check when "
            "the current year's topic paper count is at or below this value."
        ),
    )
    parser.add_argument("--pre-peak-tolerance", type=float, default=0.6)
    parser.add_argument("--validation-lift-threshold", type=float, default=1.2)
    return parser.parse_args()


def is_survey_title(title: Any) -> bool:
    return bool(re.search(r"\bsurvey\b", str(title or ""), flags=re.IGNORECASE))


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


def safe_name(text: str, max_len: int = 90) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return (out or "item")[:max_len]


def normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric = numeric.clip(lower=0.0)
    if numeric.empty:
        return numeric
    lo = float(numeric.min())
    hi = float(numeric.max())
    if math.isclose(lo, hi):
        return pd.Series([1.0 if hi > 0 else 0.0] * len(numeric), index=numeric.index)
    return (numeric - lo) / (hi - lo)


def read_reference_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    columns = ["candidate_id", "matched_paper_id", "year", "target_topic_slug"]
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        keep = [column for column in columns if column in available]
        return pd.read_parquet(path, columns=keep)


def candidate_json_to_df(path: Path, source: str) -> pd.DataFrame:
    df = read_candidate_json(path)
    if source != "cluster":
        return df.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        candidate_topic = str(record.get("canonical_topic") or "").strip()
        rows.append(
            {
                "candidate_id": str(record.get("cluster_id") or ""),
                "target_topic": record.get("target_topic"),
                "target_topic_slug": record.get("target_topic_slug"),
                "candidate_topic": candidate_topic,
                "candidate_topic_norm": normalize_candidate_topic(candidate_topic),
                "candidate_topic_type": record.get("candidate_topic_type"),
                "candidate_source": source,
                "member_count": record.get("member_count"),
                "source_paper_count": record.get("source_paper_count"),
                "member_candidate_ids": record.get("member_candidate_ids"),
                "member_topics": record.get("member_topics"),
                "source_paper_ids": record.get("source_paper_ids"),
                "source_years": record.get("source_years"),
            }
        )
    return pd.DataFrame(rows)


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    topics = load_topics(args.topics_json, args.topic)
    frames: list[pd.DataFrame] = []
    stem = "candidate_clusters" if args.candidate_source == "cluster" else "candidate_index"
    for topic in topics:
        path = topic_candidate_path(
            args.dedup_dir,
            topic,
            args.dedup_suffix,
            candidate_topic_type=args.candidate_topic_type,
            stem=stem,
        )
        if not path.exists():
            if args.topic:
                raise FileNotFoundError(f"Candidate file not found: {path}")
            continue
        frames.append(candidate_json_to_df(path, args.candidate_source))
    if not frames:
        raise FileNotFoundError(f"No candidate files found in {args.dedup_dir}")
    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset=["candidate_id"], keep="first", inplace=True)
    return out.reset_index(drop=True)


def papers_path(papers_dir: Path, topic_slug: str, year: int, suffix: str) -> Path:
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def load_topic_papers(
    papers_dir: Path,
    topic_slug: str,
    years: list[int],
    suffix: str,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, int]], dict[int, set[str]]]:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = papers_path(papers_dir, topic_slug, year, suffix)
        if not path.exists():
            frames.append(pd.DataFrame(columns=["paperId", "title", "year", "is_survey"]))
            continue
        df = pd.read_parquet(path)
        if "paperId" not in df.columns:
            frames.append(pd.DataFrame(columns=["paperId", "title", "year", "is_survey"]))
            continue
        if "title" not in df.columns:
            df["title"] = ""
        df = df[["paperId", "title"]].dropna(subset=["paperId"]).copy()
        df["paperId"] = df["paperId"].astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        df["year"] = year
        df["is_survey"] = df["title"].map(is_survey_title)
        df.drop_duplicates(subset=["paperId"], keep="first", inplace=True)
        frames.append(df)
    papers = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    paper_index: dict[str, dict[str, Any]] = {}
    stats: dict[int, dict[str, int]] = {}
    non_survey_ids: dict[int, set[str]] = {}
    for year in years:
        year_df = papers[papers["year"] == year] if not papers.empty else pd.DataFrame()
        total = int(year_df["paperId"].nunique()) if not year_df.empty else 0
        survey = int(year_df[year_df["is_survey"]]["paperId"].nunique()) if not year_df.empty else 0
        stats[year] = {"total": total, "survey_excluded": survey, "non_survey": total - survey}
        non_survey_ids[year] = set(year_df.loc[~year_df.get("is_survey", False), "paperId"].astype(str))
    for _, row in papers.iterrows():
        paper_index[str(row["paperId"])] = {
            "year": int(row["year"]),
            "title": str(row.get("title") or ""),
            "is_survey": bool(row.get("is_survey")),
        }
    return paper_index, stats, non_survey_ids


def source_counts_by_year(
    candidate: pd.Series,
    paper_index: dict[str, dict[str, Any]],
    early_years: list[int],
) -> tuple[dict[int, int], dict[int, list[str]], list[str]]:
    counts = {year: 0 for year in early_years}
    ids_by_year = {year: [] for year in early_years}
    missing: list[str] = []
    for paper_id in sorted({str(pid) for pid in list_cell(candidate.get("source_paper_ids")) if str(pid).strip()}):
        info = paper_index.get(paper_id)
        if not info:
            missing.append(paper_id)
            continue
        year = int(info["year"])
        if year not in counts or info.get("is_survey"):
            continue
        counts[year] += 1
        ids_by_year[year].append(paper_id)
    return counts, ids_by_year, missing


def reference_match_path(matching_dir: Path, topic_slug: str, suffix: str) -> Path:
    return matching_dir / topic_slug / f"reference_match_{topic_slug}{suffix}.parquet"


def reference_cache_candidates(matching_dir: Path, topic_slug: str, later_year: int, suffix: str) -> list[Path]:
    safe_suffix = slugify(suffix) if suffix else "default"
    cache_dir = matching_dir / topic_slug / "reference_cache"
    return [
        cache_dir / f"reference_edges_{topic_slug}_{later_year}_{safe_suffix}.parquet",
        cache_dir / f"reference_edges_{topic_slug}_{later_year}_default.parquet",
    ]


def later_counts_from_matches(
    matches: pd.DataFrame,
    topic_slug: str,
    later_year: int,
    non_survey_ids: set[str],
) -> dict[str, int]:
    if matches.empty:
        return {}
    subset = matches.copy()
    if "target_topic_slug" in subset.columns:
        subset = subset[subset["target_topic_slug"].astype(str) == topic_slug]
    if "year" in subset.columns:
        subset = subset[subset["year"].astype(int) == later_year]
    if subset.empty or "candidate_id" not in subset.columns or "matched_paper_id" not in subset.columns:
        return {}
    subset = subset.dropna(subset=["matched_paper_id"]).copy()
    subset = subset[subset["matched_paper_id"].astype(str).isin(non_survey_ids)]
    if subset.empty:
        return {}
    return (
        subset.drop_duplicates(["candidate_id", "matched_paper_id"])
        .groupby("candidate_id")["matched_paper_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def source_reference_map(candidates: pd.DataFrame, paper_index: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in candidates.iterrows():
        candidate_id = str(row["candidate_id"])
        for paper_id in list_cell(row.get("source_paper_ids")):
            paper_id = str(paper_id)
            info = paper_index.get(paper_id)
            if not info or info.get("is_survey"):
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "referenced_paper_id": paper_id,
                    "source_year": int(info["year"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def later_counts_from_reference_cache(
    args: argparse.Namespace,
    topic_slug: str,
    candidates: pd.DataFrame,
    paper_index: dict[str, dict[str, Any]],
    non_survey_ids: set[str],
) -> dict[str, int]:
    cache_path = next(
        (path for path in reference_cache_candidates(args.matching_dir, topic_slug, args.later_year, args.matching_suffix) if path.exists()),
        None,
    )
    if cache_path is None:
        return {}
    edges = pd.read_parquet(cache_path)
    required = {"later_paper_id", "referenced_paper_id"}
    if edges.empty or not required.issubset(edges.columns):
        return {}
    edges = edges[["later_paper_id", "referenced_paper_id"]].dropna().copy()
    edges["later_paper_id"] = edges["later_paper_id"].astype(str)
    edges["referenced_paper_id"] = edges["referenced_paper_id"].astype(str)
    edges = edges[edges["later_paper_id"].isin(non_survey_ids)]
    source_map = source_reference_map(candidates, paper_index)
    if edges.empty or source_map.empty:
        return {}
    matched = edges.merge(source_map, on="referenced_paper_id", how="inner")
    matched = matched[matched["source_year"] < args.later_year]
    if matched.empty:
        return {}
    return (
        matched.drop_duplicates(["candidate_id", "later_paper_id"])
        .groupby("candidate_id")["later_paper_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def fit_log_exponential(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    x = np.arange(len(values), dtype=float)
    y = np.log(np.asarray(values, dtype=float) + EPSILON)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    r2 = 0.0 if math.isclose(ss_tot, 0.0) else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return float(slope), float(intercept), r2


def score_onset(
    row: pd.Series,
    onset: int,
    space: str,
    early_years: list[int],
) -> dict[str, Any]:
    values = [float(row[f"topic_f_{year}"]) for year in early_years if year >= onset]
    if len(values) < 2 or max(values) <= 0 or values[-1] <= 0:
        return {
            "onset_year": onset,
            "raw_score": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "growth_factor": 0.0,
            "positive_share": 0.0,
            "end_strength": 0.0,
            "pre_peak": 0.0,
            "pre_onset_penalty": 1.0,
        }

    slope, intercept, r2 = fit_log_exponential(values)
    nonzero = [value for value in values if value > 0.0]
    first_nonzero = nonzero[0] if nonzero else 0.0
    growth = (values[-1] + EPSILON) / (first_nonzero + EPSILON)
    deltas = np.diff(np.asarray(values, dtype=float))
    positive_share = float(np.mean(deltas > 0.0)) if len(deltas) else 0.0
    end_strength = min(1.0, (values[-1] + EPSILON) / (max(values) + EPSILON))
    terminal = min(1.0, (values[-1] + EPSILON) / (values[-2] + EPSILON))
    prior_years = [year for year in early_years if year < onset]
    pre_peak = max([float(row[f"topic_f_{year}"]) for year in prior_years], default=0.0)
    start_value = float(row[f"topic_f_{onset}"])
    pre_onset_penalty = 1.0
    if pre_peak > EPSILON:
        pre_onset_penalty = min(1.0, (start_value + EPSILON) / (pre_peak + EPSILON))

    if slope <= 0.0:
        raw = 0.0
    elif space == "solution":
        nonzero_count = sum(value > 0.0 for value in values)
        nonzero_penalty = min(1.0, nonzero_count / 3.0) ** 2
        raw = (
            (r2 ** 3)
            * math.log1p(max(0.0, growth - 1.0))
            * positive_share
            * end_strength
            * terminal
            * nonzero_penalty
            * (pre_onset_penalty ** 2)
        )
    else:
        growth_reward = math.log(growth) if growth > 1.0 else 0.0
        raw = (r2 ** 2) * growth_reward * positive_share * end_strength * pre_onset_penalty

    return {
        "onset_year": onset,
        "raw_score": float(raw),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "growth_factor": float(growth),
        "positive_share": positive_share,
        "end_strength": end_strength,
        "pre_peak": float(pre_peak),
        "pre_onset_penalty": float(pre_onset_penalty),
    }


def add_scores_and_gates(
    df: pd.DataFrame,
    space: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for _, row in df.iterrows():
        fits = [score_onset(row, onset, space, args.early_years) for onset in args.onset_years]
        best = max(fits, key=lambda item: item["raw_score"])
        out = row.to_dict()
        prefix = "solution" if space == "solution" else "problem"
        out[f"{prefix}_raw_score"] = best["raw_score"]
        out[f"{prefix}_onset"] = best["onset_year"]
        out[f"{prefix}_slope"] = best["slope"]
        out[f"{prefix}_intercept"] = best["intercept"]
        out[f"{prefix}_r2"] = best["r2"]
        out[f"{prefix}_growth"] = best["growth_factor"]
        out[f"{prefix}_positive_share"] = best["positive_share"]
        out[f"{prefix}_end_strength"] = best["end_strength"]
        out[f"{prefix}_pre_peak"] = best["pre_peak"]
        out[f"{prefix}_pre_onset_penalty"] = best["pre_onset_penalty"]
        rows.append(out)
    scored = pd.DataFrame(rows)
    prefix = "solution" if space == "solution" else "problem"
    scored[f"{prefix}_fit_score"] = normalize(scored[f"{prefix}_raw_score"])
    scored["score"] = scored[f"{prefix}_fit_score"]

    topic_cols = [f"topic_f_{year}" for year in args.early_years]
    scored["early_topic_peak"] = scored[topic_cols].max(axis=1)
    onset_peak = []
    for _, row in scored.iterrows():
        onset = int(row[f"{prefix}_onset"])
        cols = [f"topic_f_{year}" for year in args.early_years if year >= onset]
        onset_peak.append(float(row[cols].max()) if cols else 0.0)
    scored["onset_to_2023_peak"] = onset_peak
    scored["gate2_2024_validation_peak"] = (
        scored["ref_f_2024"] + EPSILON >= args.validation_lift_threshold * scored["early_topic_peak"]
    )
    scored["gate1_signal_presence"] = scored["score"] > 0.0
    if space == "solution":
        gate3 = []
        gate3_strict = []
        gate3_skipped_steps = []
        gate3_failed_steps = []
        for _, row in scored.iterrows():
            onset = int(row[f"{prefix}_onset"])
            pass_gate = True
            pass_strict_gate = True
            skipped_steps = []
            failed_steps = []
            for year in args.early_years:
                next_year = year + 1
                if year < onset or next_year not in args.early_years:
                    continue
                step_pass = (
                    float(row[f"topic_f_{next_year}"]) + EPSILON
                    >= args.retention_tolerance * float(row[f"topic_f_{year}"])
                )
                if step_pass:
                    continue
                pass_strict_gate = False
                step = f"{year}->{next_year}"
                if int(row[f"topic_n_{year}"]) <= args.gate3_skip_current_n:
                    skipped_steps.append(step)
                    continue
                failed_steps.append(step)
                if not step_pass:
                    pass_gate = False
                    break
            gate3.append(pass_gate)
            gate3_strict.append(pass_strict_gate)
            gate3_skipped_steps.append(";".join(skipped_steps))
            gate3_failed_steps.append(";".join(failed_steps))
        scored["gate3_solution_strict_smooth_growth"] = gate3_strict
        scored["gate3_solution_skipped_sparse_steps"] = gate3_skipped_steps
        scored["gate3_solution_failed_steps"] = gate3_failed_steps
        scored["gate3_solution_smooth_growth"] = gate3
    else:
        scored["gate3_problem_2023_retention"] = (
            scored["topic_f_2023"] + EPSILON >= args.retention_tolerance * scored["onset_to_2023_peak"]
        )
    gate4 = []
    for _, row in scored.iterrows():
        onset = int(row[f"{prefix}_onset"])
        threshold = max(float(row[f"topic_f_{onset}"]), args.pre_peak_tolerance * float(row["topic_f_2023"]))
        gate4.append(float(row[f"{prefix}_pre_peak"]) <= threshold + EPSILON)
    scored["gate4_pre_onset_not_too_high"] = gate4
    topic_2023 = scored["topic_f_2023"].astype(float)
    ref_2024 = scored["ref_f_2024"].astype(float)
    scored["validation_lift_2024_vs_2023"] = np.where(
        topic_2023 > 0.0,
        ref_2024 / topic_2023,
        np.where(ref_2024 > 0.0, np.inf, 0.0),
    )
    scored["validation_lift_2024_vs_peak"] = np.where(
        scored["early_topic_peak"] > 0.0,
        ref_2024 / scored["early_topic_peak"],
        np.where(ref_2024 > 0.0, np.inf, 0.0),
    )
    gate_cols = gate_columns(space)
    scored["passed_all_gates"] = scored[gate_cols].all(axis=1)
    scored.sort_values("score", ascending=False, inplace=True)
    return scored.reset_index(drop=True)


def gate_columns(space: str) -> list[str]:
    cols = [
        "gate1_signal_presence",
        "gate2_2024_validation_peak",
    ]
    if space == "solution":
        cols.append("gate3_solution_smooth_growth")
    else:
        cols.append("gate3_problem_2023_retention")
    cols.append("gate4_pre_onset_not_too_high")
    return cols


def gate_labels(space: str) -> dict[str, str]:
    labels = {
        "gate1_signal_presence": "gate1_signal_presence",
        "gate2_2024_validation_peak": "gate2_2024_validation_peak",
    }
    if space == "solution":
        labels["gate3_solution_smooth_growth"] = "gate3_solution_smooth_growth"
    else:
        labels["gate3_problem_2023_retention"] = "gate3_problem_2023_retention"
    labels["gate4_pre_onset_not_too_high"] = "gate4_pre_onset_not_too_high"
    return labels


def json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (list, dict))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            )
    return out


def write_markdown(df: pd.DataFrame, path: Path, space: str) -> None:
    lines = [
        f"# Final {space.title()} Weak Signals",
        "",
        "| rank | candidate_topic | score | onset | ref_f_2024 | topic_f_2023 | passed_all_gates |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    prefix = "solution" if space == "solution" else "problem"
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['candidate_topic']} | {float(row['score']):.6g} | "
            f"{int(row[f'{prefix}_onset'])} | {float(row['ref_f_2024']):.6g} | "
            f"{float(row['topic_f_2023']):.6g} | {bool(row['passed_all_gates'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_candidate(row: pd.Series, out_path: Path, space: str, early_years: list[int], later_year: int) -> None:
    prefix = "solution" if space == "solution" else "problem"
    onset = int(row[f"{prefix}_onset"])
    slope = float(row[f"{prefix}_slope"])
    intercept = float(row[f"{prefix}_intercept"])
    topic_vals = [float(row[f"topic_f_{year}"]) for year in early_years]
    fit_years = [year for year in early_years if year >= onset]
    fit_vals = [math.exp(intercept + slope * idx) - EPSILON for idx, _ in enumerate(fit_years)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(early_years, topic_vals, marker="o", label="topic_f")
    if fit_years and any(value > 0 for value in fit_vals):
        plt.plot(fit_years, fit_vals, linestyle="--", linewidth=2.0, label="exponential fit")
    plt.scatter([later_year], [float(row["ref_f_2024"])], marker="X", s=90, label="2024 ref validation")
    plt.xlabel("Year")
    plt.ylabel("frequency")
    plt.title(f"{str(row['candidate_topic'])[:72]} ({space}, onset {onset})")
    plt.xticks(early_years + [later_year])
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_top_n(scored: pd.DataFrame, space_dir: Path, space: str, args: argparse.Namespace) -> dict[str, Path]:
    image_dir = space_dir / "top30_frequency_individual"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    top = scored.head(args.top_n_plots).copy()
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        path = image_dir / f"{rank:02d}_{safe_name(str(row['candidate_topic']))}.png"
        plot_candidate(row, path, space, args.early_years, args.later_year)
        image_paths[str(row["candidate_id"])] = path
    return image_paths


def write_space_outputs(scored: pd.DataFrame, topic_slug: str, space: str, args: argparse.Namespace) -> None:
    space_dir = args.output_dir / topic_slug / space
    space_dir.mkdir(parents=True, exist_ok=True)
    scored_out = json_safe_df(scored)
    scored_out.to_csv(space_dir / "all_scored.csv", index=False)
    scored_out.head(args.top_n_plots).to_csv(space_dir / "top30_by_score.csv", index=False)
    plot_top_n(scored, space_dir, space, args)

    for gate_col, gate_dir_name in gate_labels(space).items():
        gate_dir = space_dir / gate_dir_name
        image_dir = gate_dir / "top30_failed_images"
        gate_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        for old_image in image_dir.glob("*.png"):
            old_image.unlink()
        failed = scored[~scored[gate_col]].copy()
        json_safe_df(failed).to_csv(gate_dir / "failed.csv", index=False)
        failed_top = failed.head(args.top_n_plots).copy()
        json_safe_df(failed_top).to_csv(gate_dir / "top30_failed_by_score.csv", index=False)
        for rank, (_, row) in enumerate(failed_top.iterrows(), start=1):
            path = image_dir / f"{rank:02d}_{safe_name(str(row['candidate_topic']))}.png"
            plot_candidate(row, path, space, args.early_years, args.later_year)

    passed_dir = space_dir / "passed_all_gates"
    passed_images = passed_dir / "frequency_individual"
    passed_dir.mkdir(parents=True, exist_ok=True)
    passed_images.mkdir(parents=True, exist_ok=True)
    for old_image in passed_images.glob("*.png"):
        old_image.unlink()
    passed = scored[scored["passed_all_gates"]].copy()
    json_safe_df(passed).to_csv(passed_dir / "weak_signals.csv", index=False)
    write_markdown(passed, passed_dir / "weak_signals.md", space)
    for rank, (_, row) in enumerate(passed.iterrows(), start=1):
        path = passed_images / f"{rank:02d}_{safe_name(str(row['candidate_topic']))}.png"
        plot_candidate(row, path, space, args.early_years, args.later_year)


def compute_topic_metrics(topic_slug: str, candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    years = sorted(set(args.early_years + [args.later_year]))
    paper_index, year_stats, non_survey_ids = load_topic_papers(args.papers_dir, topic_slug, years, args.papers_suffix)
    topic_candidates = candidates[candidates["target_topic_slug"].astype(str) == topic_slug].copy()

    match_path = reference_match_path(args.matching_dir, topic_slug, args.matching_suffix)
    matches = read_reference_matches(match_path)
    later_counts = later_counts_from_matches(matches, topic_slug, args.later_year, non_survey_ids.get(args.later_year, set()))
    if not later_counts:
        later_counts = later_counts_from_reference_cache(
            args,
            topic_slug,
            topic_candidates,
            paper_index,
            non_survey_ids.get(args.later_year, set()),
        )

    rows: list[dict[str, Any]] = []
    for _, candidate in topic_candidates.iterrows():
        counts, ids_by_year, missing_ids = source_counts_by_year(candidate, paper_index, args.early_years)
        row = candidate.to_dict()
        row["missing_source_paper_ids"] = missing_ids
        for year in args.early_years:
            n = counts.get(year, 0)
            stats = year_stats.get(year, {"total": 0, "survey_excluded": 0, "non_survey": 0})
            denom = stats["non_survey"]
            row[f"n_{year}"] = n
            row[f"N_{year}"] = denom
            row[f"N_total_{year}"] = stats["total"]
            row[f"N_survey_excluded_{year}"] = stats["survey_excluded"]
            row[f"f_{year}"] = n / denom if denom else 0.0
            row[f"topic_f_{year}"] = row[f"f_{year}"]
            row[f"topic_n_{year}"] = n
            row[f"source_paper_ids_{year}"] = ids_by_year.get(year, [])
        later_n = int(later_counts.get(str(candidate["candidate_id"]), 0))
        later_stats = year_stats.get(args.later_year, {"total": 0, "survey_excluded": 0, "non_survey": 0})
        later_denom = later_stats["non_survey"]
        row[f"ref_n_{args.later_year}"] = later_n
        row[f"ref_N_{args.later_year}"] = later_denom
        row[f"ref_f_{args.later_year}"] = later_n / later_denom if later_denom else 0.0
        row["ref_f_2024"] = row[f"ref_f_{args.later_year}"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args)
    topic_slugs = sorted(candidates["target_topic_slug"].dropna().astype(str).unique())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for topic_slug in topic_slugs:
        metrics = compute_topic_metrics(topic_slug, candidates, args)
        if metrics.empty:
            continue
        for candidate_type, space in [("problem-space", "problem"), ("solution-space", "solution")]:
            subset = metrics[metrics["candidate_topic_type"].astype(str) == candidate_type].copy()
            if subset.empty:
                continue
            scored = add_scores_and_gates(subset, space, args)
            write_space_outputs(scored, topic_slug, space, args)
            print(
                f"{topic_slug}/{space}: scored={len(scored):,}, "
                f"passed={int(scored['passed_all_gates'].sum()):,}, "
                f"output={args.output_dir / topic_slug / space}"
            )


if __name__ == "__main__":
    main()
