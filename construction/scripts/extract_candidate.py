#!/usr/bin/env python
"""Extract candidate problem/solution topics from fetched topic papers.

Run examples:
  python construction/scripts/extract_candidate.py
  python construction/scripts/extract_candidate.py --topic "trustworthy AI"
  python construction/scripts/extract_candidate.py --topic "trustworthy AI" --year 2023
  python construction/scripts/extract_candidate.py --topic "trustworthy AI" --year 2023 --max-papers 3
  python construction/scripts/extract_candidate.py --topic "trustworthy AI" --year 2023 --force
  python construction/scripts/extract_candidate.py --topic "large language models" --papers-suffix _with_reference --output-suffix _with_reference
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_PAPERS_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_topics"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


SYSTEM_PROMPT = """You are an expert research topic extractor.
Your task is to extract literature-level research topics from paper abstracts.
Extract only topics that are explicitly discussed in the paper.
A good topic should be broad enough that multiple independent papers could study it,
but specific enough to be more informative than a general field label.
Prefer clean, compact topic labels that name the core research problem or method family.
Return only valid json.
"""

USER_PROMPT_TEMPLATE = """
Target established topic:
{target_topic}

This paper was retrieved as related to the target established topic above.
Use the target topic only as context for relevance filtering.
Do not output the target topic itself unless the abstract discusses a more specific reusable subtopic.

Paper metadata:
- Title: {title}
- Paper ID: {paper_id}
- Year: {year}
- Venue: {venue}
- Source query: {source_query}

Abstract:
{abstract}

Topic categories:

1. Problem-space topics:
Research problems, gaps, limitations, risks, bottlenecks, evaluation failures, or scientific questions discussed by the paper.

2. Solution-space topics:
Research methods, method families, system directions, evaluation approaches, defenses, or solution directions discussed by the paper.
Use solution-space only for standalone reusable methods, method families, systems, defenses, algorithms, datasets, benchmarks, or evaluation protocols, not for the problem that motivates them.

Task:
Extract clean, reusable research topics from this paper abstract that are conceptually related to the target established topic: "{target_topic}".
These candidates may be problem-space topics or solution-space topics.
Most abstracts describe both a problem/gap and a method/solution. Separate these roles.
Do not combine a method, remedy, evaluation detail, dataset, application setting, or implementation detail with the problem it addresses.

Specificity guidance:
- Too broad: a whole field (e.g., "machine learning", "computer vision"), broad model family (e.g., "deep learning"), or generic category label (e.g., "optimization").
- Too specific: a paper-specific method name (e.g., "CoPINet"), system name (e.g., "GPT Semantic Cache"), exact task setting (e.g., "Raven's Progressive Matrices"), implementation detail, single experimental finding (e.g., "68.8% API-call reduction"), or enumerating technical details (e.g., "using interventional data", "with linguistically regularized CNN", "additive feature attribution").
- Correct level: a reusable research direction or problem space that multiple independent papers could study using different methods, or systems. The topic should capture the core research direction without enumerating specific techniques or data types.
- Prefer compact labels such as "causal model evaluation" over long contribution phrases such as "evaluation of causal models using interventional empirical data".
- Focus on the research problem or method family, not on the specific implementation details or data modalities.
- If the abstract discusses a narrow technique or case study, abstract it to the broader research problem or method family it addresses.
- Do not phrase topics as actions (e.g., "evaluation of", "analysis of") or as this specific paper's contribution.
- A candidate topic must be a standalone research topic phrase, not a relation between a problem and a solution.
- Avoid "X for Y" topic names when X is a method and Y is a problem, goal, task, or desired property. Split them into separate candidate topics when both are explicitly supported.
- If the paper proposes method X to solve, evaluate, improve, or measure problem Y, output Y as a problem-space topic and X as a solution-space topic only when each is independently reusable. Do not output the combined phrase.

Bad topic examples (wrong abstraction level):
- "machine learning" because it is too broad
- "Empirical evaluation of causal models using interventional data" because it enumerates technical details ("using interventional data")
- "evaluation of causal modeling algorithms using interventional empirical data" because it mixes the core problem with proposed evaluation details; prefer "causal model evaluation" as the problem-space topic.
- "CoPINet for Raven's Progressive Matrices" because it is paper-specific
- "Aspect-based sentiment analysis with linguistically regularized CNN" because it enumerates method details
- "Explainability challenges in additive feature attribution" because it is too method-specific
- "deep reinforcement learning for federated edge learning resource management" because it combines a solution method with a problem/context; split into cleaner topics only if each side is independently reusable and explicitly supported.
- "Using interaction-aware explanations to solve misleading model interpretability" because it mixes a solution and a problem in one extracted topic; extract the problem topic and solution topic separately if both are explicitly supported.
- "uncertainty communication for AI trust calibration" because it mixes a solution method with the problem or goal it addresses; extract "AI trust calibration" as a problem-space topic or "uncertainty communication in AI systems" as a solution-space topic if each is explicitly supported.

Requirements:
- Output a json object with a "topics" array, which may be empty.
- Extract up to 2 topics total.
- Each topic must be explicitly supported by the abstract.
- Each topic must be reusable across multiple papers.
- Each topic must be one abstraction level broader than the paper's specific method, benchmark, dataset, or case study.
- Each topic must include exactly one "topic_type": "problem-space" or "solution-space". A mix is not allowed.
- Each topic must be conceptually related to the target established topic: "{target_topic}".
- If a topic cannot be clearly classified as problem-space or solution-space, do not emit it.
- If a topic cannot be clearly related to the target established topic, do not emit it.
- Do not phrase topics as actions, paper contributions, or problem-solution relationships.
- Do not combine a method and the problem it addresses into one topic phrase.
- Do not include proposed measurement choices, dataset choices, application settings, or implementation details in the topic label unless they are themselves the reusable research topic.
- Do not invent evidence beyond the abstract.

Return only json with this schema:
{{
  "topics": [
    {{
      "topic": "<standalone literature-level candidate topic>",
      "topic_type": "problem-space|solution-space",
      "target_topic": "{target_topic}",
      "evidence": "<short phrase grounded in the abstract>",
      "confidence": "high|medium|low"
    }}
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate problem/solution topics from fetched topic papers."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--papers-suffix", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--topic",
        help="Process only one topic from --topics-json.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years to process. Default: 2019 2020 2021 2022 2023.",
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Single year to process. Can be repeated. Overrides --years.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum papers to process per topic-year. Default: all papers.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts per paper. Default: 3.",
    )
    parser.add_argument(
        "--openai-retries",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def model_slug(model: str) -> str:
    return slugify(model).replace("-", "")


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


def clean(value: object, fallback: str = "") -> str:
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else fallback


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {"topics": []}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"topics": []}


def papers_path(
    papers_dir: Path,
    topic: str,
    year: int,
    suffix: str = "",
) -> Path:
    topic_slug = slugify(topic)
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def output_path(
    output_dir: Path,
    topic: str,
    year: int,
    suffix: str = "",
) -> Path:
    topic_slug = slugify(topic)
    return output_dir / topic_slug / f"candidate_topics_{topic_slug}_{year}{suffix}.jsonl"


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(f"{path.suffix}.bak.{int(time.time())}")
    shutil.move(str(path), str(backup))
    print(f"Backed up existing file: {backup}")


def load_completed_paper_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = (payload.get("paper") or {}).get("paperId")
            if paper_id:
                completed.add(str(paper_id))
    return completed


def load_papers(path: Path, max_papers: int | None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "paperId" in df.columns:
        df["paperId"] = df["paperId"].astype(str)
    if "abstract" in df.columns:
        df = df[df["abstract"].fillna("").astype(str).str.strip().ne("")].copy()
    if max_papers is not None:
        df = df.head(max_papers).copy()
    return df.reset_index(drop=True)


def build_messages(row: pd.Series, target_topic: str) -> list[dict[str, str]]:
    source_query = clean(row.get("source_query") or row.get("query_text"), "N/A")
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_topic=target_topic,
        title=clean(row.get("title"), "Unknown title"),
        paper_id=clean(row.get("paperId")),
        year=clean(row.get("year")),
        venue=clean(row.get("venue"), "Unknown venue"),
        source_query=source_query,
        abstract=clean(row.get("abstract")),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def make_client(api_key: str, base_url: str | None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    return OpenAI(api_key=api_key, timeout=120.0)


def model_provider() -> dict[str, str | None]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY to run candidate extraction.")
    return {
        "name": "openai",
        "api_key": openai_key,
        "base_url": OFFICIAL_OPENAI_BASE_URL,
    }


def call_model(
    provider: dict[str, str | None],
    model: str,
    messages: list[dict[str, str]],
    paper_id: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str]:
    last_exc: Exception | None = None
    provider_name = provider["name"] or "unknown"
    api_key = provider["api_key"]
    base_url = provider["base_url"]
    if not api_key:
        raise RuntimeError(f"Missing API key for provider: {provider_name}")

    for attempt in range(1, max_retries + 1):
        started = time.time()
        try:
            client = make_client(api_key, base_url)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=1600,
            )
            print(
                f"{paper_id}: {provider_name} response in "
                f"{time.time() - started:.1f}s on attempt {attempt}",
                flush=True,
            )
            return safe_json_loads(response.choices[0].message.content or ""), provider_name
        except Exception as exc:
            last_exc = exc
            print(
                f"{paper_id}: {provider_name} model call failed on attempt {attempt} "
                f"after {time.time() - started:.1f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < max_retries:
                sleep(min(retry_sleep * attempt, 20))

    if last_exc is not None:
        raise RuntimeError(
            f"Model call failed for provider {provider_name} after {max_retries} attempts: {last_exc}"
        ) from last_exc
    return {"topics": []}, "none"


def is_mixed_problem_solution_topic(topic: str) -> bool:
    text = re.sub(r"\s+", " ", topic.strip().lower())
    if not text:
        return False

    explicit_patterns = [
        r"^using .+ for .+",
        r"^methods? to solve .+",
        r"^solutions? for .+",
        r"^frameworks? for .+",
        r"^applications? of .+ to .+",
        r".+ to solve .+",
    ]
    if any(re.search(pattern, text) for pattern in explicit_patterns):
        return True

    return False


def normalize_candidate_topics(payload: dict[str, Any]) -> list[dict[str, str]]:
    topics = payload.get("topics")
    if not isinstance(topics, list):
        topics = payload.get("weak_signals")
    if not isinstance(topics, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic = clean(item.get("topic") or item.get("weak_signal"))
        if is_mixed_problem_solution_topic(topic):
            continue
        target_topic = clean(item.get("target_topic"))
        evidence = clean(item.get("evidence"))
        confidence = clean(item.get("confidence")).lower()
        topic_type = clean(item.get("topic_type") or item.get("signal_type"), "unclear")

        if not topic or not evidence:
            continue
        if topic_type not in {"problem-space", "solution-space"}:
            continue
        if confidence not in {"high", "medium", "low"}:
            confidence = "unclear"
        if not target_topic or target_topic == "none":
            continue

        normalized.append(
            {
                "topic": topic,
                "topic_type": topic_type,
                "target_topic": target_topic,
                "evidence": evidence,
                "confidence": confidence,
            }
        )
    return normalized[:2]


def paper_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "paperId": clean(row.get("paperId")),
        "title": clean(row.get("title"), "Unknown title"),
        "year": clean(row.get("year")),
        "venue": clean(row.get("venue")),
        "url": clean(row.get("url")),
        "abstract": clean(row.get("abstract")),
        "query_text": clean(row.get("query_text")),
        "query_type": clean(row.get("query_type")),
        "matched_queries": clean(row.get("matched_queries")),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_topic_year(
    *,
    topic: str,
    year: int,
    args: argparse.Namespace,
    provider: dict[str, str | None],
) -> dict[str, Any]:
    in_path = papers_path(
        args.papers_dir,
        topic,
        year,
        args.papers_suffix,
    )
    out_path = output_path(
        args.output_dir,
        topic,
        year,
        args.output_suffix,
    )
    if args.force:
        backup_file(out_path)

    papers = load_papers(in_path, args.max_papers)
    if papers.empty:
        print(f"{topic} | {year}: no papers found at {in_path}")
        return {"topic": topic, "year": year, "input_papers": 0, "processed": 0, "skipped": 0}

    completed = set() if args.force else load_completed_paper_ids(out_path)
    to_process = papers[~papers["paperId"].astype(str).isin(completed)].copy()
    print(
        f"{topic} | {year}: papers={len(papers)}, completed={len(completed)}, "
        f"remaining={len(to_process)}"
    )

    processed = 0
    for _, row in tqdm(
        to_process.iterrows(),
        total=len(to_process),
        desc=f"{topic} | {year}",
        unit="paper",
    ):
        paper_id = clean(row.get("paperId"))
        raw_response, provider_used = call_model(
            provider=provider,
            model=args.model,
            messages=build_messages(row, topic),
            paper_id=paper_id,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        result = {
            "target_topic": topic,
            "target_topic_slug": slugify(topic),
            "query_year": year,
            "paper": paper_payload(row),
            "candidate_topics": normalize_candidate_topics(raw_response),
            "raw_response": raw_response,
            "model": args.model,
            "provider": provider_used,
        }
        append_jsonl(out_path, result)
        processed += 1

    return {
        "topic": topic,
        "year": year,
        "input_papers": len(papers),
        "processed": processed,
        "skipped": len(papers) - len(to_process),
        "output_path": str(out_path),
    }


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    years = sorted(set(args.year or args.years))
    topics = load_topics(args.topics_json, args.topic)
    provider = model_provider()

    print(f"Topics: {len(topics)}")
    print(f"Years: {years}")
    print(f"Model: {args.model}")
    print(f"Model provider: {provider['name']}")
    print(f"Max retries: {args.max_retries}")
    print(f"Max papers per topic-year: {args.max_papers if args.max_papers is not None else 'all'}")

    summaries = []
    for topic in topics:
        for year in years:
            summaries.append(
                process_topic_year(
                    topic=topic,
                    year=year,
                    args=args,
                    provider=provider,
                )
            )

    print("Done.")
    for item in summaries:
        print(
            f"{item['topic']} | {item['year']}: processed={item['processed']} "
            f"skipped={item['skipped']}"
        )


if __name__ == "__main__":
    main()
