#!/usr/bin/env python
# coding: utf-8
"""
GPT-5.4-chat – weak-signal prediction (prediction only, no evaluation).

Usage example:
    python gpt5.4.py \
        --domain "Artificial intelligence and machine learning" \
        --spaces problem solution \
        --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from prediction_prompts import (
    YEAR_RANGE,
    YEAR_SLUG,
    build_prompt,
    extract_candidate_signals,
    make_topic_slug,
)

DEFAULT_MODEL = "gpt-5.4"
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
OUTPUT_MODEL_DIR = "gpt5.4"


def resolve_api_key(api_key_override: str | None = None) -> str:
    api_key = api_key_override or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY, or pass --api-key.")
    return api_key


# ---------------------------------------------------------------------------
# OpenAI-compatible API call
# ---------------------------------------------------------------------------

def run_model_once(
    client,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 8,
    retry_backoff: float = 3.0,
) -> str:
    """Call an OpenAI-compatible chat completion with retry logic."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not resp.choices:
                raise RuntimeError("Model provider returned no choices.")
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Model provider returned empty text.")
            return text
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Model provider request failed after {attempt} attempts: {exc}"
                ) from exc
            sleep_s = retry_backoff ** attempt
            print(f"  [warn] attempt {attempt}: {exc}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    domain: str,
    space: str,
    topic: str,
    response_text: str,
    signals: List[str],
    model: str,
) -> Path:
    from mainframe_topics import make_domain_slug
    domain_slug = make_domain_slug(domain)
    topic_slug = make_topic_slug(topic)
    result_dir = output_dir / OUTPUT_MODEL_DIR / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (result_dir / f"response_{timestamp}.txt").write_text(
        response_text, encoding="utf-8"
    )
    (result_dir / "response_latest.txt").write_text(
        response_text, encoding="utf-8"
    )

    signals_payload = {
        "domain": domain,
        "space": space,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "model": model,
        "provider": "openai",
        "signals": signals,
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
    p = argparse.ArgumentParser(
        description="GPT-5.4 weak-signal prediction via the OpenAI API (no evaluation)."
    )
    p.add_argument(
        "--domain", nargs="+", default=None,
        help="Domain(s) to predict. If omitted, runs all domains.",
    )
    p.add_argument(
        "--spaces", nargs="+", default=["problem", "solution"],
        choices=["problem", "solution"],
        help="Signal spaces to predict (default: problem solution).",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--model", "--openai-model", dest="model",
        default=DEFAULT_MODEL,
        help="OpenAI model id (default: gpt-5.4).",
    )
    p.add_argument(
        "--api-key", "--openai-api-key", dest="api_key",
        default=None,
        help="OpenAI API key (default: $OPENAI_API_KEY).",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="Maximum output tokens. GPT-5.4 requires >= 16.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-backoff", type=float, default=2.0)
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

    if args.max_tokens < 16:
        raise RuntimeError("Invalid --max-tokens: GPT-5.4 requires a value >= 16.")

    api_key = resolve_api_key(args.api_key)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=120.0)

    print("=" * 60)
    print(f"Domains:     {domains}")
    print(f"Spaces:      {args.spaces}")
    print(f"Year range:  {YEAR_RANGE}")
    print(f"Model:       {args.model}")
    print(f"Base URL:    {OFFICIAL_OPENAI_BASE_URL}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 60)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain)
        if topics is None:
            print(f"[warn] Unknown domain: {domain}. Skipping.")
            continue
        domain_slug = make_domain_slug(domain)

        for topic in topics:
            for space in args.spaces:
                print(f"\n{'─' * 60}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'─' * 60}")

                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / OUTPUT_MODEL_DIR / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Already exists: {result_dir}")
                    continue

                prompt = build_prompt(space, domain, topic)
                print(f"Prompt length: {len(prompt)} chars")
                print(f"Calling {args.model} ...")

                response_text = run_model_once(
                    client=client,
                    model=args.model,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                print(f"Response length: {len(response_text)} chars")

                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                result_dir = save_results(
                    args.output_dir, domain, space, topic,
                    response_text, signals,
                    model=args.model,
                )
                print(f"Results saved to: {result_dir}")
                time.sleep(10)

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
