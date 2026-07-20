#!/usr/bin/env python
"""Build construction/weak_signal.json from the released BackTrend dataset.

The evaluation stage reads the human-validated ground truth from
construction/weak_signal.json, shaped as

    { "<topic display name>": { "problem": [...], "solution": [...] } }

The released dataset on Hugging Face
(https://huggingface.co/datasets/rebeccazzzz/BackTrend) ships the same signals
as signals.jsonl (one record per signal, with grounding papers). This script
converts signals.jsonl into the weak_signal.json shape.

Usage:
    # from a local copy of signals.jsonl
    python construction/scripts/build_weak_signal_json.py --input signals.jsonl

    # or download it from Hugging Face first
    python construction/scripts/build_weak_signal_json.py --download
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_CONSTRUCTION_DIR / "weak_signal.json"
SIGNALS_URL = (
    "https://huggingface.co/datasets/rebeccazzzz/BackTrend/resolve/main/signals.jsonl"
)
SPACES = ("problem", "solution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a local signals.jsonl downloaded from the Hugging Face dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download signals.jsonl from {SIGNALS_URL} instead of reading --input.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> list[dict]:
    if args.download:
        import requests

        response = requests.get(SIGNALS_URL, timeout=120)
        response.raise_for_status()
        text = response.text
    elif args.input:
        text = args.input.read_text(encoding="utf-8")
    else:
        raise SystemExit("Pass --input <path-to-signals.jsonl> or --download.")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    records = load_records(args)

    gt: dict[str, dict[str, list[str]]] = {}
    for record in records:
        topic = record["topic"]
        space = record["space"]
        signal = record["candidate_topic"]
        if space not in SPACES:
            raise ValueError(f"Unexpected space {space!r} for topic {topic!r}.")
        signals = gt.setdefault(topic, {}).setdefault(space, [])
        if signal not in signals:
            signals.append(signal)

    args.output.write_text(
        json.dumps(gt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_signals = sum(len(v) for spaces in gt.values() for v in spaces.values())
    print(f"Wrote {args.output}: {len(gt)} topics, {n_signals} signals.")


if __name__ == "__main__":
    main()
