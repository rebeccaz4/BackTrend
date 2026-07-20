"""Data loaders for GT and prediction signals.

GT now comes from a single file: construction/weak_signal.json, whose shape is

    {
        "<topic display name>": {
            "problem":  ["signal", ...],   # optional
            "solution": ["signal", ...]    # optional
        },
        ...
    }

Only human-validated ("valid") weak signals are present. A (topic, direction)
combination with no signals in this file is treated as "no GT".

Predictions still live under
    prediction/outputs/<model>/<domain_slug>/<topic_slug>/<direction>/<year_slug>/signals_latest.json
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
GT_PATH = REPO_ROOT / "construction" / "weak_signal.json"
PREDICTION_ROOT = REPO_ROOT / "prediction" / "outputs"
YEAR_SLUG = "2019_2023"
DIRECTIONS = ("problem", "solution")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_gt_index(gt_path: Path = GT_PATH) -> dict[str, dict]:
    """Load weak_signal.json and index it by slugified topic name.

    Returns {topic_slug: {"topic_display": str,
                          "problem": list[str], "solution": list[str]}}.
    """
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}\n"
            "It ships with the repo; regenerate it from the Hugging Face dataset "
            "(rebeccazzzz/BackTrend) with:\n"
            "    python construction/scripts/build_weak_signal_json.py --download"
        )
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for topic_display, spaces in data.items():
        slug = slugify(topic_display)
        index[slug] = {
            "topic_display": topic_display,
            "problem": list(spaces.get("problem", [])),
            "solution": list(spaces.get("solution", [])),
        }
    return index


def gt_signals_for(gt_index: dict[str, dict], topic_slug: str, direction: str) -> list[str]:
    """Return the validated GT signals for a (topic, direction); [] if none."""
    entry = gt_index.get(topic_slug)
    if not entry:
        return []
    return list(entry.get(direction, []))


def load_pred_signals(
    model: str,
    domain_slug: str,
    topic_slug: str,
    direction: str,
    prediction_root: Path = PREDICTION_ROOT,
    year_slug: str = YEAR_SLUG,
) -> list[str]:
    """Load predicted signals from signals_latest.json."""
    path = (
        prediction_root / model / domain_slug / topic_slug / direction / year_slug / "signals_latest.json"
    )
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("signals", [])


@dataclass
class EvalItem:
    model: str
    domain_slug: str
    topic_slug: str
    topic_display: str
    direction: str
    gt_signals: list[str] = field(default_factory=list)
    pred_signals: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.model}/{self.topic_slug}/{self.direction}"

    @property
    def has_gt(self) -> bool:
        return len(self.gt_signals) > 0

    @property
    def has_pred(self) -> bool:
        return len(self.pred_signals) > 0


def iter_eval_items(
    models: list[str] | None = None,
    prediction_root: Path = PREDICTION_ROOT,
    gt_path: Path = GT_PATH,
) -> Iterator[EvalItem]:
    """Yield one EvalItem for every (model, topic, direction) that a model
    produced predictions for.

    The full topic universe is taken from each model's prediction tree, so
    topics with no validated GT still surface (they will have has_gt == False
    and should be recorded as "no_gt" downstream rather than dropped).
    """
    gt_index = load_gt_index(gt_path)
    available_models = models or [
        d.name for d in sorted(prediction_root.iterdir()) if d.is_dir()
    ]

    for model in available_models:
        model_dir = prediction_root / model
        if not model_dir.is_dir():
            continue
        for domain_dir in sorted(model_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            for topic_dir in sorted(domain_dir.iterdir()):
                if not topic_dir.is_dir():
                    continue
                topic_slug = topic_dir.name
                gt_entry = gt_index.get(topic_slug)
                topic_display = gt_entry["topic_display"] if gt_entry else topic_slug
                for direction in DIRECTIONS:
                    gt = gt_signals_for(gt_index, topic_slug, direction)
                    pred = load_pred_signals(
                        model, domain_dir.name, topic_slug, direction, prediction_root
                    )
                    yield EvalItem(
                        model=model,
                        domain_slug=domain_dir.name,
                        topic_slug=topic_slug,
                        topic_display=topic_display,
                        direction=direction,
                        gt_signals=gt,
                        pred_signals=pred,
                    )
