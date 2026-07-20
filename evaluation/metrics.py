"""Metric aggregation helpers."""

from __future__ import annotations


def compute_metrics(
    precision_matches: list[int],
    recall_matches: list[int],
    n_gt: int,
    n_ext: int,
) -> dict:
    """Compute P/R/F1 from two binary match vectors.

    Args:
        precision_matches: 0/1 per external topic (does it match any GT?).
        recall_matches: 0/1 per GT topic (does it match any external?).
        n_gt: number of ground-truth topics.
        n_ext: number of external topics.

    Returns:
        dict with precision, recall, f1, precision_matched, recall_matched.
    """
    precision_matched = sum(precision_matches)
    recall_matched = sum(recall_matches)

    precision = precision_matched / n_ext if n_ext > 0 else 0.0
    recall = recall_matched / n_gt if n_gt > 0 else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_matched": precision_matched,
        "recall_matched": recall_matched,
        "n_ext": n_ext,
        "n_gt": n_gt,
    }
