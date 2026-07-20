"""Shared prompts for the LLM-as-judge evaluation.

Both the set-level judge (Setting 2, llm_set_eval.py) and the signal-level
judge (Setting 4, llm_signal_eval.py) use the SAME system prompt — only the
user message differs. Rules only, no examples; kept as simple as possible.
"""
from __future__ import annotations

# Identical system prompt for both judges. A match requires the two research
# topics to denote the *same specific* topic, not merely related ones.
SYSTEM_PROMPT = (
    "You are a research topic matcher. Two research topics match only if they "
    "denote the same specific research topic: the same core method or the same "
    "problem, differing at most in wording, phrasing, or abbreviation. Topics "
    "that are merely related, adjacent, complementary, or from the same broad "
    "area do not match."
)


# Setting 2 — set-level judge: model returns precision/recall directly, plus
# the matched pairs as evidence.
SET_USER_TEMPLATE = """Compare the PREDICTED set against the GROUND TRUTH set of research topics.

Precision = fraction of predicted topics that match some ground-truth topic.
Recall = fraction of ground-truth topics that match some predicted topic.

Ground truth:
{gt_block}

Predicted:
{pred_block}

Return ONLY valid JSON:
{{"precision": <float 0-1>, "recall": <float 0-1>, "matched_pairs": [{{"gt_index": <int>, "pred_index": <int>}}]}}
Give one matched_pairs entry per matching (ground-truth, predicted) pair; empty list if none."""


# Setting 4 — signal-level judge: per-candidate 0/1 match against the reference set.
SIGNAL_USER_TEMPLATE = """For each candidate research topic, decide whether it matches any reference research topic.

Reference:
{ref_block}

Candidate:
{cand_block}

Return ONLY valid JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements, 1 if the candidate matches any reference else 0."""


def format_block(signals: list[str]) -> str:
    """Number a list of topics, 0-based (shared by both judges)."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(signals))


def build_set_prompt(gt: list[str], pred: list[str]) -> str:
    """Render the set-level (Setting 2) user message."""
    return SET_USER_TEMPLATE.format(
        gt_block=format_block(gt),
        pred_block=format_block(pred),
    )


def build_pairwise_prompt(reference: list[str], candidates: list[str]) -> str:
    """Render the signal-level (Setting 4) user message for one batch."""
    return SIGNAL_USER_TEMPLATE.format(
        ref_block=format_block(reference),
        cand_block=format_block(candidates),
        n_cand=len(candidates),
    )
