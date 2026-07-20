#!/usr/bin/env python
# coding: utf-8
"""
Shared prompt templates and utilities for all weak-signal prediction scripts.

All prediction scripts import from here to guarantee identical prompts.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YEAR_RANGE = "2019-2023"
YEAR_SLUG = "2019_2023"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROBLEM_PROMPT_TEMPLATE = """\
You are an expert analyst of frontier {domain} research. Your task is to identify early weak signals that later contributed to a specified mature target topic.

We distinguish two categories of weak signals. One category is the "solution-space weak signal": an early research method, technique, or design principle that was not yet widely adopted but later became an important solution to an already-recognized problem. 

However, the category you are asked to identify here is the "problem-space weak signal": an underrecognized research problem or problem formulation that was not yet widely recognized by the research community at the time it emerged, but that later became central to the mature target topic [{mainframe_topic}].
To be more specific, problem-space weak signals include research problems, gaps, limitations, risks, bottlenecks, evaluation failures, or scientific questions that emerged in the literature during the prediction window, rather than claims tied to a single paper.

Mature target topic:
[{mainframe_topic}]

Prediction window:
[{year_range}]

Retrospective setup:
The mature target topic should be treated as a topic that is already established or prominent by 2024. Your task is not to predict after 2024. Your task is to look backward and identify what this 2024 mature topic looked like in the 2019-2023 literature while it was still emerging.
Use the mature target topic only as 2024 relevance context. The weak signals themselves must be research problems, gaps, limitations, risks, bottlenecks, or scientific questions that appeared in [{year_range}]. Do not use 2024-or-later evidence, and do not simply restate the mature target topic unless you name a more specific predecessor formulation.

Question:
What are the early problem-space weak signals in {domain} that emerged between [{year_range}] and later contributed to the mature target topic [{mainframe_topic}]?

Specificity guidance:
- Use the mature target topic only as relevance context.
- Do not output the target topic itself unless you name a more specific reusable subtopic or predecessor direction.
- Too broad: a whole field (e.g., "machine learning", "computer vision"), broad model family (e.g., "deep learning"), or generic category label (e.g., "optimization").
- Too specific: a paper-specific method name, system name, exact dataset, benchmark instance, implementation detail, single experimental finding, or single case study.
- Correct level: a reusable research direction or problem space topic that multiple independent papers could study using different methods or systems.
- Focus on the research problem, not on specific implementation details or data modalities.
- Do not phrase signals as actions, paper contributions, or problem-solution relationships.
- Avoid "X for Y" signal names when X is a method and Y is a problem, goal, task, or desired property. For problem-space output, keep only the problem side (i.e., Y) when it is independently reusable and relevant.

Requirements:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Output a JSON object with a "weak_signals" array, which may be empty.
- Each weak signal must have exactly these fields: "signal", "what_it_was", "why_weak_signal".
- Each weak signal must be explicitly tied to the prediction window [{year_range}].
- "what_it_was" must include the year or year range within [{year_range}].
- "why_weak_signal" must explain why this was a problem-space weak signal for [{mainframe_topic}].
- Each signal must be conceptually related to [{mainframe_topic}].
- Each signal must be reusable across multiple papers.
- Do not include solution methods.
- Do not invent evidence or overclaim certainty.

Return only JSON with this schema:
{{
  "weak_signals": [
    {{
      "signal": "<short reusable problem-space weak signal name>",
      "what_it_was": "<1-2 sentences describing what it was, including the year>",
      "why_weak_signal": "<1-2 sentences explaining why it was a problem-space weak signal for [{mainframe_topic}]>"
    }}
  ]
}}
"""

SOLUTION_PROMPT_TEMPLATE = """\
You are an expert analyst of frontier {domain} research. Your task is to identify early weak signals that later contributed to a specified mature target topic.

We distinguish two categories of weak signals. One category is the "problem-space weak signal": an underrecognized research problem or problem formulation that was not yet widely recognized by the research community at the time it emerged, but that later became central to the mature target topic [{mainframe_topic}].

However, the category you are asked to identify here is the "solution-space weak signal": an early research method, technique, or design principle that was not yet widely adopted but later became an important solution to an already-recognized problem.
To be more specific, solution-space weak signals include research methods, method families, system directions, evaluation approaches, defenses, or solution directions that emerged in the literature during the prediction window, rather than claims tied to a single paper.

Mature target topic:
[{mainframe_topic}]

Prediction window:
[{year_range}]

Retrospective setup:
The mature target topic should be treated as a topic that is already established or prominent by 2024. Your task is not to predict after 2024. Your task is to look backward and identify what this 2024 mature topic looked like in the 2019-2023 literature while it was still emerging.
Use the mature target topic only as 2024 relevance context. The weak signals themselves must be research methods, method families, system directions, evaluation approaches, defenses, or solution directions that appeared in [{year_range}]. Do not use 2024-or-later evidence, and do not simply restate the mature target topic unless you name a more specific predecessor formulation.

Question:
What are the early solution-space weak signals in {domain} that emerged between [{year_range}] and later contributed to the mature target topic [{mainframe_topic}]?

Specificity guidance:
- Use the mature target topic only as relevance context.
- Do not output the target topic itself unless you name a more specific reusable subtopic or predecessor direction.
- Too broad: a whole field (e.g., "machine learning", "computer vision"), broad model family (e.g., "deep learning"), or generic category label (e.g., "optimization").
- Too specific: a paper-specific method name, system name, exact dataset, benchmark instance, implementation detail, single experimental finding, or single case study.
- Correct level: a reusable research method, method family, system direction, evaluation approach, defense, or solution direction that multiple independent papers could study.
- Focus on the research method or solution direction, not on specific implementation details, problem formulations, or data modalities.
- Do not phrase signals as actions, paper contributions, or problem-solution relationships.
- Avoid "X for Y" signal names when X is a method and Y is a problem, goal, task, or desired property. For solution-space output, keep only the solution side (i.e., X) when it is independently reusable and relevant.

Requirements:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Output a JSON object with a "weak_signals" array, which may be empty.
- Each weak signal must have exactly these fields: "signal", "what_it_was", "why_weak_signal".
- Each weak signal must be explicitly tied to the prediction window [{year_range}].
- "what_it_was" must include the year or year range within [{year_range}].
- "why_weak_signal" must explain why this was a solution-space weak signal for [{mainframe_topic}].
- Each signal must be conceptually related to [{mainframe_topic}].
- Each signal must be reusable across multiple papers.
- Do not include problem statements.
- Do not invent evidence or overclaim certainty.

Return only JSON with this schema:
{{
  "weak_signals": [
    {{
      "signal": "<short reusable solution-space weak signal name>",
      "what_it_was": "<1-2 sentences describing what it was, including the year>",
      "why_weak_signal": "<1-2 sentences explaining why it was a solution-space weak signal for [{mainframe_topic}]>"
    }}
  ]
}}
"""
PROMPT_TEMPLATES = {
    "problem": PROBLEM_PROMPT_TEMPLATE,
    "solution": SOLUTION_PROMPT_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def make_topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def build_prompt(space: str, domain: str, mainframe_topic: str) -> str:
    return PROMPT_TEMPLATES[space].format(
        domain=domain,
        mainframe_topic=mainframe_topic,
        year_range=YEAR_RANGE,
    )


def strip_reasoning_blocks(text: str) -> str:
    """Return final-answer text after Qwen-style reasoning blocks."""
    if not isinstance(text, str):
        return ""
    if not re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        return text.strip()
    parts = re.split(r"</think\s*>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return ""


def _valid_weak_signal_payload(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("weak_signals"), list)


def load_weak_signal_json(text: str) -> Dict[str, Any] | None:
    """Load the final weak-signal JSON object, ignoring reasoning text."""
    clean_text = strip_reasoning_blocks(text)
    if not clean_text:
        return None

    try:
        data = json.loads(clean_text)
        if _valid_weak_signal_payload(data):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean_text)
    if m:
        try:
            data = json.loads(m.group(1))
            if _valid_weak_signal_payload(data):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean_text):
        try:
            data, _ = decoder.raw_decode(clean_text[match.start():])
        except json.JSONDecodeError:
            continue
        if _valid_weak_signal_payload(data):
            return data

    return None


def _signals_from_payload(data: Dict[str, Any]) -> List[str]:
    signals: List[str] = []
    for ws in data.get("weak_signals", []):
        if isinstance(ws, dict):
            s = (ws.get("signal") or "").strip()
            if s:
                signals.append(s)
    return signals


def extract_candidate_signals(text: str) -> List[str]:
    """Extract weak-signal names from model response (JSON preferred, regex fallback)."""
    signals: List[str] = []

    payload = load_weak_signal_json(text)
    if payload is not None:
        return _signals_from_payload(payload)

    clean_text = strip_reasoning_blocks(text)
    if not clean_text:
        return []

    # 3) Regex for "signal": "..." fields. Some DR-Tulu outputs use curly
    # quotes, which are not valid JSON but still preserve the signal fields.
    signal_field_re = re.compile(r'["“]signal["”]\s*:\s*["“]([^"“”]+)["”]')
    signals.extend(s.strip() for s in signal_field_re.findall(clean_text) if s.strip())

    # 4) Numbered list lines
    for line in clean_text.splitlines():
        m2 = re.match(r"^\s*\d+[.)]+\s+(.+?)\s*$", line)
        if m2:
            s = m2.group(1).strip()
            if s:
                signals.append(s)

    seen: set[str] = set()
    deduped: List[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped
