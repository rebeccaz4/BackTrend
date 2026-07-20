"""
Mainframe topics loaded from construction/topics.json.

Provides:
    TOPICS_BY_DOMAIN  - dict[domain_name, list[topic_name]]
    ALL_DOMAINS       - list of all domain names
    make_domain_slug  - convert domain name to filesystem-safe slug
"""
import json
import re
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_JSON_PATH = _SCRIPT_DIR / "../construction/topics.json"
DEFAULT_DOMAIN = "Artificial intelligence and machine learning"


def _load_topics() -> list[str]:
    with open(_JSON_PATH, encoding="utf-8") as f:
        topics = json.load(f)
    if not isinstance(topics, dict):
        raise TypeError(f"Expected a JSON object in {_JSON_PATH}")
    return list(topics.keys())


TOPICS_BY_DOMAIN: dict[str, list[str]] = {DEFAULT_DOMAIN: _load_topics()}
ALL_DOMAINS: list[str] = [DEFAULT_DOMAIN]


def make_domain_slug(domain: str) -> str:
    """Convert a domain name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
