"""Numeric constraint extraction from natural-language text."""

from __future__ import annotations

import re


def parse_max_atoms(text: str, default: int = 20) -> int:
    """Extract maximum atom count from constraint text.

    Args:
        text: Free-form constraint text.
        default: Value returned when no atom-count pattern is found.

    Returns:
        The extracted atom count, or *default* if no pattern matched.
    """
    if not text or not text.strip():
        return default

    m = re.search(
        r"(?:[<\u2264]=?\s*|up\s+to\s+|at\s+most\s+|max(?:imum)?\s+)"
        r"(\d+)\s*atoms?(?:/cell)?",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return default


def parse_stability_threshold(text: str, default: float = 0.1) -> float:
    """Extract stability threshold in eV/atom from constraint text.

    Args:
        text: Free-form constraint text.
        default: Value returned when no threshold pattern is found.

    Returns:
        The extracted threshold in eV/atom, or *default* if no pattern matched.
    """
    if not text or not text.strip():
        return default

    m = re.search(
        r"(?:stable\s+within|stability\s*(?:threshold)?[<\u2264:=\s]+|"
        r"threshold\s*[<\u2264:=\s]+)"
        r"\s*(\d+(?:\.\d+)?)\s*ev",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    m = re.search(r"(\d+(?:\.\d+)?)\s*ev/atom", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return default
