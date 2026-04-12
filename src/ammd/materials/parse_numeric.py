"""Numeric constraint extraction from natural-language text.

Regex-based parsers for atom-count limits and stability thresholds
used in inorganic materials discovery requests.  No LLM calls — pure
pattern matching.

Supported patterns for max atoms:
- ``"<=20 atoms"``, ``"≤20 atoms"``
- ``"up to 15 atoms/cell"``
- ``"at most 30 atoms"``
- ``"max 20 atoms"``, ``"maximum 20 atoms"``

Supported patterns for stability threshold:
- ``"stable within 0.05 eV/atom"``
- ``"stability threshold 0.1 eV"``
- ``"stability < 0.05 eV"``
- ``"0.05 eV/atom"`` (standalone)
"""

from __future__ import annotations

import re


def parse_max_atoms(text: str, default: int = 20) -> int:
    """Extract maximum atom count from constraint text.

    Recognises ``<=N atoms``, ``up to N atoms``, ``at most N atoms``,
    and ``max N atoms`` patterns.  The ``/cell`` suffix is tolerated
    but ignored — units are always atoms per unit cell.

    Args:
        text: Free-form constraint text.
        default: Value returned when no atom-count pattern is found.

    Returns:
        The extracted atom count, or *default* if no pattern matched.
    """
    if not text or not text.strip():
        return default

    m = re.search(
        r"(?:[<≤]=?\s*|up\s+to\s+|at\s+most\s+|max(?:imum)?\s+)"
        r"(\d+)\s*atoms?(?:/cell)?",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return default


def parse_stability_threshold(text: str, default: float = 0.1) -> float:
    """Extract stability threshold in eV/atom from constraint text.

    Recognises ``stable within X eV/atom``, ``stability threshold X eV``,
    ``stability < X eV``, and standalone ``X eV/atom`` patterns.

    Args:
        text: Free-form constraint text.
        default: Value returned when no threshold pattern is found.

    Returns:
        The extracted threshold in eV/atom, or *default* if no pattern
        matched.
    """
    if not text or not text.strip():
        return default

    # "stable within 0.05 eV/atom", "stability threshold 0.1 eV",
    # "stability < 0.05 eV"
    m = re.search(
        r"(?:stable\s+within|stability\s*(?:threshold)?[<≤:=\s]+|"
        r"threshold\s*[<≤:=\s]+)"
        r"\s*(\d+(?:\.\d+)?)\s*ev",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    # Standalone "0.05 eV/atom"
    m = re.search(r"(\d+(?:\.\d+)?)\s*ev/atom", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return default
