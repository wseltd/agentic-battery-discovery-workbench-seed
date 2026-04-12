"""Stoichiometry-pattern extraction from natural-language text.

Regex-based parser that recognises abstract stoichiometry patterns like
``ABO3``, ``AB2O4``, ``A2B`` in free-form materials science queries.
No LLM calls — pure pattern matching.

A token is considered a stoichiometry pattern (not a regular word) when it
contains at least one digit *or* 3+ distinct uppercase letters.  This
distinguishes ``ABO3`` and ``ABC`` from ordinary words like ``THE`` or ``NO``.
"""

from __future__ import annotations

import re


# Pattern: word-boundary, uppercase start, then 2+ uppercase-or-digit chars.
# Anchored to word boundaries to avoid matching inside longer tokens.
_STOICHIOMETRY_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,})\b")


def parse_stoichiometry_pattern(text: str) -> str | None:
    """Extract the first stoichiometry pattern from *text*.

    Scans for uppercase-letter-plus-optional-digit tokens that look like
    abstract stoichiometry formulas (``ABO3``, ``AB2O4``, ``A2B2O7``).

    A candidate is accepted when it satisfies at least one of:
    - contains at least one digit (e.g. ``ABO3``), or
    - contains 3 or more distinct uppercase letters (e.g. ``ABC``).

    This rejects common English words like ``THE``, ``AND``, ``NOT`` which
    have fewer than 3 distinct uppercase letters and no digits.

    Args:
        text: Free-form constraint text.

    Returns:
        The first matched stoichiometry pattern, or ``None`` if nothing
        matched.
    """
    if not text:
        return None

    for m in _STOICHIOMETRY_RE.finditer(text):
        candidate = m.group(1)
        has_digit = any(ch.isdigit() for ch in candidate)
        distinct_uppers = len(set(ch for ch in candidate if ch.isupper()))
        # Accept if it has a digit (clearly numeric stoichiometry) or
        # 3+ distinct uppercase letters (like ABC — unlikely English word).
        if has_digit or distinct_uppers >= 3:
            return candidate

    return None
