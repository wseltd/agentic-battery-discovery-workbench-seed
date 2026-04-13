"""Stoichiometry-pattern extraction from natural-language text."""

from __future__ import annotations

import re

_STOICHIOMETRY_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,})\b")


def parse_stoichiometry_pattern(text: str) -> str | None:
    """Extract the first stoichiometry pattern from *text*.

    Args:
        text: Free-form constraint text.

    Returns:
        The first matched stoichiometry pattern, or ``None``.
    """
    if not text:
        return None

    for m in _STOICHIOMETRY_RE.finditer(text):
        candidate = m.group(1)
        has_digit = any(ch.isdigit() for ch in candidate)
        distinct_uppers = len(set(ch for ch in candidate if ch.isupper()))
        if has_digit or distinct_uppers >= 3:
            return candidate

    return None
