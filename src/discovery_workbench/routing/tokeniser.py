"""Tokeniser for the deterministic routing gate.

Splits free-text input into unigrams and bigrams for keyword matching.
Uses a single compiled regex to split on whitespace and common scientific
delimiters (hyphens, slashes, commas, semicolons, parentheses).

Design choice: regex split over successive str.replace calls — one pass
is clearer and avoids ordering bugs when delimiters interact.  Bigrams
are generated from the unigram sequence (not the raw text) so delimiter
boundaries produce correct adjacency.  No NLP dependencies.
"""

from __future__ import annotations

import re

# Single compiled pattern: split on one or more whitespace chars or any
# of the listed delimiters (optionally surrounded by whitespace).
# Kept as a module-level constant — compiled once at import time.
_SPLIT_RE = re.compile(r"[\s\-/,;()]+")


def tokenise(text: str) -> list[str]:
    """Tokenise free text into unigrams and bigrams for keyword matching.

    1. Lowercase the input.
    2. Split on whitespace and common delimiters using a single regex.
    3. Filter out empty tokens (from leading/trailing delimiters).
    4. Generate bigrams by joining consecutive unigram pairs with a space.
    5. Return unigrams followed by bigrams.

    Parameters
    ----------
    text:
        Raw user request string.

    Returns
    -------
    Combined list of unigrams then bigrams.  Empty input returns ``[]``.
    """
    lowered = text.lower()
    unigrams = [tok for tok in _SPLIT_RE.split(lowered) if tok]

    bigrams = [
        f"{unigrams[i]} {unigrams[i + 1]}"
        for i in range(len(unigrams) - 1)
    ]

    return unigrams + bigrams
