"""Banned-word scanning for report text.

Scans free-text fields against the BANNED_WORDS_PATTERN regex and returns
deduplicated caveat strings for each matched term.  Uses word-boundary
anchors to avoid false positives on substrings (e.g. 'unproven' does not
match 'proven').
"""

from __future__ import annotations

from agentic_discovery_core.report_constants import BANNED_WORDS_PATTERN


def scan_banned_words(texts: list[str]) -> list[str]:
    """Scan texts for banned words and return caveat strings.

    Parameters
    ----------
    texts:
        List of strings to scan (e.g. user_brief, warning messages).

    Returns
    -------
    list[str]
        Deduplicated caveat strings, one per unique matched banned word.
        Order follows first-seen occurrence across all input texts.
        Empty list when no banned words are found.
    """
    seen: dict[str, str] = {}  # lowercase word -> caveat string
    for text in texts:
        for match in BANNED_WORDS_PATTERN.finditer(text):
            word_lower = match.group().lower()
            if word_lower not in seen:
                seen[word_lower] = (
                    f'Caveat: the term "{word_lower}" is not substantiated '
                    f"by the evidence level and should be interpreted with caution."
                )
    return list(seen.values())
