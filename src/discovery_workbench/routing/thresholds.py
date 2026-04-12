"""Threshold application for confidence-scored routing decisions.

Pure function: maps a confidence score and boolean flags to an action
string.  No side effects, no domain knowledge beyond the threshold
constants imported from scorer.py.

Chose to keep this as a separate pure function rather than inlining it
in route_with_confidence because it makes boundary testing trivial —
no need to craft token sequences to hit exact float values.
"""

from __future__ import annotations

from typing import Literal

from discovery_workbench.routing.scorer import (
    CONFIDENCE_AUTO_THRESHOLD,
    CONFIDENCE_CLARIFY_THRESHOLD,
)


def apply_thresholds(
    confidence: float,
    is_ambiguous: bool,
    is_unsupported: bool,
) -> Literal["auto", "clarify", "unsupported"]:
    """Map confidence score and flags to a routing action.

    Priority order (first match wins):

    1. ``is_unsupported`` → ``"unsupported"``
    2. ``is_ambiguous`` → ``"clarify"`` (never auto-route ambiguous input)
    3. ``confidence >= 0.80`` → ``"auto"``
    4. ``confidence >= 0.55`` → ``"clarify"``
    5. Otherwise → ``"unsupported"`` (too little signal)

    Parameters
    ----------
    confidence:
        Keyword density score, expected in [0.0, 1.0].
    is_ambiguous:
        True when ambiguity keywords are present in the input.
    is_unsupported:
        True when the input matches only unsupported-domain keywords.

    Returns
    -------
    Literal["auto", "clarify", "unsupported"]
        The routing action to take.
    """
    if is_unsupported:
        return "unsupported"
    if is_ambiguous:
        return "clarify"
    if confidence >= CONFIDENCE_AUTO_THRESHOLD:
        return "auto"
    if confidence >= CONFIDENCE_CLARIFY_THRESHOLD:
        return "clarify"
    return "unsupported"
