"""Confidence-scored domain router.

Builds on the deterministic keyword router (T006) and confidence
computation (T007.b) to produce a numeric confidence score and a
threshold-gated action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_discovery_core.routing.confidence import compute_confidence
from agentic_discovery_core.routing.router import route_deterministic
from agentic_discovery_core.routing.tokeniser import tokenise

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------
CONFIDENCE_AUTO_THRESHOLD: float = 0.80
CONFIDENCE_CLARIFY_THRESHOLD: float = 0.55


@dataclass(frozen=True)
class ScoredRoutingResult:
    """Immutable result of confidence-scored routing.

    Parameters
    ----------
    domain:
        ``"small_molecule"``, ``"inorganic_materials"``, or ``None``
        when the domain cannot be determined.
    confidence:
        Keyword density score in [0.0, 1.0].
    action:
        ``"auto"`` (proceed), ``"clarify"`` (ask user), or
        ``"unsupported"`` (reject).
    matched_keywords:
        Domain keywords found in the input.
    ambiguity_hits:
        Ambiguity keywords found in the input.
    stage:
        Always ``"scored"`` -- identifies this routing layer.
    """

    domain: str | None
    confidence: float
    action: Literal["auto", "clarify", "unsupported"]
    matched_keywords: frozenset[str]
    ambiguity_hits: frozenset[str]
    stage: Literal["scored"]

    def __post_init__(self) -> None:
        if self.stage != "scored":
            raise ValueError(f"stage must be 'scored', got {self.stage!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )


def route_with_confidence(text: str) -> ScoredRoutingResult:
    """Route a user request with a confidence score.

    Parameters
    ----------
    text:
        Free-text user request.

    Returns
    -------
    ScoredRoutingResult
        Frozen verdict with confidence score and threshold-gated action.
    """
    det = route_deterministic(text)

    tokens = tokenise(text)
    if not tokens:
        unigram_count = 0
    else:
        unigram_count = (len(tokens) + 1) // 2

    confidence = compute_confidence(len(det.matched_keywords), unigram_count)

    if det.domain is not None and det.domain != "unsupported":
        resolved_domain: str | None = det.domain
        is_ambiguous = False
        is_unsupported = False
    elif det.domain == "unsupported":
        resolved_domain = None
        is_ambiguous = False
        is_unsupported = True
    else:
        resolved_domain = None
        is_ambiguous = bool(det.ambiguity_hits) or bool(det.matched_keywords)
        is_unsupported = False

    # Threshold cascade
    if is_unsupported:
        action: Literal["auto", "clarify", "unsupported"] = "unsupported"
    elif is_ambiguous:
        action = "clarify"
    elif confidence >= CONFIDENCE_AUTO_THRESHOLD:
        action = "auto"
    elif confidence >= CONFIDENCE_CLARIFY_THRESHOLD:
        action = "clarify"
    else:
        action = "unsupported"

    return ScoredRoutingResult(
        domain=resolved_domain,
        confidence=confidence,
        action=action,
        matched_keywords=det.matched_keywords,
        ambiguity_hits=det.ambiguity_hits,
        stage="scored",
    )
