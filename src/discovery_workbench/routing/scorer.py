"""Confidence-scored domain router.

Builds on the deterministic keyword router (T006) and confidence
computation (T007.b) to produce a numeric confidence score and a
threshold-gated action.  The confidence is keyword density: matched
keywords / total unigrams in the input.  This is a heuristic, not a
classifier — it measures how keyword-rich the request is relative to
its length.

Chose keyword density over raw count because a 3-keyword match in a
3-word query is much more decisive than 3 keywords in a 50-word query.
Alternative: TF-IDF or learned embeddings — rejected because this is a
deterministic gate and must run without model inference.

Threshold logic is inlined rather than calling apply_thresholds because
thresholds.py imports constants from this module — calling back would
create a circular import.  The six-line cascade is identical to
apply_thresholds and tested at every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from discovery_workbench.routing.confidence import compute_confidence
from discovery_workbench.routing.router import route_deterministic
from discovery_workbench.routing.tokeniser import tokenise

# ---------------------------------------------------------------------------
# Threshold constants — chosen conservatively for a deterministic gate.
# AUTO requires strong signal; CLARIFY catches moderate-but-ambiguous input.
# Values are boundary-inclusive: >= threshold triggers the action.
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
        Always ``"scored"`` — identifies this routing layer.
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

    Delegates domain classification to :func:`route_deterministic` (T006)
    and keyword density to :func:`compute_confidence` (T007.b), then
    applies threshold-gated action rules.

    Action rules (in priority order):

    1. Empty input or no unigrams → confidence 0.0, action ``"unsupported"``.
    2. Ambiguity keywords present → action ``"clarify"`` regardless of
       confidence (conservative: never auto-route ambiguous requests).
    3. Keywords from multiple routable domains → action ``"clarify"``
       (mixed signal needs user disambiguation).
    4. Only unsupported-domain keywords → action ``"unsupported"``.
    5. Confidence >= AUTO_THRESHOLD → action ``"auto"``.
    6. Confidence >= CLARIFY_THRESHOLD → action ``"clarify"``.
    7. Otherwise → action ``"unsupported"`` (too little signal).

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

    # Tokenise to get the unigram count for confidence denominator.
    # Bigrams are derived from unigrams (N unigrams → N-1 bigrams),
    # so unigram_count = (len(tokens) + 1) // 2.
    tokens = tokenise(text)
    if not tokens:
        unigram_count = 0
    else:
        unigram_count = (len(tokens) + 1) // 2

    confidence = compute_confidence(len(det.matched_keywords), unigram_count)

    # Map deterministic domain to scored domain and action flags.
    # "unsupported" is not a routable domain — surface it as None
    # with an unsupported action.
    if det.domain is not None and det.domain != "unsupported":
        # Clear single-domain signal from deterministic gate
        resolved_domain: str | None = det.domain
        is_ambiguous = False
        is_unsupported = False
    elif det.domain == "unsupported":
        resolved_domain = None
        is_ambiguous = False
        is_unsupported = True
    else:
        # domain is None: ambiguity keywords, mixed domains, or no signal.
        # Ambiguity/mixed → force clarify; no signal → let thresholds decide.
        resolved_domain = None
        is_ambiguous = bool(det.ambiguity_hits) or bool(det.matched_keywords)
        is_unsupported = False

    # Threshold cascade — identical to apply_thresholds (thresholds.py),
    # inlined to avoid circular import.
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
