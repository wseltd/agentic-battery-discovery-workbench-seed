"""Confidence-scored domain router.

Builds on the deterministic keyword registry (T005/T006) to produce a
numeric confidence score and a threshold-gated action.  The confidence
is keyword density: matched keywords / total unigrams in the input.
This is a heuristic, not a classifier — it measures how keyword-rich
the request is relative to its length.

Chose keyword density over raw count because a 3-keyword match in a
3-word query is much more decisive than 3 keywords in a 50-word query.
Alternative: TF-IDF or learned embeddings — rejected because this is a
deterministic gate and must run without model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from discovery_workbench.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    classify_token,
)
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

    Tokenises *text*, classifies tokens against the keyword registry,
    computes keyword density as the confidence score, and applies
    threshold-gated action rules.

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
    tokens = tokenise(text)

    # Separate unigrams from bigrams for denominator — bigrams are
    # generated from N unigrams, so there are N-1 bigrams.  We only
    # count unigrams in the denominator because bigram matches represent
    # the same underlying words.
    matched_keywords: set[str] = set()
    ambiguity_hits: set[str] = set()
    domains_seen: set[str] = set()

    for token in tokens:
        if token in AMBIGUITY_KEYWORDS:
            ambiguity_hits.add(token)
            continue
        domain = classify_token(token)
        if domain is not None:
            matched_keywords.add(token)
            domains_seen.add(domain)

    # Count unigrams: tokens before the bigram portion.  Bigrams start
    # at index N where N is the unigram count (N unigrams + N-1 bigrams
    # = 2N-1 total).  Solve: unigrams + (unigrams - 1) = len(tokens)
    # → unigrams = (len(tokens) + 1) / 2 when tokens > 0.
    if not tokens:
        unigram_count = 0
    else:
        unigram_count = (len(tokens) + 1) // 2

    # Confidence = keyword density (unique matched keywords / unigrams)
    if unigram_count == 0:
        confidence = 0.0
    else:
        raw = len(matched_keywords) / unigram_count
        confidence = min(raw, 1.0)  # clamp to [0, 1]

    # Determine domain
    routable = domains_seen - {"unsupported"}

    if len(routable) == 1:
        resolved_domain = next(iter(routable))
    elif len(routable) > 1:
        resolved_domain = None  # mixed signal
    else:
        resolved_domain = None

    # Determine action (priority order matches docstring)
    if unigram_count == 0:
        action: Literal["auto", "clarify", "unsupported"] = "unsupported"
    elif ambiguity_hits:
        action = "clarify"
    elif len(routable) > 1:
        action = "clarify"
    elif domains_seen == {"unsupported"}:
        action = "unsupported"
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
        matched_keywords=frozenset(matched_keywords),
        ambiguity_hits=frozenset(ambiguity_hits),
        stage="scored",
    )
