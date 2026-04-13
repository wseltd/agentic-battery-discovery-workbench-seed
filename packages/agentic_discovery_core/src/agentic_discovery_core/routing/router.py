"""Deterministic domain router for user requests.

Tokenises free-text input into unigrams and bigrams, classifies each
against the keyword registry, and returns a frozen routing verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_discovery_core.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    classify_token,
)
from agentic_discovery_core.routing.tokeniser import tokenise


@dataclass(frozen=True)
class DeterministicRoutingResult:
    """Immutable result of the deterministic routing gate.

    Parameters
    ----------
    domain:
        ``"small_molecule"``, ``"inorganic_materials"``, ``"unsupported"``,
        or ``None`` when routing cannot be determined.
    matched_keywords:
        All registry keywords found in the input (lowercase-normalised).
    ambiguity_hits:
        Ambiguity keywords found in the input that blocked routing.
    stage:
        Always ``"deterministic"`` -- identifies which routing layer
        produced this result.
    """

    domain: str | None
    matched_keywords: frozenset[str]
    ambiguity_hits: frozenset[str]
    stage: Literal["deterministic"]

    def __post_init__(self) -> None:
        if self.stage != "deterministic":
            raise ValueError(
                f"stage must be 'deterministic', got {self.stage!r}"
            )


def route_deterministic(text: str) -> DeterministicRoutingResult:
    """Route a user request to a domain using keyword matching.

    Tokenises *text* into unigrams and bigrams, classifies each token
    against the keyword registry, and returns a :class:`DeterministicRoutingResult`.

    Routing rules (in priority order):

    1. If any ambiguity keyword is present, ``domain`` is ``None``
       regardless of other matches.
    2. If keywords from exactly one routable domain (``small_molecule``
       or ``inorganic_materials``) are found, route there.
    3. If keywords from multiple routable domains are found, ``domain``
       is ``None`` (mixed signal).
    4. If only ``unsupported`` keywords are found, ``domain`` is
       ``"unsupported"``.
    5. If no keywords match at all, ``domain`` is ``None``.

    Parameters
    ----------
    text:
        Free-text user request.

    Returns
    -------
    DeterministicRoutingResult
        Frozen verdict with matched keywords and domain assignment.
    """
    tokens = tokenise(text)

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

    # Rule 1: ambiguity keywords block routing unconditionally
    if ambiguity_hits:
        return DeterministicRoutingResult(
            domain=None,
            matched_keywords=frozenset(matched_keywords),
            ambiguity_hits=frozenset(ambiguity_hits),
            stage="deterministic",
        )

    # Separate routable domains from unsupported
    routable = domains_seen - {"unsupported"}

    if len(routable) == 1:
        # Rule 2: single routable domain
        resolved = next(iter(routable))
    elif len(routable) > 1:
        # Rule 3: mixed domains
        resolved = None
    elif "unsupported" in domains_seen:
        # Rule 4: only unsupported keywords
        resolved = "unsupported"
    else:
        # Rule 5: no keywords at all
        resolved = None

    return DeterministicRoutingResult(
        domain=resolved,
        matched_keywords=frozenset(matched_keywords),
        ambiguity_hits=frozenset(ambiguity_hits),
        stage="deterministic",
    )
