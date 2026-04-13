"""Domain routing: keyword registry, token classification, and deterministic router."""

from agentic_discovery_core.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    INORGANIC_MATERIALS_KEYWORDS,
    SMALL_MOLECULE_KEYWORDS,
    STRUCTURED_CONSTRAINT_CUES,
    UNSUPPORTED_KEYWORDS,
    classify_token,
)
from agentic_discovery_core.routing.router import DeterministicRoutingResult, route_deterministic

__all__ = [
    "AMBIGUITY_KEYWORDS",
    "INORGANIC_MATERIALS_KEYWORDS",
    "DeterministicRoutingResult",
    "SMALL_MOLECULE_KEYWORDS",
    "STRUCTURED_CONSTRAINT_CUES",
    "UNSUPPORTED_KEYWORDS",
    "classify_token",
    "route_deterministic",
]
