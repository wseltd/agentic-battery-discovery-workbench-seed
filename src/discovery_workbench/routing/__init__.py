"""Domain routing: keyword registry, token classification, and deterministic router."""

from discovery_workbench.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    INORGANIC_MATERIALS_KEYWORDS,
    SMALL_MOLECULE_KEYWORDS,
    STRUCTURED_CONSTRAINT_CUES,
    UNSUPPORTED_KEYWORDS,
    classify_token,
)
from discovery_workbench.routing.router import RoutingResult, route_deterministic

__all__ = [
    "AMBIGUITY_KEYWORDS",
    "INORGANIC_MATERIALS_KEYWORDS",
    "RoutingResult",
    "SMALL_MOLECULE_KEYWORDS",
    "STRUCTURED_CONSTRAINT_CUES",
    "UNSUPPORTED_KEYWORDS",
    "classify_token",
    "route_deterministic",
]
