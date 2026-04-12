"""Domain routing: keyword registry and token classification."""

from discovery_workbench.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    INORGANIC_MATERIALS_KEYWORDS,
    SMALL_MOLECULE_KEYWORDS,
    STRUCTURED_CONSTRAINT_CUES,
    UNSUPPORTED_KEYWORDS,
    classify_token,
)

__all__ = [
    "AMBIGUITY_KEYWORDS",
    "INORGANIC_MATERIALS_KEYWORDS",
    "SMALL_MOLECULE_KEYWORDS",
    "STRUCTURED_CONSTRAINT_CUES",
    "UNSUPPORTED_KEYWORDS",
    "classify_token",
]
