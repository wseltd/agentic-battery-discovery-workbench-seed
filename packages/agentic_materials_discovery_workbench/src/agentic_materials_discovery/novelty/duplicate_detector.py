"""Materials duplicate detection orchestrator.

Thin orchestrator that delegates to the coarse (pre-relax) and strict
(post-relax) detection passes.  Data structures (MaterialsDuplicateResult,
PassType) and default tolerances live here because both pass modules
import them -- keeping them in one place avoids circular dependencies
and duplicated definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum, auto

from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

# Default StructureMatcher tolerances -- pymatgen defaults.
DEFAULT_LTOL: float = 0.2
DEFAULT_STOL: float = 0.3
DEFAULT_ANGLE_TOL: float = 5.0


# --- Data structures ---------------------------------------------------------


class PassType(StrEnum):
    """Which duplicate-detection pass produced this result."""

    PRE_RELAX = auto()
    POST_RELAX = auto()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@dataclass(frozen=True, slots=True)
class MaterialsDuplicateResult:
    """Result of a duplicate check for one structure.

    Attributes:
        query_id: Identifier of the structure being checked.
        duplicate_of: Identifier of the earlier structure this is a duplicate
            of, or None if not a duplicate.
        pass_type: Which detection pass produced this result.
        matcher_tolerances: Tolerance parameters used (ltol, stol, angle_tol).
        is_duplicate: Whether the structure was classified as a duplicate.
    """

    query_id: str
    duplicate_of: str | None
    pass_type: PassType
    matcher_tolerances: dict
    is_duplicate: bool


# --- Detector ----------------------------------------------------------------


class MaterialsDuplicateDetector:
    """Duplicate detector for inorganic crystal structures.

    Two-pass approach: pre-relaxation uses cheap composition + cell parameter
    matching to filter obvious duplicates before expensive relaxation.
    Post-relaxation uses pymatgen StructureMatcher for rigorous
    crystallographic comparison of relaxed geometries.
    """

    def __init__(
        self,
        ltol: float = DEFAULT_LTOL,
        stol: float = DEFAULT_STOL,
        angle_tol: float = DEFAULT_ANGLE_TOL,
    ) -> None:
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol

    @property
    def matcher_tolerances(self) -> dict:
        """Current tolerance parameters as a dict."""
        return {
            "ltol": self._ltol,
            "stol": self._stol,
            "angle_tol": self._angle_tol,
        }

    def detect_duplicates_pre_relax(
        self,
        structures: list[tuple[str, Structure]],
    ) -> list[MaterialsDuplicateResult]:
        """Fast pre-relaxation duplicate screening via coarse pass.

        Args:
            structures: List of (id, Structure) pairs to check.

        Returns:
            One MaterialsDuplicateResult per input structure, in input order.
        """
        logger.info(
            "Pre-relax duplicate check on %d structures", len(structures)
        )
        from agentic_materials_discovery.novelty.coarse_pass import (
            coarse_deduplicate,
        )

        return coarse_deduplicate(structures)

    def detect_duplicates_post_relax(
        self,
        structures: list[tuple[str, Structure]],
    ) -> list[MaterialsDuplicateResult]:
        """Rigorous post-relaxation duplicate detection via strict pass.

        Args:
            structures: List of (id, Structure) pairs to check.

        Returns:
            One MaterialsDuplicateResult per input structure, in input order.
        """
        logger.info(
            "Post-relax duplicate check on %d structures", len(structures)
        )
        from agentic_materials_discovery.novelty.strict_pass import (
            strict_deduplicate,
        )

        return strict_deduplicate(structures)
