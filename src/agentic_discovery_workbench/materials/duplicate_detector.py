"""Materials duplicate detection using composition matching and structure comparison.

Pre-relaxation pass uses composition formula and cell parameter similarity
for fast screening. Post-relaxation pass uses pymatgen StructureMatcher
with Niggli-reduced cells for rigorous crystallographic comparison.

Chose two-pass design because pre-relax structures may have distorted cells
that confuse StructureMatcher, while post-relax structures have cleaner
geometry suitable for strict matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum, auto

from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

# Default StructureMatcher tolerances — pymatgen defaults are ltol=0.2,
# stol=0.3, angle_tol=5. We use the same to avoid surprising users who
# expect pymatgen-standard behaviour.
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
class DuplicateResult:
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


# --- Helpers -----------------------------------------------------------------


def _niggli_reduce_safe(structure: Structure) -> Structure:
    """Niggli-reduce a structure, falling back to the original on failure.

    Degenerate cells (near-zero volume, coplanar vectors) can cause Niggli
    reduction to fail numerically. We fall back to the unreduced structure
    rather than crashing the entire batch.

    Args:
        structure: Pymatgen Structure to reduce.

    Returns:
        Niggli-reduced structure, or the original if reduction fails.
    """
    try:
        return structure.get_reduced_structure(reduction_algo="niggli")
    except Exception:
        logger.warning(
            "Niggli reduction failed for structure; using unreduced cell"
        )
        return structure


def _cell_params_close(
    s1: Structure,
    s2: Structure,
    ltol: float,
    angle_tol: float,
) -> bool:
    """Compare lattice parameters of two structures within tolerance.

    Lengths are sorted before comparison to handle axis permutations.
    Length tolerance is fractional (relative to mean). Angle tolerance is
    absolute in degrees.

    Args:
        s1: First structure.
        s2: Second structure.
        ltol: Fractional length tolerance.
        angle_tol: Absolute angle tolerance in degrees.

    Returns:
        True if cell parameters match within tolerance.
    """
    p1 = s1.lattice.parameters  # (a, b, c, alpha, beta, gamma)
    p2 = s2.lattice.parameters

    lengths1 = sorted(p1[:3])
    lengths2 = sorted(p2[:3])
    for l1, l2 in zip(lengths1, lengths2):
        mean_len = (l1 + l2) / 2.0
        if mean_len > 0 and abs(l1 - l2) / mean_len > ltol:
            return False

    angles1 = sorted(p1[3:])
    angles2 = sorted(p2[3:])
    for a1, a2 in zip(angles1, angles2):
        if abs(a1 - a2) > angle_tol:
            return False

    return True


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
        """Initialise detector with matching tolerances.

        Args:
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance in degrees.
        """
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
    ) -> list[DuplicateResult]:
        """Fast pre-relaxation duplicate screening by composition and cell shape.

        Groups structures by reduced formula, then within each group compares
        sorted lattice parameters. This is a heuristic — it catches obvious
        copies but cannot detect symmetry-equivalent structures with different
        cell conventions.

        Args:
            structures: List of (id, Structure) pairs to check.

        Returns:
            One DuplicateResult per input structure, in input order.
        """
        logger.info(
            "Pre-relax duplicate check on %d structures", len(structures)
        )
        if not structures:
            return []

        results: list[DuplicateResult] = []
        seen: dict[str, list[tuple[str, Structure]]] = {}

        for struct_id, structure in structures:
            formula = structure.composition.reduced_formula
            duplicate_of = None

            if formula in seen:
                for earlier_id, earlier_struct in seen[formula]:
                    if _cell_params_close(
                        structure,
                        earlier_struct,
                        self._ltol,
                        self._angle_tol,
                    ):
                        duplicate_of = earlier_id
                        break

            if duplicate_of is None:
                seen.setdefault(formula, []).append((struct_id, structure))

            results.append(
                DuplicateResult(
                    query_id=struct_id,
                    duplicate_of=duplicate_of,
                    pass_type=PassType.PRE_RELAX,
                    matcher_tolerances=self.matcher_tolerances,
                    is_duplicate=duplicate_of is not None,
                )
            )

        return results

    def detect_duplicates_post_relax(
        self,
        structures: list[tuple[str, Structure]],
    ) -> list[DuplicateResult]:
        """Rigorous post-relaxation duplicate detection using StructureMatcher.

        Applies Niggli reduction before comparison so that different cell
        choices for the same crystal are correctly identified as duplicates.
        Falls back to unreduced cells if Niggli reduction fails (e.g.,
        degenerate zero-volume cells).

        Args:
            structures: List of (id, Structure) pairs to check.

        Returns:
            One DuplicateResult per input structure, in input order.
        """
        logger.info(
            "Post-relax duplicate check on %d structures", len(structures)
        )
        if not structures:
            return []

        matcher = StructureMatcher(
            ltol=self._ltol,
            stol=self._stol,
            angle_tol=self._angle_tol,
        )

        reduced: list[tuple[str, Structure]] = [
            (sid, _niggli_reduce_safe(s)) for sid, s in structures
        ]

        results: list[DuplicateResult] = []
        canonical: list[tuple[str, Structure]] = []

        for struct_id, structure in reduced:
            duplicate_of = None

            for earlier_id, earlier_struct in canonical:
                if matcher.fit(structure, earlier_struct):
                    duplicate_of = earlier_id
                    break

            if duplicate_of is None:
                canonical.append((struct_id, structure))

            results.append(
                DuplicateResult(
                    query_id=struct_id,
                    duplicate_of=duplicate_of,
                    pass_type=PassType.POST_RELAX,
                    matcher_tolerances=self.matcher_tolerances,
                    is_duplicate=duplicate_of is not None,
                )
            )

        return results
