"""Coarse pre-relax duplicate grouping by composition and cell volume.

Groups structures by reduced formula, then sub-groups by reduced cell
volume (volume per atom), then runs a lenient StructureMatcher pairwise
within each volume sub-group.
"""

from __future__ import annotations

import logging

from pymatgen.core import Composition, Structure
from pymatgen.core.structure_matcher import StructureMatcher

from agentic_materials_discovery.novelty.duplicate_detector import (
    MaterialsDuplicateResult,
    PassType,
)

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

COARSE_LTOL: float = 0.3
COARSE_STOL: float = 0.5
COARSE_ANGLE_TOL: float = 8.0

DEFAULT_VOLUME_TOLERANCE: float = 0.15


# --- Helpers -----------------------------------------------------------------


def _volume_per_atom(structure: Structure) -> float:
    """Compute volume per atom for supercell-invariant volume comparison."""
    return structure.volume / len(structure)


def _volumes_close(v1: float, v2: float, tolerance: float) -> bool:
    """Check whether two volumes are within a fractional tolerance."""
    mean_vol = (v1 + v2) / 2.0
    if mean_vol <= 0:
        return False
    return abs(v1 - v2) / mean_vol <= tolerance


# --- Public API --------------------------------------------------------------


def coarse_deduplicate(
    structures: list[tuple[str, Structure]],
    volume_tolerance: float = DEFAULT_VOLUME_TOLERANCE,
) -> list[MaterialsDuplicateResult]:
    """Coarse pre-relax duplicate detection by composition and cell volume.

    Args:
        structures: List of (id, Structure) pairs to check.
        volume_tolerance: Maximum fractional deviation in volume per atom.

    Returns:
        One MaterialsDuplicateResult per input structure, in input order.

    Raises:
        ValueError: If volume_tolerance is negative.
    """
    if volume_tolerance < 0:
        raise ValueError(
            f"volume_tolerance must be non-negative, got {volume_tolerance}"
        )

    logger.info(
        "Coarse dedup on %d structures (volume_tolerance=%.3f)",
        len(structures),
        volume_tolerance,
    )

    if not structures:
        return []

    matcher = StructureMatcher(
        ltol=COARSE_LTOL,
        stol=COARSE_STOL,
        angle_tol=COARSE_ANGLE_TOL,
    )
    tolerances = {
        "ltol": COARSE_LTOL,
        "stol": COARSE_STOL,
        "angle_tol": COARSE_ANGLE_TOL,
    }

    results: list[MaterialsDuplicateResult] = []

    formula_groups: dict[
        str, list[list[tuple[str, Structure, float]]]
    ] = {}

    for struct_id, structure in structures:
        formula = Composition(structure.composition).reduced_formula
        vol = _volume_per_atom(structure)
        duplicate_of = _find_duplicate_in_groups(
            struct_id, structure, vol, volume_tolerance,
            formula_groups.get(formula, []), matcher,
        )

        if duplicate_of is None:
            sub_groups = formula_groups.setdefault(formula, [])
            _insert_into_volume_group(
                sub_groups, struct_id, structure, vol, volume_tolerance,
            )

        results.append(
            MaterialsDuplicateResult(
                query_id=struct_id,
                duplicate_of=duplicate_of,
                pass_type=PassType.PRE_RELAX,
                matcher_tolerances=tolerances,
                is_duplicate=duplicate_of is not None,
            )
        )

    duplicates_found = sum(1 for r in results if r.is_duplicate)
    logger.info(
        "Coarse dedup complete: %d duplicates in %d structures",
        duplicates_found,
        len(structures),
    )
    return results


def _find_duplicate_in_groups(
    struct_id: str,
    structure: Structure,
    vol: float,
    volume_tolerance: float,
    sub_groups: list[list[tuple[str, Structure, float]]],
    matcher: StructureMatcher,
) -> str | None:
    """Search existing volume sub-groups for a duplicate of structure."""
    for group in sub_groups:
        ref_vol = group[0][2]
        if not _volumes_close(vol, ref_vol, volume_tolerance):
            continue

        for earlier_id, earlier_struct, _ in group:
            if matcher.fit(structure, earlier_struct):
                logger.info(
                    "Structure %s is coarse duplicate of %s",
                    struct_id,
                    earlier_id,
                )
                return earlier_id

    return None


def _insert_into_volume_group(
    sub_groups: list[list[tuple[str, Structure, float]]],
    struct_id: str,
    structure: Structure,
    vol: float,
    volume_tolerance: float,
) -> None:
    """Insert a non-duplicate structure into the correct volume sub-group."""
    for group in sub_groups:
        ref_vol = group[0][2]
        if _volumes_close(vol, ref_vol, volume_tolerance):
            group.append((struct_id, structure, vol))
            return

    sub_groups.append([(struct_id, structure, vol)])
