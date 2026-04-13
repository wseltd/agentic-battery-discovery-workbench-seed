"""Coarse pre-relax duplicate grouping by composition and cell volume.

Groups structures by reduced formula, then sub-groups by reduced cell
volume (volume per atom), then runs a lenient StructureMatcher pairwise
within each volume sub-group.  This catches obvious copies before
expensive relaxation without the cost of full crystallographic comparison
on the entire batch.

Chose volume-per-atom rather than raw cell volume so that supercells
of the same crystal land in the same volume bucket.  The StructureMatcher
tolerances (ltol=0.3, stol=0.5, angle_tol=8) are deliberately looser
than pymatgen defaults to favour recall over precision in this pre-relax
screening step.
"""

from __future__ import annotations

import logging

from pymatgen.core import Composition, Structure
from pymatgen.core.structure_matcher import StructureMatcher

from agentic_discovery_workbench.materials.duplicate_detector import (
    DuplicateResult,
    PassType,
)

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

# Coarse-pass StructureMatcher tolerances — deliberately looser than pymatgen
# defaults (0.2/0.3/5) to favour recall in the cheap screening pass.
COARSE_LTOL: float = 0.3
COARSE_STOL: float = 0.5
COARSE_ANGLE_TOL: float = 8.0

# Default fractional tolerance for volume-per-atom comparison
DEFAULT_VOLUME_TOLERANCE: float = 0.15


# --- Helpers -----------------------------------------------------------------


def _volume_per_atom(structure: Structure) -> float:
    """Compute volume per atom for supercell-invariant volume comparison.

    Args:
        structure: Pymatgen Structure.

    Returns:
        Cell volume divided by number of sites.
    """
    return structure.volume / len(structure)


def _volumes_close(v1: float, v2: float, tolerance: float) -> bool:
    """Check whether two volumes are within a fractional tolerance.

    Args:
        v1: First volume per atom.
        v2: Second volume per atom.
        tolerance: Maximum allowed fractional deviation from the mean.

    Returns:
        True if |v1 - v2| / mean(v1, v2) <= tolerance.
    """
    mean_vol = (v1 + v2) / 2.0
    if mean_vol <= 0:
        return False
    return abs(v1 - v2) / mean_vol <= tolerance


# --- Public API --------------------------------------------------------------


def coarse_deduplicate(
    structures: list[tuple[str, Structure]],
    volume_tolerance: float = DEFAULT_VOLUME_TOLERANCE,
) -> list[DuplicateResult]:
    """Coarse pre-relax duplicate detection by composition and cell volume.

    Three-stage filter applied in order:
    1. Group by ``Composition(s.composition).reduced_formula``.
    2. Within each composition group, sub-group by volume per atom
       within ±volume_tolerance fraction.
    3. Within each volume sub-group, run pairwise
       ``StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8).fit(s1, s2)``,
       marking later structures as duplicates of the first match.

    Args:
        structures: List of (id, Structure) pairs to check.
        volume_tolerance: Maximum fractional deviation in volume per atom
            for two structures to be considered in the same volume group.
            Must be non-negative.

    Returns:
        One DuplicateResult per input structure, in input order, with
        ``pass_type=PassType.PRE_RELAX``.

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

    # Track per-structure results keyed by original index to preserve order
    results: list[DuplicateResult] = []

    # Stage 1: group by reduced formula
    # Each formula maps to a list of volume sub-groups.
    # A sub-group is a list of (id, structure, volume_per_atom) tuples.
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
            # Add to appropriate volume sub-group, or create a new one
            sub_groups = formula_groups.setdefault(formula, [])
            _insert_into_volume_group(
                sub_groups, struct_id, structure, vol, volume_tolerance,
            )

        results.append(
            DuplicateResult(
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
    """Search existing volume sub-groups for a duplicate of structure.

    Args:
        struct_id: Identifier of the query structure.
        structure: The query structure.
        vol: Volume per atom of the query structure.
        volume_tolerance: Fractional volume tolerance.
        sub_groups: Existing volume sub-groups for this composition.
        matcher: StructureMatcher to use for pairwise comparison.

    Returns:
        ID of the first matching earlier structure, or None.
    """
    for group in sub_groups:
        # Check volume against the first member of this sub-group
        ref_vol = group[0][2]
        if not _volumes_close(vol, ref_vol, volume_tolerance):
            continue

        # Volume matches — run StructureMatcher pairwise
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
    """Insert a non-duplicate structure into the correct volume sub-group.

    If volume matches an existing sub-group, appends there. Otherwise
    creates a new sub-group.

    Args:
        sub_groups: Mutable list of volume sub-groups.
        struct_id: Structure identifier.
        structure: The structure.
        vol: Volume per atom.
        volume_tolerance: Fractional volume tolerance.
    """
    for group in sub_groups:
        ref_vol = group[0][2]
        if _volumes_close(vol, ref_vol, volume_tolerance):
            group.append((struct_id, structure, vol))
            return

    sub_groups.append([(struct_id, structure, vol)])
