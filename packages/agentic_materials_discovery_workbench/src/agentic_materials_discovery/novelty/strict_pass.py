"""Strict post-relax duplicate detection using Niggli reduction and StructureMatcher.

Applies safe_niggli_reduce to normalise each structure's lattice to a
canonical shortest representation, then runs pairwise StructureMatcher
with pymatgen-default tolerances (ltol=0.2, stol=0.3, angle_tol=5).
"""

from __future__ import annotations

import logging

from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher

from agentic_materials_discovery.novelty.duplicate_detector import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    MaterialsDuplicateResult,
    PassType,
)
from agentic_materials_discovery.novelty.niggli_reduce import safe_niggli_reduce

logger = logging.getLogger(__name__)


def strict_deduplicate(
    structures: list[tuple[str, Structure]],
) -> list[MaterialsDuplicateResult]:
    """Strict post-relax duplicate detection via Niggli reduction and StructureMatcher.

    Args:
        structures: List of (id, Structure) pairs to check.

    Returns:
        One MaterialsDuplicateResult per input structure, in input order.
    """
    logger.info("Strict post-relax dedup on %d structures", len(structures))

    if not structures:
        return []

    matcher = StructureMatcher(
        ltol=DEFAULT_LTOL,
        stol=DEFAULT_STOL,
        angle_tol=DEFAULT_ANGLE_TOL,
    )
    tolerances = {
        "ltol": DEFAULT_LTOL,
        "stol": DEFAULT_STOL,
        "angle_tol": DEFAULT_ANGLE_TOL,
    }

    reduced: list[tuple[str, Structure]] = [
        (sid, safe_niggli_reduce(s)) for sid, s in structures
    ]

    results: list[MaterialsDuplicateResult] = []
    canonical: list[tuple[str, Structure]] = []

    for struct_id, structure in reduced:
        duplicate_of = _find_first_match(struct_id, structure, canonical, matcher)

        if duplicate_of is None:
            canonical.append((struct_id, structure))

        results.append(
            MaterialsDuplicateResult(
                query_id=struct_id,
                duplicate_of=duplicate_of,
                pass_type=PassType.POST_RELAX,
                matcher_tolerances=tolerances,
                is_duplicate=duplicate_of is not None,
            )
        )

    duplicates_found = sum(1 for r in results if r.is_duplicate)
    logger.info(
        "Strict dedup complete: %d duplicates in %d structures",
        duplicates_found,
        len(structures),
    )
    return results


def _find_first_match(
    struct_id: str,
    structure: Structure,
    canonical: list[tuple[str, Structure]],
    matcher: StructureMatcher,
) -> str | None:
    """Find the first canonical structure that matches via StructureMatcher."""
    for earlier_id, earlier_struct in canonical:
        if matcher.fit(structure, earlier_struct):
            logger.info(
                "Structure %s is strict duplicate of %s",
                struct_id,
                earlier_id,
            )
            return earlier_id
    return None
