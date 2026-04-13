"""Structure matching via pymatgen StructureMatcher.

Isolates the expensive StructureMatcher comparison into a single focused
module.  The caller provides a candidate structure and a list of reference
(id, Structure) tuples; the function returns the first matching reference
ID or None.

Validates inputs at the boundary because StructureMatcher silently
misbehaves with negative tolerances rather than failing fast.  No base
class or registry — just one function with clear inputs and outputs.
"""

from __future__ import annotations

import logging

from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

# Pymatgen-default StructureMatcher tolerances, same as T038 duplicate
# detection and T039.a novelty checker.  The values are pymatgen's own
# defaults — keeping them identical across pipeline stages avoids
# silent divergence between dedup and novelty matching.
DEFAULT_LTOL: float = 0.2
DEFAULT_STOL: float = 0.3
DEFAULT_ANGLE_TOL: float = 5.0


def match_against_references(
    structure: Structure,
    references: list[tuple[str, Structure]],
    ltol: float = DEFAULT_LTOL,
    stol: float = DEFAULT_STOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
) -> str | None:
    """Compare a structure against reference crystals and return the first match.

    Uses pymatgen StructureMatcher to detect crystallographic equivalence.
    Returns as soon as the first match is found — order matters if
    multiple references could match.

    Args:
        structure: Post-relaxation pymatgen Structure to check.
        references: List of (reference_id, Structure) tuples to compare against.
        ltol: Fractional length tolerance for StructureMatcher.
        stol: Site tolerance for StructureMatcher.
        angle_tol: Angle tolerance in degrees for StructureMatcher.

    Returns:
        The reference ID of the first matching structure, or None if no
        reference matches.

    Raises:
        TypeError: If structure is not a pymatgen Structure.
        ValueError: If any tolerance is negative.
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"structure must be a pymatgen Structure, got {type(structure).__name__}"
        )
    if ltol < 0:
        raise ValueError(f"ltol must be non-negative, got {ltol}")
    if stol < 0:
        raise ValueError(f"stol must be non-negative, got {stol}")
    if angle_tol < 0:
        raise ValueError(f"angle_tol must be non-negative, got {angle_tol}")

    if not references:
        logger.info("No references to match against")
        return None

    logger.info(
        "Matching structure against %d references "
        "(ltol=%s, stol=%s, angle_tol=%s)",
        len(references),
        ltol,
        stol,
        angle_tol,
    )
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    for ref_id, ref_struct in references:
        if matcher.fit(structure, ref_struct):
            logger.info("Match found: reference_id=%s", ref_id)
            return ref_id

    logger.info("No match found among %d references", len(references))
    return None
