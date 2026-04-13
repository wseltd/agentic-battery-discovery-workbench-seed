"""Space-group symmetry scoring for candidate materials.

Scores how well a candidate's verified space group matches a requested
symmetry target.  Supports exact space-group matching and crystal-system
level matching.

Imports crystal system constants from ammd.materials.symmetry rather than
duplicating them — that module is the canonical source for SG-to-system
mappings in this codebase.
"""

from __future__ import annotations

import logging

from ammd.materials.symmetry import (
    CRYSTAL_SYSTEM_SG_RANGES,
    sg_number_to_crystal_system,
)

logger = logging.getLogger(__name__)

# Minimum valid space group number (ITA convention).
_MIN_SG = 1

# Maximum valid space group number (ITA convention).
_MAX_SG = 230

# Partial credit when the crystal system matches but exact SG does not.
PARTIAL_SYSTEM_MATCH_SCORE = 0.5

# Known crystal system names, derived from the canonical mapping.
_VALID_CRYSTAL_SYSTEMS = frozenset(CRYSTAL_SYSTEM_SG_RANGES)


def _validate_sg(sg: int, label: str) -> None:
    """Raise ValueError if sg is outside 1-230.

    Args:
        sg: Space group number to validate.
        label: Parameter name for the error message.

    Raises:
        ValueError: If sg is outside 1-230.
    """
    if not (_MIN_SG <= sg <= _MAX_SG):
        raise ValueError(
            f"{label} must be between {_MIN_SG} and {_MAX_SG}, got {sg}"
        )


def symmetry_score(
    verified_sg: int,
    requested_sg: int | None = None,
    requested_crystal_system: str | None = None,
) -> float:
    """Score how well a verified space group matches the symmetry request.

    Returns 1.0 for exact SG match, PARTIAL_SYSTEM_MATCH_SCORE (0.5) for
    same crystal system, 0.0 for mismatch.  When neither target is given,
    returns 1.0 (no constraint means no penalty).

    When both requested_sg and requested_crystal_system are provided,
    requested_sg takes priority — if the exact SG matches the score is 1.0,
    otherwise the crystal system of verified_sg is compared against
    requested_crystal_system for partial credit.

    Args:
        verified_sg: The verified space group number (1-230).
        requested_sg: Desired space group number, or None if no preference.
        requested_crystal_system: Desired crystal system name (lowercase),
            or None if no preference.

    Returns:
        Symmetry score in {0.0, 0.5, 1.0}.

    Raises:
        ValueError: If verified_sg or requested_sg is outside 1-230.
        ValueError: If requested_crystal_system is not a recognised name.
    """
    _validate_sg(verified_sg, "verified_sg")

    if requested_sg is not None:
        _validate_sg(requested_sg, "requested_sg")

    if requested_crystal_system is not None:
        normalised = requested_crystal_system.strip().lower()
        if normalised not in _VALID_CRYSTAL_SYSTEMS:
            valid_names = ", ".join(sorted(_VALID_CRYSTAL_SYSTEMS))
            raise ValueError(
                f"Unknown crystal system {requested_crystal_system!r}. "
                f"Valid systems: {valid_names}"
            )

    # No constraint — no penalty.
    if requested_sg is None and requested_crystal_system is None:
        logger.info("No symmetry constraint, score=1.0")
        return 1.0

    # Exact SG match takes priority.
    if requested_sg is not None and verified_sg == requested_sg:
        logger.info(
            "Exact SG match verified_sg=%d requested_sg=%d score=1.0",
            verified_sg, requested_sg,
        )
        return 1.0

    # Determine the target crystal system — from requested_sg if given,
    # otherwise from requested_crystal_system directly.
    verified_system = sg_number_to_crystal_system(verified_sg)
    if requested_sg is not None:
        target_system = sg_number_to_crystal_system(requested_sg)
    else:
        # requested_crystal_system is not None at this point (guarded above).
        target_system = requested_crystal_system.strip().lower()  # type: ignore[union-attr]

    if verified_system == target_system:
        logger.info(
            "Crystal system match verified=%s target=%s score=%s",
            verified_system, target_system, PARTIAL_SYSTEM_MATCH_SCORE,
        )
        return PARTIAL_SYSTEM_MATCH_SCORE

    logger.info(
        "No symmetry match verified=%s target=%s score=0.0",
        verified_system, target_system,
    )
    return 0.0
