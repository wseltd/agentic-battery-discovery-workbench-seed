"""Crystal-system / space-group symmetry utilities.

Provides the canonical mapping between the 7 crystal systems and their
space-group number ranges (ITA convention, 1-230), plus lightweight
helpers for P1 policy enforcement.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# Immutable mapping: lowercase crystal system -> (min_sg, max_sg) inclusive.
CRYSTAL_SYSTEM_SG_RANGES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "triclinic": (1, 2),
        "monoclinic": (3, 15),
        "orthorhombic": (16, 74),
        "tetragonal": (75, 142),
        "trigonal": (143, 167),
        "hexagonal": (168, 194),
        "cubic": (195, 230),
    }
)

# Pre-built reverse lookup: sg_number -> crystal system name.
_SG_TO_SYSTEM: dict[int, str] = {}
for _system, (_lo, _hi) in CRYSTAL_SYSTEM_SG_RANGES.items():
    for _sg in range(_lo, _hi + 1):
        _SG_TO_SYSTEM[_sg] = _system


def crystal_system_to_sg_range(system_name: str) -> tuple[int, int]:
    """Return the (min_sg, max_sg) range for a crystal system.

    Parameters
    ----------
    system_name:
        Crystal system name (case-insensitive).

    Returns
    -------
    tuple[int, int]
        Inclusive (min, max) space-group numbers.

    Raises
    ------
    ValueError
        If *system_name* is not one of the 7 crystal systems.
    """
    key = system_name.strip().lower()
    try:
        return CRYSTAL_SYSTEM_SG_RANGES[key]
    except KeyError:
        valid = ", ".join(sorted(CRYSTAL_SYSTEM_SG_RANGES))
        raise ValueError(
            f"Unknown crystal system {system_name!r}. "
            f"Valid systems: {valid}"
        ) from None


def sg_number_to_crystal_system(sg: int) -> str:
    """Return the crystal system name for a space-group number.

    Parameters
    ----------
    sg:
        Space-group number (1-230).

    Returns
    -------
    str
        Lowercase crystal system name.

    Raises
    ------
    ValueError
        If *sg* is outside 1-230.
    """
    try:
        return _SG_TO_SYSTEM[sg]
    except KeyError:
        raise ValueError(
            f"Space-group number {sg} is outside the valid range 1-230."
        ) from None


def is_p1(sg: int) -> bool:
    """Return True if *sg* is the P1 space group (number 1)."""
    return sg == 1


def enforce_p1_policy(
    sg: int, *, allow_p1: bool
) -> tuple[bool, str]:
    """Check whether a structure's space group passes the P1 policy.

    Parameters
    ----------
    sg:
        Space-group number.
    allow_p1:
        Whether P1 structures are permitted.

    Returns
    -------
    tuple[bool, str]
        (ok, reason) -- *ok* is False when the structure is blocked.
    """
    if is_p1(sg) and not allow_p1:
        return False, "P1 structures require allow_P1=true; auto-down-ranked per Q20 policy"
    return True, ""
