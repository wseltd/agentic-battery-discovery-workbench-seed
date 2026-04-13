"""Composite pre-relaxation structure validation for materials discovery.

Runs Q22 Phase A checks in sequence: lattice sanity -> allowed elements ->
atom count -> interatomic distances -> coordination sanity -> dimensionality.
Hard failures (lattice, elements, atom count) reject immediately.
Soft failures (distance, coordination, dimensionality) produce warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymatgen.core import Structure

from agentic_materials_discovery.validation.checks import (
    validate_allowed_elements,
    validate_atom_count,
    validate_coordination_sanity,
    validate_interatomic_distances,
    validate_lattice_sanity,
)
from agentic_materials_discovery.validation.dimensionality import (
    check_dimensionality,
)

logger = logging.getLogger(__name__)

# Warning type strings emitted by soft checks.  Tests assert on these exact
# strings, so treat them as part of the public contract.
WARNING_SHORT_DISTANCE = "short_interatomic_distance"
WARNING_EXTREME_COORDINATION = "extreme_coordination"
WARNING_LOW_DIMENSIONALITY = "low_dimensionality"


@dataclass(frozen=True, slots=True)
class StructureValidationResult:
    """Composite result of all pre-relaxation validation stages.

    Args:
        is_valid: True if every hard check passed.
        rejection_reason: Message from the first hard failure, or None.
        warnings: Warning type strings from soft checks that fired.
    """

    is_valid: bool
    rejection_reason: str | None
    warnings: tuple[str, ...]


def validate_structure(structure: Structure) -> StructureValidationResult:
    """Run all pre-relaxation validation stages on a crystal structure.

    Stages execute in order so that cheap checks (element membership, atom
    count) gate expensive ones (CrystalNN coordination, dimensionality).

    Hard stages -- any failure returns immediately with ``is_valid=False``:
      1. Lattice sanity (NaN/Inf, volume, determinant)
      2. Allowed elements (no noble gases, Tc, Pm, Z >= 84)
      3. Atom count (1-20 inclusive)

    Soft stages -- failures append to ``warnings``, structure stays valid:
      4. Interatomic distances (< 0.5x summed covalent radii)
      5. Coordination sanity (CN 0 or CN > 12 via CrystalNN)
      6. Dimensionality (non-3D topology via Larsen's method)

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`StructureValidationResult`.

    Raises:
        TypeError: If *structure* is not a pymatgen Structure.
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"Expected pymatgen Structure, got {type(structure).__name__}"
        )

    logger.info(
        "Validating structure %s (%d sites)",
        structure.composition.reduced_formula,
        len(structure),
    )

    # --- Hard checks (reject on first failure) ---

    for check in (validate_lattice_sanity, validate_allowed_elements, validate_atom_count):
        result = check(structure)
        if not result.passed:
            logger.info("Hard reject at stage=%s: %s", result.stage, result.message)
            return StructureValidationResult(
                is_valid=False,
                rejection_reason=result.message,
                warnings=(),
            )

    # --- Soft checks (collect warnings) ---

    warnings: list[str] = []

    distance_result = validate_interatomic_distances(structure)
    if not distance_result.passed:
        warnings.append(WARNING_SHORT_DISTANCE)

    coordination_result = validate_coordination_sanity(structure)
    if not coordination_result.passed:
        warnings.append(WARNING_EXTREME_COORDINATION)

    try:
        dim_result = check_dimensionality(structure)
        if not dim_result.is_3d:
            warnings.append(WARNING_LOW_DIMENSIONALITY)
    except ValueError:
        # Pathological geometry (e.g. single atom in huge cell) -- the bonding
        # graph cannot be built, which itself signals a non-3D structure.
        logger.warning(
            "Dimensionality check failed for %s -- treating as non-3D",
            structure.composition.reduced_formula,
        )
        warnings.append(WARNING_LOW_DIMENSIONALITY)

    logger.info(
        "Validation passed, warnings=%s",
        warnings if warnings else "none",
    )

    return StructureValidationResult(
        is_valid=True,
        rejection_reason=None,
        warnings=tuple(warnings),
    )
