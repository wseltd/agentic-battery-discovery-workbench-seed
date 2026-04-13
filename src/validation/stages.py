"""Stage 3 element and atom-count validation: check_allowed_elements, check_atom_count.

Delegates to :func:`agentic_discovery.materials.validation.validate_allowed_elements`
and :func:`agentic_discovery.materials.validation.validate_atom_count` for the
actual checks.  This module provides thin entry points that match the
stage-function naming convention used by the validation pipeline.
"""

from __future__ import annotations

from agentic_discovery.materials.validation import (
    MaterialsValidationResult,
    validate_allowed_elements,
    validate_atom_count,
)
from pymatgen.core import Structure


def check_allowed_elements(structure: Structure) -> MaterialsValidationResult:
    """Check that *structure* contains no forbidden elements.

    Rejects structures containing noble gases, Tc (Z=43), Pm (Z=61),
    and all elements with Z >= 84.  These are impractical targets for
    inorganic-materials generation.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='allowed_elements'`` and
        ``severity='hard'``.
    """
    return validate_allowed_elements(structure)


def check_atom_count(structure: Structure) -> MaterialsValidationResult:
    """Check that *structure* has between 1 and 20 atoms (inclusive).

    Empty structures are physically meaningless; very large cells are
    outside the scope of the fast-validation pipeline.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='atom_count'`` and
        ``severity='hard'``.
    """
    return validate_atom_count(structure)
