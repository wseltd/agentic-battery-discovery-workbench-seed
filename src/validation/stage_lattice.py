"""Stage 2 lattice validation: check_lattice pure function.

Delegates to :func:`agentic_discovery.materials.validation.validate_lattice_sanity`
for the actual lattice checks (NaN/Inf rejection, non-positive determinant,
near-singular volume).  This module provides a thin ``check_lattice`` entry
point that matches the stage-function naming convention used by the
validation pipeline.
"""

from __future__ import annotations

from agentic_discovery.materials.validation import (
    ValidationResult,
    validate_lattice_sanity,
)
from pymatgen.core import Structure


def check_lattice(structure: Structure) -> ValidationResult:
    """Check that *structure* has a numerically well-formed lattice.

    Rejects lattice matrices containing NaN or Inf, determinants that are
    non-positive (det <= 0), and near-singular lattices whose volume falls
    below the minimum threshold (0.1 Å³).

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`ValidationResult` with ``stage='lattice_sanity'`` and
        ``severity='hard'``.
    """
    return validate_lattice_sanity(structure)
