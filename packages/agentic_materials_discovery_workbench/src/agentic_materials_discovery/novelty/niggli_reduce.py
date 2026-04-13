"""Safe Niggli reduction with SpacegroupAnalyzer fallback.

Niggli reduction normalises lattice vectors to a canonical shortest
representation.  Degenerate cells (near-zero volume, coplanar vectors)
can cause the reduction to fail numerically.  In those cases we fall
back to SpacegroupAnalyzer's primitive standard structure, which is
more tolerant of pathological geometry.
"""

from __future__ import annotations

import logging

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)


def safe_niggli_reduce(structure: Structure) -> Structure:
    """Niggli-reduce a structure, falling back to primitive standard on failure.

    Attempts ``structure.lattice.get_niggli_reduced_lattice()`` and rebuilds
    the structure with the reduced lattice.  If Niggli reduction raises any
    exception (e.g. degenerate cell), falls back to
    ``SpacegroupAnalyzer(structure).get_primitive_standard_structure()``.

    Args:
        structure: Pymatgen Structure to reduce.

    Returns:
        Niggli-reduced structure, or primitive standard structure if Niggli
        reduction fails.
    """
    logger.info("Attempting Niggli reduction for %s", structure.formula)
    try:
        reduced_lattice = structure.lattice.get_niggli_reduced_lattice()
        reduced = Structure(
            reduced_lattice,
            structure.species,
            structure.cart_coords,
            coords_are_cartesian=True,
        )
        logger.info("Niggli reduction succeeded")
        return reduced
    except Exception as exc:
        logger.warning(
            "Niggli reduction failed (%s); falling back to "
            "SpacegroupAnalyzer primitive standard structure",
            exc,
        )
        return SpacegroupAnalyzer(structure).get_primitive_standard_structure()
