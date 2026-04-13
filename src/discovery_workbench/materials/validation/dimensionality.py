"""Crystal structure dimensionality analysis using Larsen's method.

Wraps pymatgen's ``get_dimensionality_larsen`` to classify structures as
0D (molecular), 1D (chain), 2D (layered), or 3D (framework).  Uses
``MinimumDistanceNN`` for bond detection — ``CrystalNN`` fails to detect
bonds in many ionic structures without oxidation states set, which would
produce false 0D results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

__all__ = ["DimensionalityResult", "check_dimensionality"]

from pymatgen.analysis.dimensionality import (
    get_dimensionality_larsen,
    get_structure_components,
)
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# Range of physically meaningful dimensionality values (0D–3D).
_MIN_DIMENSIONALITY = 0
_MAX_DIMENSIONALITY = 3

_ANALYSIS_METHOD = "larsen"


@dataclass(frozen=True, slots=True)
class DimensionalityResult:
    """Result of dimensionality analysis on a crystal structure.

    Args:
        dimensionality: Topological dimensionality (0=molecular, 1=chain,
            2=layered, 3=framework).
        is_3d: Whether the structure is a 3D framework.
        component_count: Number of connected components in the bonding graph.
        method: Name of the algorithm used for analysis.

    Raises:
        ValueError: If dimensionality is outside 0–3 or component_count < 0.
    """

    dimensionality: int
    is_3d: bool
    component_count: int
    method: str

    def __post_init__(self) -> None:
        if not (_MIN_DIMENSIONALITY <= self.dimensionality <= _MAX_DIMENSIONALITY):
            raise ValueError(
                f"dimensionality must be between {_MIN_DIMENSIONALITY} and "
                f"{_MAX_DIMENSIONALITY}, got {self.dimensionality}"
            )
        if self.component_count < 0:
            raise ValueError(
                f"component_count must be >= 0, got {self.component_count}"
            )


def check_dimensionality(structure: Structure) -> DimensionalityResult:
    """Analyse the topological dimensionality of a crystal structure.

    Builds a bonding graph with ``MinimumDistanceNN`` and applies
    Larsen's algorithm to determine whether the structure is 0D
    (molecular), 1D (chain), 2D (layered), or 3D (framework).

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`DimensionalityResult` with dimensionality, component
        count, and method metadata.

    Raises:
        TypeError: If *structure* is not a pymatgen Structure.
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"Expected pymatgen Structure, got {type(structure).__name__}"
        )

    logger.info(
        "Analysing dimensionality for %s (%d sites)",
        structure.composition.reduced_formula,
        len(structure),
    )

    bonded = StructureGraph.from_local_env_strategy(structure, MinimumDistanceNN())
    # pymatgen returns numpy int/bool — cast to native Python types so that
    # identity checks (``is True``) and JSON serialisation work correctly.
    dimensionality = int(get_dimensionality_larsen(bonded))
    components = get_structure_components(bonded)
    component_count = len(components)

    result = DimensionalityResult(
        dimensionality=dimensionality,
        is_3d=bool(dimensionality == _MAX_DIMENSIONALITY),
        component_count=component_count,
        method=_ANALYSIS_METHOD,
    )

    logger.info(
        "Dimensionality result: dim=%d, is_3d=%s, components=%d",
        result.dimensionality,
        result.is_3d,
        result.component_count,
    )

    return result
