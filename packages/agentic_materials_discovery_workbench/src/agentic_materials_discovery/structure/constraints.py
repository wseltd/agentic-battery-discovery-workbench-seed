"""Materials constraint parsing for inorganic materials discovery requests.

Parses natural-language constraint text into a structured MaterialsConstraints
dataclass.
"""

from __future__ import annotations

import dataclasses
import logging

from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup

from agentic_materials_discovery.structure.parse_elements import parse_element_list
from agentic_materials_discovery.structure.parse_numeric import parse_max_atoms, parse_stability_threshold
from agentic_materials_discovery.structure.parse_space_group import (
    CRYSTAL_SYSTEM_RANGES,
    parse_space_group,
)
from agentic_materials_discovery.structure.parse_stoichiometry import parse_stoichiometry_pattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

_EXCLUDED_SYMBOLS = frozenset({"Tc", "Pm", "He", "Ne", "Ar", "Kr", "Xe"})

ALLOWED_ELEMENTS: frozenset[str] = frozenset(
    el.symbol
    for el in Element
    if el.Z <= 83 and el.symbol not in _EXCLUDED_SYMBOLS
)

# Hermann-Mauguin symbol -> space-group number (1-230).
SG_SYMBOL_TO_NUMBER: dict[str, int] = {}
for _sg_num in range(1, 231):
    try:
        _sg = SpaceGroup.from_int_number(_sg_num)
        SG_SYMBOL_TO_NUMBER[_sg.symbol] = _sg_num
    except ValueError:
        logger.warning("pymatgen has no SpaceGroup for number %d", _sg_num)


def crystal_system_to_sg_range(system: str) -> tuple[int, int] | None:
    """Map a crystal system name to its inclusive space-group number range.

    Args:
        system: Lowercase crystal system name (e.g. ``"cubic"``).

    Returns:
        ``(low, high)`` inclusive SG range, or ``None`` if *system* is
        not a recognised crystal system name.
    """
    return CRYSTAL_SYSTEM_RANGES.get(system)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MaterialsConstraints:
    """Structured constraints for an inorganic-materials generation request.

    All fields default to their unconstrained state (None or a permissive
    value) so that only explicitly parsed constraints restrict generation.
    """

    allowed_elements: list[str] | None = None
    excluded_elements: list[str] | None = None
    stoichiometry_pattern: str | None = None
    space_group_number: int | None = None
    space_group_range: tuple[int, int] | None = None
    crystal_system: str | None = None
    max_atoms: int = 20
    stability_threshold_ev: float = 0.1
    chemistry_scope: str | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_MAX_ATOMS_DEFAULT = 20
_STABILITY_THRESHOLD_DEFAULT = 0.1


def parse_materials_constraints(text: str) -> MaterialsConstraints:
    """Parse natural-language constraint text into a MaterialsConstraints.

    Args:
        text: Free-form constraint text.

    Returns:
        A ``MaterialsConstraints`` populated with every constraint the
        parser could extract.
    """
    constraints = MaterialsConstraints()

    if not text or not text.strip():
        return constraints

    constraints.max_atoms = parse_max_atoms(text, default=_MAX_ATOMS_DEFAULT)
    constraints.stability_threshold_ev = parse_stability_threshold(
        text, default=_STABILITY_THRESHOLD_DEFAULT,
    )

    sg_number, crystal_system = parse_space_group(text, SG_SYMBOL_TO_NUMBER)
    constraints.space_group_number = sg_number
    constraints.crystal_system = crystal_system

    if crystal_system is not None and sg_number is None:
        constraints.space_group_range = crystal_system_to_sg_range(crystal_system)

    constraints.stoichiometry_pattern = parse_stoichiometry_pattern(text)

    allowed, excluded, scope = parse_element_list(text, ALLOWED_ELEMENTS)
    constraints.allowed_elements = allowed
    constraints.excluded_elements = excluded
    constraints.chemistry_scope = scope

    return constraints
