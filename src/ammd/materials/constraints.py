"""Materials constraint parsing for inorganic materials discovery requests.

Parses natural-language constraint text into a structured MaterialsConstraints
dataclass.  Delegates to the single-concern sub-parsers in this package
(parse_elements, parse_stoichiometry, parse_space_group, parse_numeric) and
assembles their results.  Parsing is regex-based and heuristic — it handles
common phrasings but is not a full NLP pipeline.
"""

from __future__ import annotations

import dataclasses
import logging

from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup

from ammd.materials.parse_elements import parse_element_list
from ammd.materials.parse_numeric import parse_max_atoms, parse_stability_threshold
from ammd.materials.parse_space_group import (
    CRYSTAL_SYSTEM_RANGES,
    parse_space_group,
)
from ammd.materials.parse_stoichiometry import parse_stoichiometry_pattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants — defined once, imported everywhere (standard §12)
# ---------------------------------------------------------------------------

# Elements with Z <= 83, excluding Tc (43), Pm (61), and noble gases
# (He, Ne, Ar, Kr, Xe).  Noble gases are excluded because they almost
# never form stable extended solids, and Tc/Pm are radioactive with no
# stable isotopes — impractical for materials discovery.
_EXCLUDED_SYMBOLS = frozenset({"Tc", "Pm", "He", "Ne", "Ar", "Kr", "Xe"})

ALLOWED_ELEMENTS: frozenset[str] = frozenset(
    el.symbol
    for el in Element
    if el.Z <= 83 and el.symbol not in _EXCLUDED_SYMBOLS
)

# Hermann-Mauguin symbol → space-group number (1–230).
# Built from pymatgen at import time so we track upstream corrections
# automatically rather than maintaining a 230-row dict by hand.
SG_SYMBOL_TO_NUMBER: dict[str, int] = {}
for _sg_num in range(1, 231):
    try:
        _sg = SpaceGroup.from_int_number(_sg_num)
        SG_SYMBOL_TO_NUMBER[_sg.symbol] = _sg_num
    except ValueError:
        logger.warning("pymatgen has no SpaceGroup for number %d", _sg_num)


def crystal_system_to_sg_range(system: str) -> tuple[int, int] | None:
    """Map a crystal system name to its inclusive space-group number range.

    Thin wrapper around CRYSTAL_SYSTEM_RANGES from parse_space_group —
    exists as a named function so downstream callers (T028+) have a
    stable callable interface rather than reaching into the dict directly.

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

    Args:
        allowed_elements: Element symbols that the generated structure may
            contain.  ``None`` means no restriction.
        excluded_elements: Element symbols that must not appear.
        stoichiometry_pattern: An abstract formula like ``"ABO3"`` or
            ``"AB2O4"``.
        space_group_number: A single space-group number (1–230).
        space_group_range: An inclusive (low, high) range of space-group
            numbers, typically set when a crystal system is specified.
        crystal_system: One of the seven crystal-system names (lowercase).
        max_atoms: Upper bound on atoms in the unit cell.
        stability_threshold_ev: Energy-above-hull threshold in eV/atom.
        chemistry_scope: Free-text scope string (e.g. ``"oxides"``).
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
# Parser — wires sub-parsers together, no parsing logic of its own
# ---------------------------------------------------------------------------

_MAX_ATOMS_DEFAULT = 20
_STABILITY_THRESHOLD_DEFAULT = 0.1


def parse_materials_constraints(text: str) -> MaterialsConstraints:
    """Parse natural-language constraint text into a MaterialsConstraints.

    Heuristic regex parser — handles common phrasings but is not exhaustive.
    Unrecognised tokens are silently skipped so that downstream callers
    always receive a valid (possibly default) result.

    Delegates to single-concern sub-parsers:
    - ``parse_element_list`` for allowed/excluded elements and chemistry scope
    - ``parse_stoichiometry_pattern`` for abstract formulas
    - ``parse_space_group`` for SG numbers, symbols, and crystal systems
    - ``parse_max_atoms`` / ``parse_stability_threshold`` for numeric limits

    Args:
        text: Free-form constraint text, e.g. ``"cubic perovskites ABO3 in
              Li-Fe-P-O, space group 62, <=20 atoms, stable within 0.05
              eV/atom"``.

    Returns:
        A ``MaterialsConstraints`` populated with every constraint the
        parser could extract.
    """
    constraints = MaterialsConstraints()

    if not text or not text.strip():
        return constraints

    # Numeric constraints
    constraints.max_atoms = parse_max_atoms(text, default=_MAX_ATOMS_DEFAULT)
    constraints.stability_threshold_ev = parse_stability_threshold(
        text, default=_STABILITY_THRESHOLD_DEFAULT,
    )

    # Space group and crystal system
    sg_number, crystal_system = parse_space_group(text, SG_SYMBOL_TO_NUMBER)
    constraints.space_group_number = sg_number
    constraints.crystal_system = crystal_system

    # Resolve crystal system → SG range only when no explicit SG number
    if crystal_system is not None and sg_number is None:
        constraints.space_group_range = crystal_system_to_sg_range(crystal_system)

    # Stoichiometry pattern
    constraints.stoichiometry_pattern = parse_stoichiometry_pattern(text)

    # Elements, exclusions, and chemistry scope
    allowed, excluded, scope = parse_element_list(text, ALLOWED_ELEMENTS)
    constraints.allowed_elements = allowed
    constraints.excluded_elements = excluded
    constraints.chemistry_scope = scope

    return constraints
