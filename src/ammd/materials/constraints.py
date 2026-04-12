"""Materials constraint parsing for inorganic materials discovery requests.

Parses natural-language constraint text into a structured MaterialsConstraints
dataclass.  Parsing is regex-based and heuristic — it handles common phrasings
but is not a full NLP pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
import re

from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup

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

# Crystal-system name → inclusive (min, max) space-group-number range.
_CRYSTAL_SYSTEM_RANGES: dict[str, tuple[int, int]] = {
    "triclinic": (1, 2),
    "monoclinic": (3, 15),
    "orthorhombic": (16, 74),
    "tetragonal": (75, 142),
    "trigonal": (143, 167),
    "hexagonal": (168, 194),
    "cubic": (195, 230),
}

# Regex for a single element symbol: uppercase letter + optional lowercase
_ELEMENT_RE = r"[A-Z][a-z]?"


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
# Parser
# ---------------------------------------------------------------------------

def parse_materials_constraints(text: str) -> MaterialsConstraints:
    """Parse natural-language constraint text into a MaterialsConstraints.

    Heuristic regex parser — handles common phrasings but is not exhaustive.
    Unrecognised tokens are silently skipped so that downstream callers
    always receive a valid (possibly default) result.

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

    _parse_stability_threshold(text, constraints)
    _parse_max_atoms(text, constraints)
    _parse_space_group(text, constraints)
    _parse_crystal_system(text, constraints)
    _parse_stoichiometry(text, constraints)
    _parse_elements(text, constraints)
    _parse_excluded_elements(text, constraints)
    _parse_chemistry_scope(text, constraints)

    return constraints


# ---------------------------------------------------------------------------
# Private parse helpers — one concern each
# ---------------------------------------------------------------------------

def _parse_stability_threshold(text: str, c: MaterialsConstraints) -> None:
    """Extract stability threshold in eV/atom."""
    # Matches: "stable within 0.05 eV/atom", "stability < 0.1 eV",
    #          "0.05 eV/atom stability", "threshold 0.05 eV"
    m = re.search(
        r"(?:stable\s+within|stability\s*(?:threshold)?[<≤:=\s]+|"
        r"threshold\s*[<≤:=\s]+)"
        r"\s*(\d+(?:\.\d+)?)\s*ev",
        text,
        re.IGNORECASE,
    )
    if m:
        c.stability_threshold_ev = float(m.group(1))
        return
    # "0.05 eV/atom" standalone
    m = re.search(r"(\d+(?:\.\d+)?)\s*ev/atom", text, re.IGNORECASE)
    if m:
        c.stability_threshold_ev = float(m.group(1))


def _parse_max_atoms(text: str, c: MaterialsConstraints) -> None:
    """Extract maximum atom count."""
    # "<=20 atoms", "≤20 atoms", "up to 30 atoms", "at most 20 atoms",
    # "max 20 atoms", "maximum 20 atoms"
    m = re.search(
        r"(?:[<≤]=?\s*|up\s+to\s+|at\s+most\s+|max(?:imum)?\s+)"
        r"(\d+)\s*atoms?",
        text,
        re.IGNORECASE,
    )
    if m:
        c.max_atoms = int(m.group(1))


def _parse_space_group(text: str, c: MaterialsConstraints) -> None:
    """Extract space group number from explicit number or Hermann-Mauguin symbol."""
    # "space group 62", "SG 62", "spacegroup 62"
    m = re.search(
        r"space\s*group\s+(\d+)|SG\s+(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        num = int(m.group(1) or m.group(2))
        if 1 <= num <= 230:
            c.space_group_number = num
        else:
            logger.warning(
                "Space group number %d out of range 1–230, ignored", num
            )
        return

    # Hermann-Mauguin symbol, e.g. "Fm-3m", "P21/c", "Pnma".
    # Sort longest-first so "Fm-3m" matches before "Fm-3".
    for symbol in sorted(SG_SYMBOL_TO_NUMBER, key=len, reverse=True):
        if symbol in text:
            c.space_group_number = SG_SYMBOL_TO_NUMBER[symbol]
            return


def _parse_crystal_system(text: str, c: MaterialsConstraints) -> None:
    """Extract crystal system name and set the corresponding SG range."""
    lower = text.lower()
    for system, sg_range in _CRYSTAL_SYSTEM_RANGES.items():
        if system in lower:
            c.crystal_system = system
            # Only set SG range if no explicit SG number was given
            if c.space_group_number is None:
                c.space_group_range = sg_range
            return


def _parse_stoichiometry(text: str, c: MaterialsConstraints) -> None:
    """Extract abstract stoichiometry pattern like ABO3 or AB2O4.

    Matches uppercase-letter tokens with digits that look like stoichiometry
    rather than element lists.  We require at least one digit to distinguish
    from plain element symbols.
    """
    # Matches: ABO3, AB2O4, A2B2O7, etc.
    m = re.search(r"\b([A-Z][A-Z0-9]{2,})\b", text)
    if m:
        candidate = m.group(1)
        # Must contain at least one digit and at least two distinct uppercase
        # letters to look like a stoichiometry pattern
        if re.search(r"\d", candidate) and len(set(re.findall(r"[A-Z]", candidate))) >= 2:
            c.stoichiometry_pattern = candidate


def _parse_elements(text: str, c: MaterialsConstraints) -> None:
    """Extract allowed element list from hyphenated, comma-separated, or natural-language forms."""
    # Hyphenated: "Li-Fe-P-O", "in Li-Fe-P-O"
    m = re.search(
        r"(?:in\s+|from\s+|containing\s+|elements?\s+)?"
        r"(" + _ELEMENT_RE + r"(?:\s*-\s*" + _ELEMENT_RE + r"){2,})",
        text,
    )
    if m:
        elements = re.findall(_ELEMENT_RE, m.group(1))
        valid = [e for e in elements if e in ALLOWED_ELEMENTS]
        if valid:
            c.allowed_elements = valid
            return

    # Comma-separated: "Li, Fe, P, O" or "Li,Fe,P,O"
    m = re.search(
        r"(?:elements?\s*[:=]?\s*|containing\s+|from\s+|in\s+|with\s+)"
        r"(" + _ELEMENT_RE + r"(?:\s*,\s*" + _ELEMENT_RE + r")+)",
        text,
    )
    if m:
        elements = re.findall(_ELEMENT_RE, m.group(1))
        valid = [e for e in elements if e in ALLOWED_ELEMENTS]
        if valid:
            c.allowed_elements = valid
            return

    # Natural language: "containing lithium, iron, phosphorus and oxygen"
    # Map common element names to symbols
    name_map = _build_element_name_map()
    lower = text.lower()
    found: list[str] = []
    for name, symbol in name_map.items():
        if name in lower and symbol in ALLOWED_ELEMENTS:
            found.append(symbol)
    if len(found) >= 2:
        c.allowed_elements = found


def _parse_excluded_elements(text: str, c: MaterialsConstraints) -> None:
    """Extract excluded elements from 'exclude X, Y' or 'without X, Y' patterns."""
    m = re.search(
        r"(?:exclud(?:e|ing)|without|no)\s+"
        r"(" + _ELEMENT_RE + r"(?:[\s,]+(?:and\s+)?" + _ELEMENT_RE + r")*)",
        text,
        re.IGNORECASE,
    )
    if m:
        elements = re.findall(_ELEMENT_RE, m.group(1))
        valid = [e for e in elements if e in ALLOWED_ELEMENTS or e in _EXCLUDED_SYMBOLS]
        if valid:
            c.excluded_elements = valid


def _parse_chemistry_scope(text: str, c: MaterialsConstraints) -> None:
    """Extract chemistry scope keywords like 'oxides', 'nitrides', etc."""
    scope_keywords = [
        "oxides", "nitrides", "sulfides", "carbides", "halides",
        "fluorides", "chlorides", "bromides", "iodides",
        "borides", "phosphides", "silicides", "selenides",
        "tellurides", "intermetallics", "perovskites",
    ]
    lower = text.lower()
    found = [kw for kw in scope_keywords if kw in lower]
    if found:
        c.chemistry_scope = " ".join(found)


# Built once at module level — not per call (standard §5/§15)
_ELEMENT_NAME_MAP: dict[str, str] | None = None


def _build_element_name_map() -> dict[str, str]:
    """Lazy-build a lowercase element-name → symbol map.

    Cached in module state so the iteration over Element happens only once.
    """
    global _ELEMENT_NAME_MAP  # noqa: PLW0603
    if _ELEMENT_NAME_MAP is None:
        _ELEMENT_NAME_MAP = {}
        for el in Element:
            if el.Z <= 83:
                _ELEMENT_NAME_MAP[el.long_name.lower()] = el.symbol
    return _ELEMENT_NAME_MAP
