"""Constraint parsing for discovery workbench property ranges.

Parses human-readable range strings like '300-450 Da' or '<=0.1 eV/atom'
into structured ParsedRange objects with optional unit normalisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit aliases — single source of truth for unit normalisation.
# Maps variant spellings to their canonical form.
UNIT_ALIASES: dict[str, str] = {
    "Da": "Da",
    "dalton": "Da",
    "daltons": "Da",
    "g/mol": "Da",
    "eV/atom": "eV/atom",
    "Å²": "Å²",
    "angstrom²": "Å²",
    "A2": "Å²",
    "angstrom^2": "Å²",
    "GPa": "GPa",
    "gpa": "GPa",
    "eV": "eV",
    "ev": "eV",
    "Å": "Å",
    "angstrom": "Å",
    "A": "Å",
}

# Pre-built case-insensitive lookup for normalise_unit.
_UNIT_ALIASES_LOWER: dict[str, str] = {k.lower(): v for k, v in UNIT_ALIASES.items()}

# Regex for range strings: supports "MIN-MAX", "<=X", ">=X", "<X", ">X"
# with optional trailing unit. Decimal numbers supported.
_RANGE_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?P<min_val>[0-9]*\.?[0-9]+)\s*[-–]\s*(?P<max_val>[0-9]*\.?[0-9]+)"  # range
    r"|(?P<op><=?|>=?)\s*(?P<op_val>[0-9]*\.?[0-9]+)"  # comparison
    r")"
    r"(?:\s+(?P<unit>\S+))?"  # optional unit
    r"\s*$"
)


@dataclass
class ParsedRange:
    """A parsed numeric range with optional unit."""

    min_val: float | None
    max_val: float | None
    unit: str | None


def normalise_unit(unit: str) -> str:
    """Normalise a unit string to its canonical form via UNIT_ALIASES.

    Args:
        unit: Raw unit string (e.g. 'daltons', 'gpa').

    Returns:
        Canonical unit string. Unknown units are returned unchanged.
    """
    # Try exact match first (handles case-sensitive aliases like 'A2'),
    # then fall back to case-insensitive lookup.
    if unit in UNIT_ALIASES:
        return UNIT_ALIASES[unit]
    return _UNIT_ALIASES_LOWER.get(unit.lower(), unit)


def parse_range(text: str) -> ParsedRange:
    """Parse a human-readable range string into a ParsedRange.

    Supported formats:
        '300-450'       -> ParsedRange(300.0, 450.0, None)
        '<=0.1'         -> ParsedRange(None, 0.1, None)
        '>=5'           -> ParsedRange(5.0, None, None)
        '<10'           -> ParsedRange(None, 10.0, None)
        '>2'            -> ParsedRange(2.0, None, None)
        '5-6 eV'        -> ParsedRange(5.0, 6.0, 'eV')

    Args:
        text: The range string to parse.

    Returns:
        ParsedRange with parsed values and normalised unit.

    Raises:
        ValueError: If the string cannot be parsed as a valid range.
    """
    match = _RANGE_RE.match(text)
    if not match:
        raise ValueError(
            f"Cannot parse range: {text!r}. "
            f"Expected formats: 'MIN-MAX', '<=X', '>=X', '<X', '>X', "
            f"optionally followed by a unit."
        )

    raw_unit = match.group("unit")
    unit = normalise_unit(raw_unit) if raw_unit else None

    if match.group("min_val") is not None:
        return ParsedRange(
            min_val=float(match.group("min_val")),
            max_val=float(match.group("max_val")),
            unit=unit,
        )

    op = match.group("op")
    val = float(match.group("op_val"))

    if op in ("<=", "<"):
        return ParsedRange(min_val=None, max_val=val, unit=unit)
    # op in (">=", ">")
    return ParsedRange(min_val=val, max_val=None, unit=unit)
