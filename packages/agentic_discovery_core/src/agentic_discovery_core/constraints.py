"""Constraint parsing for discovery workbench property ranges.

Parses human-readable range strings like '300-450 Da' or '<=0.1 eV/atom'
into structured ParsedRange objects with optional unit normalisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit aliases -- single source of truth for unit normalisation.
# Maps variant spellings to their canonical form.
UNIT_ALIASES: dict[str, str] = {
    "Da": "Da",
    "dalton": "Da",
    "daltons": "Da",
    "g/mol": "Da",
    "eV/atom": "eV/atom",
    "A2": "A2",
    "angstrom^2": "A2",
    "GPa": "GPa",
    "gpa": "GPa",
    "eV": "eV",
    "ev": "eV",
    "A": "A",
    "angstrom": "A",
}

# Pre-built case-insensitive lookup for normalise_unit.
_UNIT_ALIASES_LOWER: dict[str, str] = {k.lower(): v for k, v in UNIT_ALIASES.items()}

# Regex for range strings: supports "MIN-MAX", "<=X", ">=X", "<X", ">X"
# with optional trailing unit. Decimal numbers supported.
_RANGE_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?P<min_val>[0-9]*\.?[0-9]+)\s*[-\u2013]\s*(?P<max_val>[0-9]*\.?[0-9]+)"  # range
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
