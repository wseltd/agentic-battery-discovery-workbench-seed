"""Space-group extraction from natural-language text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

CRYSTAL_SYSTEM_RANGES: dict[str, tuple[int, int]] = {
    "triclinic": (1, 2),
    "monoclinic": (3, 15),
    "orthorhombic": (16, 74),
    "tetragonal": (75, 142),
    "trigonal": (143, 167),
    "hexagonal": (168, 194),
    "cubic": (195, 230),
}

_SG_NUMBER_RE = re.compile(
    r"space\s*group\s+(\d+)|SG\s+(\d+)",
    re.IGNORECASE,
)


def parse_space_group(
    text: str,
    symbol_to_number: dict[str, int],
) -> tuple[int | None, str | None]:
    """Extract space group number and crystal system name from text.

    Args:
        text: Free-form constraint text.
        symbol_to_number: Mapping from Hermann-Mauguin symbol strings to
            their integer space-group numbers (1-230).

    Returns:
        A 2-tuple ``(sg_number, crystal_system)``.
    """
    if not text or not text.strip():
        return (None, None)

    sg_number = _extract_sg_number(text)
    if sg_number is None:
        sg_number = _extract_sg_symbol(text, symbol_to_number)

    crystal_system = _extract_crystal_system(text)

    return (sg_number, crystal_system)


def _extract_sg_number(text: str) -> int | None:
    """Match explicit numeric space-group references."""
    m = _SG_NUMBER_RE.search(text)
    if not m:
        return None
    num = int(m.group(1) or m.group(2))
    if 1 <= num <= 230:
        return num
    logger.warning("Space group number %d out of range 1-230, ignored", num)
    return None


def _extract_sg_symbol(
    text: str,
    symbol_to_number: dict[str, int],
) -> int | None:
    """Match Hermann-Mauguin symbols by longest-first search."""
    for symbol in sorted(symbol_to_number, key=len, reverse=True):
        if symbol in text:
            return symbol_to_number[symbol]
    return None


def _extract_crystal_system(text: str) -> str | None:
    """Match one of the seven crystal-system names (case-insensitive)."""
    lower = text.lower()
    for system in CRYSTAL_SYSTEM_RANGES:
        if system in lower:
            return system
    return None
