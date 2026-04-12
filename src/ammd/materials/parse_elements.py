"""Element-list extraction from natural-language text.

Regex-based parser that recognises common element-list formats used in
materials science queries.  No LLM calls — pure pattern matching.

Supported formats:
- Hyphenated: ``"Li-Fe-P-O"``
- Comma-separated with keyword prefix: ``"containing Li, Fe, P, and O"``
- System notation: ``"in the Li-Fe-P-O system"``
- Exclusion: ``"excluding Pb and Cd"``
- Full element names: ``"containing lithium, iron, phosphorus and oxygen"``

Each extracted symbol is validated against a caller-supplied ``allowed``
frozenset so that typos and non-element tokens are silently dropped.
"""

from __future__ import annotations

import re

from pymatgen.core.periodic_table import Element

# Regex for a single element symbol: uppercase letter + optional lowercase.
_ELEMENT_RE = r"[A-Z][a-z]?"

# Lazy-built element name → symbol map (built once, reused).
_ELEMENT_NAME_MAP: dict[str, str] | None = None

# Chemistry-scope keywords recognised by the parser.
_SCOPE_KEYWORDS: list[str] = [
    "oxides", "nitrides", "sulfides", "carbides", "halides",
    "fluorides", "chlorides", "bromides", "iodides",
    "borides", "phosphides", "silicides", "selenides",
    "tellurides", "intermetallics", "perovskites",
]


def _get_element_name_map() -> dict[str, str]:
    """Return a lowercase element-name → symbol map, built once."""
    global _ELEMENT_NAME_MAP  # noqa: PLW0603
    if _ELEMENT_NAME_MAP is None:
        _ELEMENT_NAME_MAP = {}
        for el in Element:
            if el.Z <= 83:
                _ELEMENT_NAME_MAP[el.long_name.lower()] = el.symbol
    return _ELEMENT_NAME_MAP


def _extract_hyphenated(text: str, allowed: frozenset[str]) -> list[str] | None:
    """Match dash-separated element lists like 'Li-Fe-P-O'."""
    m = re.search(
        r"(?:in\s+(?:the\s+)?|from\s+|containing\s+|elements?\s+)?"
        r"(" + _ELEMENT_RE + r"(?:\s*-\s*" + _ELEMENT_RE + r"){2,})"
        r"(?:\s+system)?",
        text,
    )
    if not m:
        return None
    symbols = re.findall(_ELEMENT_RE, m.group(1))
    valid = [s for s in symbols if s in allowed]
    return valid or None


def _extract_comma_separated(
    text: str, allowed: frozenset[str],
) -> list[str] | None:
    """Match comma-separated lists like 'containing Li, Fe, P, and O'."""
    # Require a keyword prefix to avoid false positives on random text.
    m = re.search(
        r"(?:elements?\s*[:=]?\s*|containing\s+|from\s+|in\s+|with\s+)"
        r"(" + _ELEMENT_RE + r"(?:\s*,\s*(?:and\s+)?" + _ELEMENT_RE + r")"
        r"+(?:\s*,?\s*and\s+" + _ELEMENT_RE + r")?)",
        text,
    )
    if not m:
        return None
    symbols = re.findall(_ELEMENT_RE, m.group(1))
    valid = [s for s in symbols if s in allowed]
    return valid or None


def _extract_natural_language(
    text: str, allowed: frozenset[str],
) -> list[str] | None:
    """Match full element names like 'lithium, iron, phosphorus and oxygen'."""
    name_map = _get_element_name_map()
    lower = text.lower()
    found: list[str] = []
    for name, symbol in name_map.items():
        if name in lower and symbol in allowed:
            found.append(symbol)
    # Require at least two elements to avoid false positives
    return found if len(found) >= 2 else None


def _extract_excluded(text: str, allowed: frozenset[str]) -> list[str] | None:
    """Match exclusion patterns like 'excluding Pb and Cd'."""
    m = re.search(
        r"(?:exclud(?:e|ing)|without|no)\s+"
        r"(" + _ELEMENT_RE + r"(?:[\s,]+(?:and\s+)?" + _ELEMENT_RE + r")*)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    # For exclusions, validate against the full periodic table (Z<=83),
    # not just the allowed set — a user might exclude Tc even though
    # it's not in the allowed set.
    all_symbols = frozenset(el.symbol for el in Element if el.Z <= 83)
    symbols = re.findall(_ELEMENT_RE, m.group(1))
    valid = [s for s in symbols if s in all_symbols]
    return valid or None


def _extract_scope(text: str) -> str | None:
    """Match chemistry scope keywords like 'oxides', 'nitrides'."""
    lower = text.lower()
    found = [kw for kw in _SCOPE_KEYWORDS if kw in lower]
    return " ".join(found) if found else None


def parse_element_list(
    text: str,
    allowed: frozenset[str],
) -> tuple[list[str] | None, list[str] | None, str | None]:
    """Extract element constraints from natural-language text.

    Tries multiple formats in order of specificity (hyphenated first,
    then comma-separated, then full element names).  Each symbol is
    validated against *allowed*; unrecognised symbols are silently
    dropped.

    Args:
        text: Free-form constraint text, e.g. ``"in Li-Fe-P-O excluding
              Pb"``.
        allowed: Frozenset of valid element symbols to accept for the
                 allowed-element list.

    Returns:
        A 3-tuple ``(allowed_elements, excluded_elements, chemistry_scope)``.
        Any component is ``None`` if the corresponding pattern was not found.
        Returns ``(None, None, None)`` if no element patterns are detected.
    """
    if not text or not text.strip():
        return (None, None, None)

    # Try formats in order of specificity — hyphenated is the most
    # unambiguous, comma-separated next, natural-language last.
    allowed_elements = (
        _extract_hyphenated(text, allowed)
        or _extract_comma_separated(text, allowed)
        or _extract_natural_language(text, allowed)
    )

    excluded_elements = _extract_excluded(text, allowed)
    chemistry_scope = _extract_scope(text)

    return (allowed_elements, excluded_elements, chemistry_scope)
