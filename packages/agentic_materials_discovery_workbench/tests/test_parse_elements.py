"""Tests for parse_element_list — the regex-heavy element extraction function.

The parser is heuristic, so tests focus on the tricky regex edge cases:
ambiguous symbols, mixed formats, boundary conditions, and false-positive
resistance.
"""

from __future__ import annotations

from agentic_materials_discovery.structure.constraints import ALLOWED_ELEMENTS
from agentic_materials_discovery.structure.parse_elements import parse_element_list


# ---------------------------------------------------------------------------
# Hyphenated format
# ---------------------------------------------------------------------------


def test_hyphenated_basic():
    """Standard dash-separated list is parsed correctly."""
    allowed, excluded, scope = parse_element_list(
        "Generate structures in Li-Fe-P-O", ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}
    assert excluded is None


def test_hyphenated_with_system_suffix():
    """'in the Li-Fe-P-O system' should work with 'the' and 'system'."""
    allowed, _, _ = parse_element_list(
        "in the Li-Fe-P-O system", ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}


def test_hyphenated_invalid_symbol_dropped():
    """Unknown symbols like 'Zz' are silently dropped from hyphenated lists."""
    allowed, _, _ = parse_element_list("in Li-Zz-Fe-O", ALLOWED_ELEMENTS)
    assert allowed is not None
    assert "Zz" not in allowed
    assert set(allowed) == {"Li", "Fe", "O"}


# ---------------------------------------------------------------------------
# Comma-separated format
# ---------------------------------------------------------------------------


def test_comma_separated_with_containing():
    """'containing Li, Fe, P, O' is parsed."""
    allowed, _, _ = parse_element_list(
        "containing Li, Fe, P, O", ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}


def test_comma_separated_with_and():
    """'containing Li, Fe, P, and O' handles trailing 'and'."""
    allowed, _, _ = parse_element_list(
        "containing Li, Fe, P, and O", ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}


# ---------------------------------------------------------------------------
# Natural-language element names
# ---------------------------------------------------------------------------


def test_natural_language_element_names():
    """Full element names like 'lithium, iron' are resolved to symbols."""
    allowed, _, _ = parse_element_list(
        "containing lithium, iron, phosphorus and oxygen", ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}


def test_natural_language_single_name_returns_none():
    """A single element name should not trigger — too ambiguous."""
    allowed, excluded, scope = parse_element_list("contains iron", ALLOWED_ELEMENTS)
    assert (allowed, excluded, scope) == (None, None, None)


# ---------------------------------------------------------------------------
# Exclusion patterns
# ---------------------------------------------------------------------------


def test_excluding_pattern():
    """'excluding Pb and Cd' populates excluded_elements."""
    _, excluded, _ = parse_element_list(
        "excluding Pb and Cd", ALLOWED_ELEMENTS,
    )
    assert excluded is not None
    assert set(excluded) == {"Pb", "Cd"}


def test_exclude_without_keyword():
    """'without Tc' uses the without-keyword path."""
    _, excluded, _ = parse_element_list("without Tc", ALLOWED_ELEMENTS)
    assert excluded is not None
    assert "Tc" in excluded


def test_exclude_radioactive_not_in_allowed():
    """Excluding Tc works even though Tc is not in ALLOWED_ELEMENTS."""
    _, excluded, _ = parse_element_list("exclude Tc, Pm", ALLOWED_ELEMENTS)
    assert excluded is not None
    assert set(excluded) == {"Tc", "Pm"}


# ---------------------------------------------------------------------------
# Chemistry scope
# ---------------------------------------------------------------------------


def test_scope_oxides():
    """'oxides' is extracted as chemistry scope."""
    _, _, scope = parse_element_list("generate oxides", ALLOWED_ELEMENTS)
    assert scope is not None
    assert "oxides" in scope


def test_scope_multiple_keywords():
    """Multiple scope keywords are combined."""
    _, _, scope = parse_element_list(
        "nitrides and carbides", ALLOWED_ELEMENTS,
    )
    assert scope is not None
    assert "nitrides" in scope
    assert "carbides" in scope


# ---------------------------------------------------------------------------
# Edge cases and combined input
# ---------------------------------------------------------------------------


def test_empty_text_returns_triple_none():
    """Empty or whitespace-only text returns (None, None, None)."""
    assert parse_element_list("", ALLOWED_ELEMENTS) == (None, None, None)
    assert parse_element_list("   ", ALLOWED_ELEMENTS) == (None, None, None)


def test_no_patterns_returns_triple_none():
    """Text with no recognisable patterns returns (None, None, None)."""
    assert parse_element_list(
        "make something interesting", ALLOWED_ELEMENTS,
    ) == (None, None, None)


def test_combined_allowed_excluded_scope():
    """All three components extracted from one sentence."""
    allowed, excluded, scope = parse_element_list(
        "Generate oxides in Li-Fe-P-O excluding Pb",
        ALLOWED_ELEMENTS,
    )
    assert allowed is not None
    assert set(allowed) == {"Li", "Fe", "P", "O"}
    assert excluded is not None
    assert "Pb" in excluded
    assert scope is not None
    assert "oxides" in scope


def test_all_symbols_invalid_returns_none():
    """If every symbol in a hyphenated list is invalid, return None."""
    allowed, excluded, scope = parse_element_list("in Xx-Yy-Zz", ALLOWED_ELEMENTS)
    assert (allowed, excluded, scope) == (None, None, None)
