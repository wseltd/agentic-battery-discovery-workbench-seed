"""Tests for parse_space_group — the riskiest part is regex ambiguity
between SG symbols, crystal-system names, and numeric patterns.

Spend the most test effort on edge cases: out-of-range numbers, partial
symbol matches, and crystal-system extraction when an SG number is also
present.
"""

from __future__ import annotations

from agentic_materials_discovery.structure.parse_space_group import (
    CRYSTAL_SYSTEM_RANGES,
    parse_space_group,
)

# Minimal symbol dict for tests — avoids importing pymatgen in unit tests.
# Chose symbols that exercise partial-match and dash-containing names.
_TEST_SYMBOLS: dict[str, int] = {
    "Fm-3m": 225,
    "Fm-3": 202,
    "Pnma": 62,
    "P1": 1,
    "P21/c": 14,
    "R-3m": 166,
    "I4/mmm": 139,
}


# ---------------------------------------------------------------------------
# Explicit numeric SG references
# ---------------------------------------------------------------------------


def test_sg_number_space_group_keyword():
    """'space group 62' extracts SG number 62."""
    sg, crystal = parse_space_group("space group 62", _TEST_SYMBOLS)
    assert sg == 62
    assert crystal is None


def test_sg_number_sg_abbreviation():
    """'SG 225' extracts SG number 225."""
    sg, crystal = parse_space_group("SG 225", _TEST_SYMBOLS)
    assert sg == 225


def test_sg_number_spacegroup_no_space():
    """'spacegroup 14' (no space between words) is also matched."""
    sg, _ = parse_space_group("spacegroup 14", _TEST_SYMBOLS)
    assert sg == 14


def test_sg_number_case_insensitive():
    """'Space Group 62' with mixed case still matches."""
    sg, _ = parse_space_group("Space Group 62", _TEST_SYMBOLS)
    assert sg == 62


def test_sg_number_boundary_low():
    """SG number 1 (lowest valid) is accepted."""
    sg, _ = parse_space_group("space group 1", _TEST_SYMBOLS)
    assert sg == 1


def test_sg_number_boundary_high():
    """SG number 230 (highest valid) is accepted."""
    sg, _ = parse_space_group("space group 230", _TEST_SYMBOLS)
    assert sg == 230


def test_sg_number_out_of_range_zero():
    """SG number 0 is out of range — returns None."""
    sg, crystal = parse_space_group("space group 0", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


def test_sg_number_out_of_range_high():
    """SG number 300 is out of range — returns None."""
    sg, crystal = parse_space_group("space group 300", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


def test_sg_number_out_of_range_negative():
    """Negative numbers don't match the \\d+ regex, returns None."""
    sg, crystal = parse_space_group("space group -5", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


# ---------------------------------------------------------------------------
# Hermann-Mauguin symbol lookup
# ---------------------------------------------------------------------------


def test_sg_symbol_fm3m():
    """'Fm-3m' resolves to 225 via the symbol dict."""
    sg, _ = parse_space_group("structure in Fm-3m", _TEST_SYMBOLS)
    assert sg == 225


def test_sg_symbol_pnma():
    """'Pnma' resolves to 62."""
    sg, _ = parse_space_group("Pnma structure", _TEST_SYMBOLS)
    assert sg == 62


def test_sg_symbol_longest_match_wins():
    """'Fm-3m' must match before 'Fm-3' even though both are in the dict.

    This verifies the longest-first sorting prevents partial matches.
    """
    sg, _ = parse_space_group("in Fm-3m phase", _TEST_SYMBOLS)
    assert sg == 225  # not 202 (Fm-3)


def test_sg_symbol_with_slash():
    """'P21/c' containing a slash is matched correctly."""
    sg, _ = parse_space_group("synthesised in P21/c", _TEST_SYMBOLS)
    assert sg == 14


def test_sg_symbol_unknown_not_matched():
    """A symbol not in the dict returns None (no false positives)."""
    sg, crystal = parse_space_group("structure in Cmcm", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


# ---------------------------------------------------------------------------
# Crystal-system extraction
# ---------------------------------------------------------------------------


def test_crystal_system_cubic():
    """'cubic' is extracted as crystal system."""
    _, crystal = parse_space_group("cubic perovskites", _TEST_SYMBOLS)
    assert crystal == "cubic"


def test_crystal_system_orthorhombic():
    """'orthorhombic' is extracted."""
    _, crystal = parse_space_group("orthorhombic structures", _TEST_SYMBOLS)
    assert crystal == "orthorhombic"


def test_crystal_system_tetragonal():
    """'tetragonal' is extracted."""
    _, crystal = parse_space_group("tetragonal symmetry", _TEST_SYMBOLS)
    assert crystal == "tetragonal"


def test_crystal_system_case_insensitive():
    """Crystal-system match is case-insensitive."""
    _, crystal = parse_space_group("Hexagonal lattice", _TEST_SYMBOLS)
    assert crystal == "hexagonal"


def test_crystal_system_all_seven_recognised():
    """All seven crystal systems in CRYSTAL_SYSTEM_RANGES are detected."""
    for system in CRYSTAL_SYSTEM_RANGES:
        _, crystal = parse_space_group(f"some {system} material", _TEST_SYMBOLS)
        assert crystal == system, f"Failed to detect crystal system '{system}'"


# ---------------------------------------------------------------------------
# Combined and edge cases
# ---------------------------------------------------------------------------


def test_both_sg_number_and_crystal_system():
    """When both are present, both are returned."""
    sg, crystal = parse_space_group(
        "cubic, space group 225", _TEST_SYMBOLS,
    )
    assert sg == 225
    assert crystal == "cubic"


def test_sg_symbol_and_crystal_system():
    """SG symbol + crystal system name both extracted."""
    sg, crystal = parse_space_group(
        "cubic Fm-3m structure", _TEST_SYMBOLS,
    )
    assert sg == 225
    assert crystal == "cubic"


def test_empty_text_returns_none_none():
    """Empty string returns (None, None)."""
    sg, crystal = parse_space_group("", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


def test_whitespace_only_returns_none_none():
    """Whitespace-only string returns (None, None)."""
    sg, crystal = parse_space_group("   ", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


def test_no_match_returns_none_none():
    """Text with no space-group references returns (None, None)."""
    sg, crystal = parse_space_group("generate stable oxides", _TEST_SYMBOLS)
    assert (sg, crystal) == (None, None)


def test_numeric_preferred_over_symbol():
    """Explicit numeric SG takes precedence over symbol match.

    When both 'space group 62' and 'Fm-3m' appear, the number wins
    because numbers are tried first (more explicit).
    """
    sg, _ = parse_space_group("space group 62 in Fm-3m", _TEST_SYMBOLS)
    assert sg == 62


def test_empty_symbol_dict():
    """With an empty symbol dict, only numeric and crystal-system match."""
    sg, crystal = parse_space_group("cubic, space group 225", {})
    assert sg == 225
    assert crystal == "cubic"


def test_symbol_not_found_falls_through_to_crystal():
    """If symbol lookup fails, crystal system is still extracted."""
    sg, crystal = parse_space_group("cubic Cmcm", _TEST_SYMBOLS)
    assert sg is None  # Cmcm not in dict
    assert crystal == "cubic"
