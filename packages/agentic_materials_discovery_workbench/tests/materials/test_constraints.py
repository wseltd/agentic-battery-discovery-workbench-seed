"""Tests for MaterialsConstraints and parse_materials_constraints.

Focused on the regex-based parser which is the riskiest part of this module.
"""

from __future__ import annotations

from agentic_materials_discovery.structure.constraints import (
    ALLOWED_ELEMENTS,
    SG_SYMBOL_TO_NUMBER,
    parse_materials_constraints,
)


# ---------------------------------------------------------------------------
# Element-list parsing
# ---------------------------------------------------------------------------


def test_parse_element_list_hyphenated():
    """Hyphenated element lists like 'Li-Fe-P-O' are a common shorthand."""
    result = parse_materials_constraints("Generate structures in Li-Fe-P-O")
    assert result.allowed_elements is not None
    assert set(result.allowed_elements) == {"Li", "Fe", "P", "O"}


def test_parse_element_list_comma_separated():
    """Comma-separated lists with a keyword prefix."""
    result = parse_materials_constraints("containing Li, Fe, P, O")
    assert result.allowed_elements is not None
    assert set(result.allowed_elements) == {"Li", "Fe", "P", "O"}


def test_parse_element_list_natural_language():
    """Full element names like 'lithium, iron, phosphorus and oxygen'."""
    result = parse_materials_constraints(
        "containing lithium, iron, phosphorus and oxygen"
    )
    assert result.allowed_elements is not None
    assert set(result.allowed_elements) == {"Li", "Fe", "P", "O"}


# ---------------------------------------------------------------------------
# Space-group parsing
# ---------------------------------------------------------------------------


def test_parse_sg_number():
    """Explicit 'space group 62' sets space_group_number."""
    result = parse_materials_constraints("space group 62")
    assert result.space_group_number == 62


def test_parse_sg_symbol_fmbar3m():
    """Hermann-Mauguin symbol 'Fm-3m' resolves to SG 225."""
    result = parse_materials_constraints("structure in Fm-3m")
    assert result.space_group_number == SG_SYMBOL_TO_NUMBER["Fm-3m"]
    assert result.space_group_number == 225


# ---------------------------------------------------------------------------
# Crystal-system parsing
# ---------------------------------------------------------------------------


def test_parse_crystal_system_cubic():
    """'cubic' sets crystal_system and the cubic SG range."""
    result = parse_materials_constraints("cubic perovskites")
    assert result.crystal_system == "cubic"
    assert result.space_group_range == (195, 230)


def test_parse_crystal_system_orthorhombic():
    """'orthorhombic' sets crystal_system and the orthorhombic SG range."""
    result = parse_materials_constraints("orthorhombic structures")
    assert result.crystal_system == "orthorhombic"
    assert result.space_group_range == (16, 74)


# ---------------------------------------------------------------------------
# Stoichiometry parsing
# ---------------------------------------------------------------------------


def test_parse_stoichiometry_abo3():
    """'ABO3' is recognised as a perovskite stoichiometry pattern."""
    result = parse_materials_constraints("cubic perovskites ABO3")
    assert result.stoichiometry_pattern == "ABO3"


def test_parse_stoichiometry_ab2o4():
    """'AB2O4' is recognised as a spinel stoichiometry pattern."""
    result = parse_materials_constraints("spinel AB2O4 structures")
    assert result.stoichiometry_pattern == "AB2O4"


# ---------------------------------------------------------------------------
# Max-atoms parsing
# ---------------------------------------------------------------------------


def test_parse_max_atoms_leq():
    """'<=20 atoms' sets max_atoms to 20."""
    result = parse_materials_constraints("<=20 atoms")
    assert result.max_atoms == 20


def test_parse_max_atoms_up_to():
    """'up to 30 atoms' sets max_atoms to 30."""
    result = parse_materials_constraints("up to 30 atoms")
    assert result.max_atoms == 30


# ---------------------------------------------------------------------------
# Stability-threshold parsing
# ---------------------------------------------------------------------------


def test_parse_stability_threshold():
    """'stable within 0.05 eV/atom' sets stability_threshold_ev."""
    result = parse_materials_constraints("stable within 0.05 eV/atom")
    assert abs(result.stability_threshold_ev - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Excluded-elements parsing
# ---------------------------------------------------------------------------


def test_parse_excluded_elements_tc_pm():
    """'exclude Tc, Pm' sets excluded_elements."""
    result = parse_materials_constraints("exclude Tc, Pm")
    assert result.excluded_elements is not None
    assert set(result.excluded_elements) == {"Tc", "Pm"}


# ---------------------------------------------------------------------------
# Edge cases and defaults
# ---------------------------------------------------------------------------


def test_parse_empty_text_returns_defaults():
    """Empty or whitespace-only text returns all defaults."""
    result = parse_materials_constraints("")
    assert result.allowed_elements is None
    assert result.excluded_elements is None
    assert result.stoichiometry_pattern is None
    assert result.space_group_number is None
    assert result.space_group_range is None
    assert result.crystal_system is None
    assert result.max_atoms == 20
    assert abs(result.stability_threshold_ev - 0.1) < 1e-9
    assert result.chemistry_scope is None

    result_ws = parse_materials_constraints("   ")
    assert result_ws.allowed_elements is None


def test_parse_combined_constraints():
    """Multiple constraints in one sentence are all extracted."""
    result = parse_materials_constraints(
        "Generate oxides in Li-Fe-P-O, space group 62, "
        "<=20 atoms, stable within 0.05 eV/atom"
    )
    assert set(result.allowed_elements) == {"Li", "Fe", "P", "O"}
    assert result.space_group_number == 62
    assert result.max_atoms == 20
    assert abs(result.stability_threshold_ev - 0.05) < 1e-9
    assert result.chemistry_scope == "oxides"


def test_invalid_element_symbol_ignored():
    """Unknown symbols like 'Zz' are silently dropped."""
    result = parse_materials_constraints("in Li-Zz-Fe-O")
    assert result.allowed_elements is not None
    assert "Zz" not in result.allowed_elements
    assert set(result.allowed_elements) == {"Li", "Fe", "O"}


def test_sg_number_out_of_range_ignored():
    """Space group numbers outside 1-230 are ignored, defaults preserved."""
    result = parse_materials_constraints("space group 300")
    assert result.space_group_number is None
    # Verify the out-of-range number did not leak through as a value
    assert result.space_group_range is None
    assert result.max_atoms == 20
    assert abs(result.stability_threshold_ev - 0.1) < 1e-9


def test_parse_chemistry_scope_string():
    """Chemistry scope keywords like 'oxides' are extracted."""
    result = parse_materials_constraints("generate oxides")
    assert result.chemistry_scope is not None
    assert "oxides" in result.chemistry_scope


# ---------------------------------------------------------------------------
# Constant sanity checks
# ---------------------------------------------------------------------------


def test_allowed_elements_excludes_noble_gases_and_radioactive():
    """ALLOWED_ELEMENTS must not contain Tc, Pm, or noble gases."""
    for sym in ("Tc", "Pm", "He", "Ne", "Ar", "Kr", "Xe"):
        assert sym not in ALLOWED_ELEMENTS


def test_allowed_elements_contains_common_metals():
    """Spot-check that common elements are present."""
    for sym in ("Li", "Fe", "O", "Si", "C", "N", "Ti", "Cu", "Zn"):
        assert sym in ALLOWED_ELEMENTS


def test_sg_symbol_to_number_has_230_entries():
    """All 230 space groups should be mapped."""
    assert len(SG_SYMBOL_TO_NUMBER) == 230
    assert SG_SYMBOL_TO_NUMBER["Fm-3m"] == 225
    assert SG_SYMBOL_TO_NUMBER["Pnma"] == 62
