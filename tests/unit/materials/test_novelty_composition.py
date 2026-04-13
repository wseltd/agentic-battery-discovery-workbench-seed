"""Tests for composition-based novelty pre-filtering utilities."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.novelty_composition import (
    build_composition_filter,
    get_reduced_formula,
    group_structures_by_composition,
)


# ---------------------------------------------------------------------------
# Helpers — reusable structure builders
# ---------------------------------------------------------------------------


def _nacl_rocksalt() -> Structure:
    """NaCl rock-salt conventional cell (Fm-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ]
    return Structure(lattice, species, coords)


def _fe_bcc() -> Structure:
    """BCC iron primitive cell (Im-3m, 2 atoms)."""
    lattice = Lattice.cubic(2.87)
    return Structure(lattice, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def _tio2_rutile() -> Structure:
    """Rutile TiO2 unit cell (P42/mnm, 6 atoms)."""
    lattice = Lattice.tetragonal(4.594, 2.959)
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.3053, 0.3053, 0.0],
        [0.6947, 0.6947, 0.0],
        [0.1947, 0.8053, 0.5],
        [0.8053, 0.1947, 0.5],
    ]
    return Structure(lattice, species, coords)


def _na4cl4_supercell() -> Structure:
    """Na4Cl4 supercell — same reduced formula as NaCl.

    Tests that formula reduction correctly normalises stoichiometry.
    """
    lattice = Lattice.cubic(11.28)
    species = (
        ["Na"] * 4 * 8 + ["Cl"] * 4 * 8
    )
    # Simple 2x2x2 supercell placement — exact coords don't matter
    # for composition testing, just that we have 32Na + 32Cl.
    coords = []
    for i in range(32):
        coords.append([i * 0.03, 0.0, 0.0])
    for i in range(32):
        coords.append([0.5 + i * 0.01, 0.5, 0.5])
    return Structure(lattice, species, coords)


# ---------------------------------------------------------------------------
# get_reduced_formula tests
# ---------------------------------------------------------------------------


class TestGetReducedFormula:
    """Tests for get_reduced_formula — the core formula extraction."""

    def test_binary_compound_nacl(self) -> None:
        """NaCl rock-salt should return 'NaCl'."""
        assert get_reduced_formula(_nacl_rocksalt()) == "NaCl"

    def test_elemental_iron(self) -> None:
        """Elemental Fe should return 'Fe'."""
        assert get_reduced_formula(_fe_bcc()) == "Fe"

    def test_ternary_oxide_tio2(self) -> None:
        """TiO2 rutile should return 'TiO2'."""
        assert get_reduced_formula(_tio2_rutile()) == "TiO2"

    def test_supercell_reduces_to_same_formula(self) -> None:
        """Na4Cl4 supercell should reduce to 'NaCl', not 'Na4Cl4'.

        This is the key property: formula reduction normalises away
        the cell multiplicity so all NaCl variants group together.
        """
        assert get_reduced_formula(_na4cl4_supercell()) == "NaCl"

    def test_empty_structure_raises(self) -> None:
        """An empty structure (zero sites) must raise ValueError."""
        lattice = Lattice.cubic(5.0)
        empty = Structure(lattice, [], [])
        with pytest.raises(ValueError, match="empty structure") as exc_info:
            get_reduced_formula(empty)
        assert "empty structure" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_composition_filter tests
# ---------------------------------------------------------------------------


class TestBuildCompositionFilter:
    """Tests for build_composition_filter — API query dict construction."""

    def test_nacl_filter(self) -> None:
        """NaCl should produce {'formula': 'NaCl'}."""
        result = build_composition_filter("NaCl")
        assert result == {"formula": "NaCl"}

    def test_complex_formula(self) -> None:
        """Multi-element formula should be preserved verbatim."""
        result = build_composition_filter("Ba2YCu3O7")
        assert result == {"formula": "Ba2YCu3O7"}

    def test_filter_has_exactly_one_key(self) -> None:
        """Filter dict should contain only the 'formula' key."""
        result = build_composition_filter("Fe2O3")
        assert list(result.keys()) == ["formula"]

    def test_empty_string_raises(self) -> None:
        """Empty formula string must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string") as exc_info:
            build_composition_filter("")
        assert "non-empty string" in str(exc_info.value)

    def test_none_raises(self) -> None:
        """None input must raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="non-empty string") as exc_info:
            build_composition_filter(None)  # type: ignore[arg-type]
        assert "non-empty string" in str(exc_info.value)

    def test_roundtrip_with_get_reduced_formula(self) -> None:
        """Filter built from a structure's formula should contain that formula."""
        formula = get_reduced_formula(_tio2_rutile())
        filt = build_composition_filter(formula)
        assert filt["formula"] == "TiO2"


# ---------------------------------------------------------------------------
# group_structures_by_composition tests
# ---------------------------------------------------------------------------


class TestGroupByComposition:
    """Tests for group_structures_by_composition — the batch grouping logic."""

    def test_single_composition_group(self) -> None:
        """All NaCl structures should land in one group."""
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        groups = group_structures_by_composition([s1, s2])

        assert len(groups) == 1
        assert "NaCl" in groups
        assert len(groups["NaCl"]) == 2

    def test_multiple_composition_groups(self) -> None:
        """Structures with different compositions should separate."""
        nacl = _nacl_rocksalt()
        fe = _fe_bcc()
        tio2 = _tio2_rutile()
        groups = group_structures_by_composition([nacl, fe, tio2])

        assert set(groups.keys()) == {"NaCl", "Fe", "TiO2"}
        assert len(groups["NaCl"]) == 1
        assert len(groups["Fe"]) == 1
        assert len(groups["TiO2"]) == 1

    def test_supercell_groups_with_primitive(self) -> None:
        """Na4Cl4 supercell must group with NaCl primitive cell.

        This tests the critical reduction property: cells of the same
        material but different sizes must end up in the same group.
        """
        primitive = _nacl_rocksalt()
        supercell = _na4cl4_supercell()
        groups = group_structures_by_composition([primitive, supercell])

        assert len(groups) == 1
        assert "NaCl" in groups
        assert len(groups["NaCl"]) == 2

    def test_empty_list_returns_empty_dict(self) -> None:
        """Grouping an empty list should return an empty dict."""
        groups = group_structures_by_composition([])
        assert groups == {}

    def test_order_preserved_within_groups(self) -> None:
        """Structures within a group should maintain input order.

        This matters for deterministic downstream processing.
        """
        s1 = _nacl_rocksalt()
        fe = _fe_bcc()
        s2 = _nacl_rocksalt()
        groups = group_structures_by_composition([s1, fe, s2])

        # s1 and s2 are the NaCl structures; s1 should come first
        assert groups["NaCl"][0] is s1
        assert groups["NaCl"][1] is s2

    def test_empty_structure_in_list_raises(self) -> None:
        """A structure with no sites in the list must raise ValueError."""
        good = _nacl_rocksalt()
        empty = Structure(Lattice.cubic(5.0), [], [])
        with pytest.raises(ValueError, match="index 1") as exc_info:
            group_structures_by_composition([good, empty])
        assert "index 1" in str(exc_info.value)

    def test_return_type_is_plain_dict(self) -> None:
        """Return value should be a plain dict, not defaultdict.

        Callers should get KeyError on missing keys, not silent empty lists.
        """
        groups = group_structures_by_composition([_fe_bcc()])
        assert type(groups) is dict
        with pytest.raises(KeyError):
            _ = groups["nonexistent"]
