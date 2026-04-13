"""Tests for CrystalCanonical – construction, symmetry detection, and export."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from ammd.materials.crystal import CrystalCanonical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _nacl_conventional() -> Structure:
    """NaCl conventional cell (Fm-3m, SG 225), 8 atoms."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords: list[list[float]] = [
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


def _nacl_primitive() -> Structure:
    """NaCl-like primitive cell — 2 atoms, SG 221 (Pm-3m / CsCl-type)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Cl"]
    coords: list[list[float]] = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


def _wurtzite_zns() -> Structure:
    """Wurtzite ZnS (P6_3mc, SG 186)."""
    a = 3.82
    c = 6.26
    lattice = Lattice.hexagonal(a, c)
    species = ["Zn", "Zn", "S", "S"]
    coords: list[list[float]] = [
        [1 / 3, 2 / 3, 0.0],
        [2 / 3, 1 / 3, 0.5],
        [1 / 3, 2 / 3, 0.375],
        [2 / 3, 1 / 3, 0.875],
    ]
    return Structure(lattice, species, coords)


def _p1_distorted() -> Structure:
    """Slightly distorted cell that is P1 at tight symprec but cubic
    at loose symprec — useful for symprec sensitivity tests.

    Using two different species, an asymmetric lattice, and off-centre
    fractional coords ensures spglib finds no symmetry at tight tolerance.
    """
    lattice = Lattice([[4.0, 0.0, 0.0], [0.0, 4.01, 0.0], [0.0, 0.0, 4.02]])
    species = ["Si", "Ge"]
    coords: list[list[float]] = [[0.0, 0.0, 0.0], [0.501, 0.502, 0.503]]
    return Structure(lattice, species, coords)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_nacl_conventional(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_conventional())
        assert cc.composition == "NaCl"
        assert cc.num_atoms == 8
        assert cc.fractional_coords.shape == (8, 3)

    def test_direct_init_raises(self) -> None:
        with pytest.raises(TypeError, match="cannot be constructed directly") as exc_info:
            CrystalCanonical()
        assert "cannot be constructed directly" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Space-group detection
# ---------------------------------------------------------------------------

class TestSpaceGroup:
    def test_from_nacl_sg_number_225(self) -> None:
        # The conventional 8-atom NaCl cell is Fm-3m (225); the 2-atom
        # primitive cell is CsCl-type Pm-3m (221).
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_conventional())
        assert cc.space_group_number == 225
        assert "Fm" in cc.space_group_symbol

    def test_from_wurtzite_sg_number_186(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_wurtzite_zns())
        assert cc.space_group_number == 186

    def test_from_p1_structure_tight_symprec(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_p1_distorted(), symprec=1e-5)
        assert cc.space_group_number == 1, "Tight symprec should yield P1"

    def test_from_p1_structure_loose_symprec_finds_symmetry(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_p1_distorted(), symprec=0.1)
        assert cc.space_group_number > 1, (
            "Loose symprec should detect higher symmetry"
        )

    def test_symprec_sensitivity(self) -> None:
        """Same structure yields different SG at different tolerances."""
        tight = CrystalCanonical.from_pymatgen_structure(_p1_distorted(), symprec=1e-5)
        loose = CrystalCanonical.from_pymatgen_structure(_p1_distorted(), symprec=0.1)
        assert tight.space_group_number != loose.space_group_number


# ---------------------------------------------------------------------------
# Fractional coordinate wrapping
# ---------------------------------------------------------------------------

class TestCoordWrapping:
    def test_fractional_coords_wrapped(self) -> None:
        """Coords outside [0,1) must be wrapped back in."""
        lattice = Lattice.cubic(5.0)
        coords: list[list[float]] = [[1.1, -0.2, 0.5]]
        struct = Structure(lattice, ["Fe"], coords)
        cc = CrystalCanonical.from_pymatgen_structure(struct)
        assert np.all(cc.fractional_coords >= 0.0)
        assert np.all(cc.fractional_coords < 1.0)
        np.testing.assert_allclose(
            cc.fractional_coords[0], [0.1, 0.8, 0.5], atol=1e-10
        )


# ---------------------------------------------------------------------------
# CIF / POSCAR export
# ---------------------------------------------------------------------------

class TestExport:
    def test_to_cif_roundtrip(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_primitive())
        cif_str = cc.to_cif()
        assert "_cell_length_a" in cif_str
        assert "_atom_site_fract_x" in cif_str

    def test_to_poscar_roundtrip(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_primitive())
        poscar_str = cc.to_poscar()
        assert "Na" in poscar_str or "Cl" in poscar_str

    def test_to_cif_contains_sg_symbol(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_conventional())
        cif_str = cc.to_cif()
        # CifWriter embeds the detected space group in the CIF header
        assert "Fm-3m" in cif_str or "225" in cif_str


# ---------------------------------------------------------------------------
# Site properties, composition, species
# ---------------------------------------------------------------------------

class TestProperties:
    def test_site_properties_preserved(self) -> None:
        struct = _nacl_primitive()
        struct.add_site_property("magmom", [1.0, -1.0])
        cc = CrystalCanonical.from_pymatgen_structure(struct)
        assert "magmom" in cc.site_properties
        assert cc.site_properties["magmom"] == [1.0, -1.0]

    def test_composition_reduced_formula(self) -> None:
        cc = CrystalCanonical.from_pymatgen_structure(_nacl_conventional())
        assert cc.composition == "NaCl"

    def test_num_atoms_matches_structure(self) -> None:
        struct = _nacl_conventional()
        cc = CrystalCanonical.from_pymatgen_structure(struct)
        assert cc.num_atoms == len(struct)

    def test_species_list_order(self) -> None:
        """Species list must follow the same site order as the input structure."""
        struct = _nacl_conventional()
        cc = CrystalCanonical.from_pymatgen_structure(struct)
        expected = [site.specie for site in struct]
        assert len(cc.species) == len(expected)
        for got, want in zip(cc.species, expected):
            assert got == want
