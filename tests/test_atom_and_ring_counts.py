"""Tests for molecular_properties.atom_and_ring_counts."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molecular_properties.atom_and_ring_counts import (
    calc_aromatic_ring_count,
    calc_heavy_atom_count,
    calc_ring_count,
)


class TestHeavyAtomCountKnown:
    """Verify heavy atom counts against known molecules."""

    def test_methane_has_one_heavy_atom(self):
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert calc_heavy_atom_count(mol) == 1

    def test_ethanol_has_three_heavy_atoms(self):
        """C, C, O — hydrogens excluded."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        assert calc_heavy_atom_count(mol) == 3

    def test_benzene_has_six_heavy_atoms(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_heavy_atom_count(mol) == 6

    def test_aspirin_has_thirteen_heavy_atoms(self):
        """CC(=O)Oc1ccccc1C(=O)O — 9C + 4O = 13 heavy atoms."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert calc_heavy_atom_count(mol) == 13


class TestHeavyAtomCountEdge:
    """Edge cases for heavy atom count."""

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_heavy_atom_count(None)

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_heavy_atom_count("C")

    def test_explicit_hydrogen_molecule(self):
        """[H][H] parsed with sanitization has 0 heavy atoms."""
        mol = Chem.MolFromSmiles("[H][H]")
        assert mol is not None
        assert calc_heavy_atom_count(mol) == 0

    def test_return_type_is_int(self):
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        assert isinstance(calc_heavy_atom_count(mol), int)


class TestRingCountKnown:
    """Verify total ring counts against known molecules."""

    def test_benzene_has_one_ring(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_ring_count(mol) == 1

    def test_naphthalene_has_two_rings(self):
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        assert mol is not None
        assert calc_ring_count(mol) == 2

    def test_cyclohexane_has_one_ring(self):
        mol = Chem.MolFromSmiles("C1CCCCC1")
        assert mol is not None
        assert calc_ring_count(mol) == 1

    def test_acyclic_molecule_has_zero_rings(self):
        """Ethanol — no rings."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        assert calc_ring_count(mol) == 0

    def test_spiro_compound_has_two_rings(self):
        """Spiro[4.4]nonane — two fused-at-one-atom rings."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        assert mol is not None
        assert calc_ring_count(mol) == 2


class TestRingCountEdge:
    """Edge cases for ring count."""

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_ring_count(None)

    def test_int_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_ring_count(42)

    def test_return_type_is_int(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert isinstance(calc_ring_count(mol), int)


class TestAromaticRingCountKnown:
    """Verify aromatic ring counts against known molecules."""

    def test_benzene_has_one_aromatic_ring(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 1

    def test_naphthalene_has_two_aromatic_rings(self):
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 2

    def test_cyclohexane_has_zero_aromatic_rings(self):
        """Saturated ring — not aromatic."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 0

    def test_acyclic_molecule_has_zero_aromatic_rings(self):
        mol = Chem.MolFromSmiles("CCCC")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 0

    def test_indole_has_two_aromatic_rings(self):
        """Indole — benzene ring fused with pyrrole, both aromatic."""
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 2

    def test_caffeine_has_two_aromatic_rings(self):
        """Caffeine's imidazole and pyrimidine rings are both aromatic."""
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert mol is not None
        assert calc_aromatic_ring_count(mol) == 2


class TestAromaticRingCountEdge:
    """Edge cases for aromatic ring count."""

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_aromatic_ring_count(None)

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_aromatic_ring_count("c1ccccc1")

    def test_aromatic_never_exceeds_total_ring_count(self):
        """Aromatic ring count must be <= total ring count for any molecule."""
        for smi in ["c1ccccc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CC1"]:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            assert calc_aromatic_ring_count(mol) <= calc_ring_count(mol)

    def test_return_type_is_int(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert isinstance(calc_aromatic_ring_count(mol), int)
