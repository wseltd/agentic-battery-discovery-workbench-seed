"""Tests for molecular_properties.molecular_weight.calc_molecular_weight."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molecular_properties.molecular_weight import calc_molecular_weight

# Tolerance for floating-point comparison of molecular weights.
# ExactMolWt returns values to ~3 decimal places of precision.
_MW_TOLERANCE = 0.001


class TestKnownMolecules:
    """Verify exact molecular weight against known values."""

    def test_water_weight(self):
        mol = Chem.MolFromSmiles("O")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        assert abs(weight - 18.011) < _MW_TOLERANCE

    def test_aspirin_weight(self):
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        assert abs(weight - 180.042) < _MW_TOLERANCE

    def test_caffeine_weight(self):
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        assert abs(weight - 194.080) < _MW_TOLERANCE


class TestEdgeCases:
    """Edge cases and error paths — where bugs are most likely."""

    def test_single_atom_hydrogen(self):
        """Explicit hydrogen atom — smallest possible molecule."""
        mol = Chem.MolFromSmiles("[H][H]")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        # H2 = 2 * 1.00794 ≈ 2.016
        assert abs(weight - 2.016) < _MW_TOLERANCE

    def test_heavy_molecule_cholesterol(self):
        """Larger molecule to verify no truncation or overflow issues."""
        mol = Chem.MolFromSmiles("C([C@@H]1CC2[C@]([C@@H]1O)([C@H]1CC[C@@H]([C@@]1(CC2)C)C=CC=CC(C)C)C)O")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        # Should be a reasonable weight for a sterol (~400 Da range)
        assert weight > 300.0
        assert weight < 500.0

    def test_charged_molecule(self):
        """Charged species should still return a valid weight."""
        mol = Chem.MolFromSmiles("[NH4+]")
        assert mol is not None
        weight = calc_molecular_weight(mol)
        # NH4+ has same atomic composition as NH4, MW ≈ 18.034
        assert weight > 0.0

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_molecular_weight(None)

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_molecular_weight("CCO")

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol"):
            calc_molecular_weight(42)

    def test_weight_is_positive(self):
        """Any valid molecule must have positive weight."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert calc_molecular_weight(mol) > 0.0

    def test_return_type_is_float(self):
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert isinstance(calc_molecular_weight(mol), float)
