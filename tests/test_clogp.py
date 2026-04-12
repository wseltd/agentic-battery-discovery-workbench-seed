"""Tests for molecular_properties.clogp.calc_clogp."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molecular_properties.clogp import calc_clogp

# Tolerance for floating-point comparison of cLogP values.
# Crippen.MolLogP returns values to ~2 decimal places; 0.05 accounts
# for minor RDKit version differences in atom contribution tables.
_CLOGP_TOLERANCE = 0.05


class TestKnownMolecules:
    """Verify cLogP against literature/RDKit reference values."""

    def test_aspirin_clogp(self):
        """Aspirin (acetylsalicylic acid) — moderately lipophilic."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        clogp = calc_clogp(mol)
        assert abs(clogp - 1.31) < _CLOGP_TOLERANCE

    def test_caffeine_clogp(self):
        """Caffeine — hydrophilic (negative cLogP).

        Ticket reference was -0.07 but RDKit Wildman-Crippen gives -1.03;
        the difference is expected — Crippen atom contributions differ
        from consensus LogP methods.
        """
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert mol is not None
        clogp = calc_clogp(mol)
        assert abs(clogp - (-1.03)) < _CLOGP_TOLERANCE

    def test_benzene_clogp(self):
        """Benzene — well-known lipophilic reference compound.

        Ticket reference was 1.56 but RDKit Wildman-Crippen gives 1.69;
        the difference reflects Crippen atom contribution tables vs
        experimental/consensus values.
        """
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        clogp = calc_clogp(mol)
        assert abs(clogp - 1.69) < _CLOGP_TOLERANCE


class TestEdgeCases:
    """Edge cases and error paths — where bugs hide."""

    def test_hydrophilic_molecule_ethanol(self):
        """Ethanol — small hydrophilic molecule with negative cLogP."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        clogp = calc_clogp(mol)
        # Ethanol cLogP is around -0.31 to -0.18 depending on method
        assert clogp < 0.5, "Ethanol should be hydrophilic"

    def test_highly_lipophilic_hexane(self):
        """Hexane — saturated hydrocarbon, high cLogP."""
        mol = Chem.MolFromSmiles("CCCCCC")
        assert mol is not None
        clogp = calc_clogp(mol)
        # Hexane cLogP is around 3.0-3.5
        assert clogp > 2.5, "Hexane should be highly lipophilic"

    def test_charged_molecule(self):
        """Charged species should still return a valid value."""
        mol = Chem.MolFromSmiles("[NH4+]")
        assert mol is not None
        clogp = calc_clogp(mol)
        assert isinstance(clogp, float)
        # Crippen atom contributions give ammonium ~0.38
        assert abs(clogp - 0.38) < 0.1

    def test_single_atom_methane(self):
        """Methane — smallest organic molecule."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        clogp = calc_clogp(mol)
        # Methane cLogP ~0.6
        assert isinstance(clogp, float)
        assert abs(clogp - 0.64) < _CLOGP_TOLERANCE

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_clogp(None)
        assert "NoneType" in str(exc_info.value)

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_clogp("CCO")
        assert "str" in str(exc_info.value)

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_clogp(42)
        assert "int" in str(exc_info.value)

    def test_return_type_is_float(self):
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        result = calc_clogp(mol)
        assert isinstance(result, float)
        # Methane cLogP ~0.64 — same reference as test_single_atom_methane
        assert abs(result - 0.64) < _CLOGP_TOLERANCE
