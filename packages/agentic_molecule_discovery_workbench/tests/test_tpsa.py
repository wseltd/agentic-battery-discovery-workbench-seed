"""Tests for molecular_properties.tpsa.calc_tpsa."""

from __future__ import annotations

import pytest
from rdkit import Chem

from agentic_molecule_discovery.properties.tpsa import calc_tpsa

# Tolerance for floating-point comparison of TPSA values.
# Descriptors.TPSA returns values to ~2 decimal places; 0.1 accounts
# for minor RDKit version differences in fragment contribution tables.
_TPSA_TOLERANCE = 0.1


class TestKnownMolecules:
    """Verify TPSA against known RDKit reference values."""

    def test_aspirin_tpsa(self):
        """Aspirin — three oxygen-containing groups contribute ~63.6 Å²."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        tpsa = calc_tpsa(mol)
        assert abs(tpsa - 63.60) < _TPSA_TOLERANCE

    def test_caffeine_tpsa(self):
        """Caffeine — nitrogen and oxygen heteroatoms.

        Ticket reference was 58.4 but RDKit Ertl TPSA gives 61.82;
        the difference reflects Ertl fragment contributions vs
        other TPSA calculation methods.
        """
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert mol is not None
        tpsa = calc_tpsa(mol)
        assert abs(tpsa - 61.82) < _TPSA_TOLERANCE

    def test_ethanol_tpsa(self):
        """Ethanol — single hydroxyl group contributes ~20.2 Å²."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        tpsa = calc_tpsa(mol)
        assert abs(tpsa - 20.23) < _TPSA_TOLERANCE


class TestEdgeCases:
    """Edge cases and error paths — where bugs hide."""

    def test_nonpolar_molecule_returns_zero(self):
        """Benzene has no polar atoms, so TPSA must be exactly 0."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        tpsa = calc_tpsa(mol)
        assert tpsa == 0.0

    def test_methane_returns_zero(self):
        """Methane — no polar atoms, TPSA must be 0."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert calc_tpsa(mol) == 0.0

    def test_charged_molecule(self):
        """Charged species should still return a valid TPSA."""
        mol = Chem.MolFromSmiles("[NH4+]")
        assert mol is not None
        tpsa = calc_tpsa(mol)
        # Ammonium nitrogen contributes ~36.5 Å²
        assert abs(tpsa - 36.50) < _TPSA_TOLERANCE

    def test_tpsa_is_non_negative(self):
        """TPSA is a surface area — must never be negative."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        assert mol is not None
        assert calc_tpsa(mol) >= 0.0

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_tpsa(None)
        assert "NoneType" in str(exc_info.value)

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_tpsa("CCO")
        assert "str" in str(exc_info.value)

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_tpsa(42)
        assert "int" in str(exc_info.value)

    def test_return_type_is_float(self):
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        result = calc_tpsa(mol)
        assert isinstance(result, float)
        # Ethanol ~20.23 — same reference as test_ethanol_tpsa
        assert abs(result - 20.23) < _TPSA_TOLERANCE
