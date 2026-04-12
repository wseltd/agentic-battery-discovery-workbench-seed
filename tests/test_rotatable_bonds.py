"""Tests for molecular_properties.rotatable_bonds.calc_rotatable_bonds."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molecular_properties.rotatable_bonds import calc_rotatable_bonds


class TestKnownMolecules:
    """Verify rotatable bond counts against known values."""

    def test_benzene_rigid_zero_rotatable(self):
        """Benzene — fully rigid aromatic ring, zero rotatable bonds."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 0

    def test_butane_flexible(self):
        """Butane (CCCC) — one rotatable C-C bond in the middle."""
        mol = Chem.MolFromSmiles("CCCC")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 1

    def test_hexane_flexible(self):
        """Hexane (CCCCCC) — three rotatable bonds in a flexible chain."""
        mol = Chem.MolFromSmiles("CCCCCC")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 3

    def test_aspirin(self):
        """Aspirin — ester O-C(ring) and methyl C-C(=O) are rotatable."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 2

    def test_ethane_no_rotatable(self):
        """Ethane (CC) — terminal single bond is not rotatable."""
        mol = Chem.MolFromSmiles("CC")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 0


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_methane_zero(self):
        """Methane — single atom, no bonds at all."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 0

    def test_cyclohexane_ring_bonds_not_rotatable(self):
        """Cyclohexane — ring bonds are not counted as rotatable."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        assert mol is not None
        assert calc_rotatable_bonds(mol) == 0

    def test_return_type_is_int(self):
        """Return value must be an integer, not float."""
        mol = Chem.MolFromSmiles("CCCC")
        assert mol is not None
        result = calc_rotatable_bonds(mol)
        assert isinstance(result, int)
        assert result == 1

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_rotatable_bonds(None)
        assert "NoneType" in str(exc_info.value)

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_rotatable_bonds("CCCC")
        assert "str" in str(exc_info.value)

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_rotatable_bonds(42)
        assert "int" in str(exc_info.value)

    def test_amide_bond_not_rotatable(self):
        """Amide bonds have partial double-bond character — RDKit excludes them."""
        # N-methylacetamide: CH3-CO-NH-CH3
        mol = Chem.MolFromSmiles("CC(=O)NC")
        assert mol is not None
        # Only the C-C and N-C single bonds outside the amide are candidates;
        # the amide C-N itself is not rotatable in the Lipinski definition.
        assert calc_rotatable_bonds(mol) == 0
