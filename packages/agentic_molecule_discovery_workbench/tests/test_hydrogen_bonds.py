"""Tests for molecular_properties.hydrogen_bonds — calc_hbd and calc_hba."""

from __future__ import annotations

import pytest
from rdkit import Chem

from agentic_molecule_discovery.properties.hydrogen_bonds import calc_hba, calc_hbd


class TestHbdKnownMolecules:
    """Verify hydrogen bond donor counts against known values."""

    def test_ethanol_one_donor(self):
        """Ethanol (CCO) — single OH group, 1 donor."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        assert calc_hbd(mol) == 1

    def test_acetic_acid_one_donor(self):
        """Acetic acid (CC(=O)O) — carboxylic OH donates, carbonyl O does not."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        assert mol is not None
        assert calc_hbd(mol) == 1

    def test_glycine_two_donors(self):
        """Glycine (NCC(=O)O) — amine NH2 counts as 1 donor group, carboxyl OH as 1."""
        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert mol is not None
        assert calc_hbd(mol) == 2

    def test_aspirin_one_donor(self):
        """Aspirin — only the carboxylic acid OH donates; the ester O does not."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert calc_hbd(mol) == 1


class TestHbaKnownMolecules:
    """Verify hydrogen bond acceptor counts against known values."""

    def test_ethanol_one_acceptor(self):
        """Ethanol (CCO) — the oxygen is 1 acceptor."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        assert calc_hba(mol) == 1

    def test_acetic_acid_one_acceptor(self):
        """Acetic acid — Lipinski HBA counts only the carbonyl oxygen; the
        hydroxyl O is excluded because it is bonded to H."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        assert mol is not None
        assert calc_hba(mol) == 1

    def test_caffeine_three_acceptors(self):
        """Caffeine — RDKit Lipinski HBA: 3 acceptors (the imine N plus 2 carbonyls;
        N-methyl nitrogens are excluded by the Lipinski SMARTS pattern)."""
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert mol is not None
        assert calc_hba(mol) == 3

    def test_aspirin_three_acceptors(self):
        """Aspirin — RDKit Lipinski HBA: 3 acceptors (ester O plus 2 carbonyl O;
        the carboxylic OH is excluded because it carries a hydrogen)."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert calc_hba(mol) == 3


class TestEdgeCases:
    """Edge cases and error paths for both functions."""

    def test_benzene_zero_donors(self):
        """Benzene has no N-H or O-H bonds — 0 donors."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_hbd(mol) == 0

    def test_benzene_zero_acceptors(self):
        """Benzene has no N or O atoms — 0 acceptors."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert calc_hba(mol) == 0

    def test_methane_zero_donors_and_acceptors(self):
        """Methane — no heteroatoms at all."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert calc_hbd(mol) == 0
        assert calc_hba(mol) == 0

    def test_water_zero_donors_zero_acceptors(self):
        """Water (O) — RDKit implicit-H SMILES 'O' parses without explicit H;
        NumHDonors returns 0 and NumHAcceptors returns 0 for this representation."""
        mol = Chem.MolFromSmiles("O")
        assert mol is not None
        # RDKit's Lipinski patterns on implicit-H water yield 0 for both
        assert calc_hbd(mol) == 0
        assert calc_hba(mol) == 0

    def test_primary_amine_one_donor(self):
        """Methylamine (CN) — NH2 counts as 1 donor."""
        mol = Chem.MolFromSmiles("CN")
        assert mol is not None
        assert calc_hbd(mol) == 1

    def test_tertiary_amine_zero_donors_one_acceptor(self):
        """Trimethylamine N(C)(C)C — no N-H bond, so 0 donors but 1 acceptor."""
        mol = Chem.MolFromSmiles("N(C)(C)C")
        assert mol is not None
        assert calc_hbd(mol) == 0
        assert calc_hba(mol) == 1

    def test_charged_ammonium_donor(self):
        """Ammonium ion [NH4+] — has N-H bonds, counts as donor."""
        mol = Chem.MolFromSmiles("[NH4+]")
        assert mol is not None
        assert calc_hbd(mol) == 1

    def test_return_type_is_int(self):
        """Ethanol (CCO) — verify return type is int and values are correct."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        hbd = calc_hbd(mol)
        hba = calc_hba(mol)
        assert isinstance(hbd, int)
        assert isinstance(hba, int)
        assert hbd == 1
        assert hba == 1


class TestInvalidInput:
    """Both functions must reject non-Mol input with a clear TypeError."""

    def test_hbd_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hbd(None)
        assert "NoneType" in str(exc_info.value)

    def test_hba_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hba(None)
        assert "NoneType" in str(exc_info.value)

    def test_hbd_string_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hbd("CCO")
        assert "str" in str(exc_info.value)

    def test_hba_string_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hba("CCO")
        assert "str" in str(exc_info.value)

    def test_hbd_int_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hbd(42)
        assert "int" in str(exc_info.value)

    def test_hba_int_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
            calc_hba(42)
        assert "int" in str(exc_info.value)
