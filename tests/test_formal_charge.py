"""Tests for molecular_validity.formal_charge.check_formal_charge."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molecular_validity.formal_charge import check_formal_charge


# --- helpers -----------------------------------------------------------------

def _mol_with_charge(smiles: str) -> Chem.Mol:
    """Parse SMILES and return the molecule. Fails the test if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    return mol


# --- passing cases -----------------------------------------------------------

class TestNeutralMolecule:
    """A fully neutral molecule should always pass the default range."""

    def test_neutral_passes(self):
        mol = _mol_with_charge("CCO")  # ethanol — all atoms neutral
        result = check_formal_charge(mol)
        assert result.is_valid is True
        assert result.errors == []
        assert result.mol is mol


class TestSingleChargeWithinDefault:
    """Molecules with +1 or -1 formal charges pass the default [-1, +1] range."""

    def test_positive_one_passes(self):
        # Ammonium ion: nitrogen carries +1
        mol = _mol_with_charge("[NH4+]")
        result = check_formal_charge(mol)
        assert result.is_valid is True
        assert result.errors == []

    def test_negative_one_passes(self):
        # Chloride ion: Cl carries -1
        mol = _mol_with_charge("[Cl-]")
        result = check_formal_charge(mol)
        assert result.is_valid is True
        assert result.errors == []


# --- failing cases -----------------------------------------------------------

class TestPositiveTwoFails:
    """An atom with +2 formal charge must fail the default range."""

    def test_plus_two_rejected(self):
        # Mg2+ has formal charge +2
        mol = _mol_with_charge("[Mg+2]")
        result = check_formal_charge(mol)
        assert result.is_valid is False
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.error_type == "formal_charge"
        assert "Mg" in err.message
        assert "+2" not in err.message or "2" in err.message  # charge value present
        assert "0" in str(err)  # atom index 0


class TestNegativeTwoFails:
    """An atom with -2 formal charge must fail the default range."""

    def test_minus_two_rejected(self):
        # Sulfate-like oxygen with -2
        mol = _mol_with_charge("[O-2]")
        result = check_formal_charge(mol)
        assert result.is_valid is False
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.error_type == "formal_charge"
        assert "O" in err.message
        assert "-2" in err.message


# --- custom range override ---------------------------------------------------

class TestCustomRangeOverride:
    """Callers can widen or narrow the acceptable charge range."""

    def test_wider_range_accepts_plus_two(self):
        mol = _mol_with_charge("[Mg+2]")
        result = check_formal_charge(mol, min_charge=-2, max_charge=2)
        assert result.is_valid is True
        assert result.errors == []

    def test_narrower_range_rejects_plus_one(self):
        mol = _mol_with_charge("[NH4+]")
        result = check_formal_charge(mol, min_charge=0, max_charge=0)
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_asymmetric_range(self):
        # Allow -2 but not +2
        mol = _mol_with_charge("[O-2]")
        result = check_formal_charge(mol, min_charge=-2, max_charge=1)
        assert result.is_valid is True


# --- edge cases and boundary validation --------------------------------------

class TestBoundaryValidation:
    """Invalid range parameters must raise immediately."""

    def test_min_greater_than_max_raises(self):
        mol = _mol_with_charge("C")
        with pytest.raises(ValueError, match="min_charge.*must be <= max_charge"):
            check_formal_charge(mol, min_charge=2, max_charge=-2)

    def test_equal_min_max_allows_exact_charge(self):
        # Only charge=0 allowed
        mol = _mol_with_charge("C")
        result = check_formal_charge(mol, min_charge=0, max_charge=0)
        assert result.is_valid is True

    def test_equal_min_max_rejects_other_charge(self):
        mol = _mol_with_charge("[NH4+]")
        result = check_formal_charge(mol, min_charge=0, max_charge=0)
        assert result.is_valid is False


class TestMultipleViolations:
    """When multiple atoms violate, each gets its own error."""

    def test_two_violating_atoms_produce_two_errors(self):
        # Build a molecule with two atoms carrying charges outside default range
        # Manually set charges via RWMol for precise control
        mol = Chem.RWMol()
        idx_a = mol.AddAtom(Chem.Atom(7))  # nitrogen
        idx_b = mol.AddAtom(Chem.Atom(8))  # oxygen
        mol.AddBond(idx_a, idx_b, Chem.BondType.SINGLE)
        mol.GetAtomWithIdx(idx_a).SetFormalCharge(3)
        mol.GetAtomWithIdx(idx_b).SetFormalCharge(-3)
        frozen = mol.GetMol()

        result = check_formal_charge(frozen)
        assert result.is_valid is False
        assert len(result.errors) == 2
        # Both atom indices reported
        messages = " ".join(e.message for e in result.errors)
        assert "N" in messages
        assert "O" in messages
