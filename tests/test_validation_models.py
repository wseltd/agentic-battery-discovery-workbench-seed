"""Tests for validation.models — ValidationError and ValidationResult."""

import dataclasses

import pytest
from rdkit import Chem

from validation.models import ValidationError, ValidationResult


# ---------------------------------------------------------------------------
# ValidationError — construction, immutability, field access
# ---------------------------------------------------------------------------


class TestValidationErrorConstruction:
    """Basic construction and field access."""

    def test_fields_stored_correctly(self):
        err = ValidationError(error_type="syntax", message="bad input", smiles="XYZ")
        assert err.error_type == "syntax"
        assert err.message == "bad input"
        assert err.smiles == "XYZ"

    def test_empty_strings_allowed(self):
        err = ValidationError(error_type="", message="", smiles="")
        assert err.error_type == ""
        assert err.message == ""
        assert err.smiles == ""


class TestValidationErrorFrozen:
    """Frozen semantics — fields must not be reassignable."""

    def test_cannot_set_error_type(self):
        err = ValidationError(error_type="valence", message="m", smiles="C")
        with pytest.raises(dataclasses.FrozenInstanceError):
            err.error_type = "syntax"

    def test_cannot_set_message(self):
        err = ValidationError(error_type="valence", message="m", smiles="C")
        with pytest.raises(dataclasses.FrozenInstanceError):
            err.message = "new"

    def test_cannot_set_smiles(self):
        err = ValidationError(error_type="valence", message="m", smiles="C")
        with pytest.raises(dataclasses.FrozenInstanceError):
            err.smiles = "O"


class TestValidationErrorEquality:
    """Frozen dataclass equality and hashing."""

    def test_equal_instances(self):
        a = ValidationError(error_type="syntax", message="m", smiles="C")
        b = ValidationError(error_type="syntax", message="m", smiles="C")
        assert a == b

    def test_different_instances(self):
        a = ValidationError(error_type="syntax", message="m", smiles="C")
        b = ValidationError(error_type="valence", message="m", smiles="C")
        assert a != b

    def test_hashable(self):
        err = ValidationError(error_type="syntax", message="m", smiles="C")
        # Frozen dataclasses are hashable — usable in sets and as dict keys.
        assert {err} == {err}


# ---------------------------------------------------------------------------
# ValidationResult — construction, field access, immutability, edge cases
# ---------------------------------------------------------------------------


class TestValidationResultValid:
    """Valid-molecule construction path."""

    def test_valid_result_with_mol(self):
        mol = Chem.MolFromSmiles("CCO")
        result = ValidationResult(is_valid=True, mol=mol, errors=[])
        assert result.is_valid is True
        assert result.mol is mol
        assert result.errors == []

    def test_valid_result_smiles_round_trip(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = ValidationResult(is_valid=True, mol=mol, errors=[])
        assert Chem.MolToSmiles(result.mol) == "c1ccccc1"


class TestValidationResultInvalid:
    """Invalid-molecule construction path."""

    def test_invalid_result_none_mol(self):
        err = ValidationError(error_type="syntax", message="unparseable", smiles="XYZ")
        result = ValidationResult(is_valid=False, mol=None, errors=[err])
        assert result.is_valid is False
        assert result.mol is None
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "syntax"

    def test_multiple_errors(self):
        e1 = ValidationError(error_type="syntax", message="a", smiles="X")
        e2 = ValidationError(error_type="valence", message="b", smiles="X")
        result = ValidationResult(is_valid=False, mol=None, errors=[e1, e2])
        assert len(result.errors) == 2
        assert result.errors[0] is e1
        assert result.errors[1] is e2


class TestValidationResultFrozen:
    """Frozen semantics — top-level fields must not be reassignable."""

    def test_cannot_set_is_valid(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.is_valid = False

    def test_cannot_set_mol(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.mol = Chem.MolFromSmiles("C")
        assert result.mol is None

    def test_cannot_set_errors(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.errors = [ValidationError("x", "y", "z")]


class TestValidationResultEdgeCases:
    """Boundary conditions and unusual but valid inputs."""

    def test_invalid_result_with_mol_present(self):
        """A partially-valid molecule may still carry errors (e.g. valence issues
        detected after parsing succeeds)."""
        mol = Chem.MolFromSmiles("C")
        err = ValidationError(error_type="valence", message="bad valence", smiles="C")
        result = ValidationResult(is_valid=False, mol=mol, errors=[err])
        assert result.is_valid is False
        assert result.mol is not None
        assert len(result.errors) == 1

    def test_valid_result_with_empty_errors_list(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        assert result.errors == []

    def test_error_type_preserves_arbitrary_strings(self):
        """error_type is a free-form string — no enum restriction."""
        err = ValidationError(
            error_type="custom_check_42", message="detail", smiles="[Cu]"
        )
        assert err.error_type == "custom_check_42"

    def test_unicode_in_message(self):
        err = ValidationError(
            error_type="syntax", message="invalid bond → here", smiles="C→C"
        )
        assert "→" in err.message
        assert "→" in err.smiles

    def test_long_smiles_string(self):
        long_smiles = "C" * 10_000
        err = ValidationError(error_type="syntax", message="too long", smiles=long_smiles)
        assert len(err.smiles) == 10_000
