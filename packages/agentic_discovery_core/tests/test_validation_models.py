"""Tests for validation.models -- ValidationError and ValidationResult.

Domain-agnostic tests: they operate on generic object values (no rdkit required).
"""

import dataclasses

import pytest

from agentic_discovery_core.validation.models import ValidationError, ValidationResult


class TestValidationErrorConstruction:
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
        assert {err} == {err}


class TestValidationResultValid:
    def test_valid_result_without_mol(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        assert result.is_valid is True
        assert result.mol is None
        assert result.errors == []

    def test_valid_result_with_arbitrary_mol(self):
        """mol can be any object -- the core does not require rdkit."""
        sentinel = object()
        result = ValidationResult(is_valid=True, mol=sentinel, errors=[])
        assert result.mol is sentinel


class TestValidationResultInvalid:
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
    def test_cannot_set_is_valid(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.is_valid = False

    def test_cannot_set_mol(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.mol = object()

    def test_cannot_set_errors(self):
        result = ValidationResult(is_valid=True, mol=None, errors=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.errors = [ValidationError("x", "y", "z")]


class TestValidationResultEdgeCases:
    def test_error_type_preserves_arbitrary_strings(self):
        err = ValidationError(
            error_type="custom_check_42", message="detail", smiles="[Cu]"
        )
        assert err.error_type == "custom_check_42"

    def test_unicode_in_message(self):
        err = ValidationError(
            error_type="syntax", message="invalid bond \u2192 here", smiles="C\u2192C"
        )
        assert "\u2192" in err.message
        assert "\u2192" in err.smiles

    def test_long_smiles_string(self):
        long_smiles = "C" * 10_000
        err = ValidationError(error_type="syntax", message="too long", smiles=long_smiles)
        assert len(err.smiles) == 10_000
