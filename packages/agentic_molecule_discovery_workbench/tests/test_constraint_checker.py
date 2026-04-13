"""Tests for molecular constraint checking.

Uses aspirin (CC(=O)Oc1ccccc1C(=O)O, MW ~180.16, cLogP ~1.24)
as the primary test molecule because its properties are well-known
and it contains aromatic rings (useful for SMARTS tests).
"""

from __future__ import annotations

import logging

import pytest
from rdkit import Chem

from agentic_molecule_discovery.constraints.constraint_checker import ConstraintChecker

# Aspirin SMILES — MW ~180.16, cLogP ~1.24, HBD 1, HBA 4
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_MOL = Chem.MolFromSmiles(ASPIRIN)


class TestMolecularWeightRange:
    """MW range checks — the bread-and-butter numeric constraint."""

    def test_mw_within_range_passes(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 100.0},
            {"type": "numeric", "property": "molecular_weight", "operator": "<=", "value": 300.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True
        assert all(r.passed for r in result.results)

    def test_mw_outside_range_fails(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 200.0},
            {"type": "numeric", "property": "molecular_weight", "operator": "<=", "value": 300.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        # The >= 200 constraint should fail (aspirin MW ~180)
        mw_lower = result.results[0]
        assert mw_lower.passed is False
        assert mw_lower.actual_value is not None


class TestClogpBounds:
    """One-sided cLogP constraints."""

    def test_clogp_upper_bound_only(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "clogp", "operator": "<=", "value": 5.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True

    def test_clogp_lower_bound_only(self) -> None:
        # Aspirin cLogP ~1.24, so lower bound of 3.0 should fail
        checker = ConstraintChecker([
            {"type": "numeric", "property": "clogp", "operator": ">=", "value": 3.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        assert result.results[0].reason is not None


class TestSmartsConstraints:
    """SMARTS substructure matching — required and forbidden patterns."""

    def test_smarts_required_present_passes(self) -> None:
        # Aspirin contains an aromatic ring
        checker = ConstraintChecker([
            {"type": "smarts", "pattern": "c1ccccc1", "mode": "required"},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True

    def test_smarts_required_absent_fails(self) -> None:
        # Aspirin does not contain a nitrogen
        checker = ConstraintChecker([
            {"type": "smarts", "pattern": "[#7]", "mode": "required"},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        assert result.results[0].reason is not None
        assert "not found" in result.results[0].reason

    def test_smarts_forbidden_present_fails(self) -> None:
        # Aspirin has an aromatic ring — forbidding it should fail
        checker = ConstraintChecker([
            {"type": "smarts", "pattern": "c1ccccc1", "mode": "forbidden"},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        assert result.results[0].reason is not None
        assert "present" in result.results[0].reason

    def test_smarts_forbidden_absent_passes(self) -> None:
        # Aspirin has no nitrogen — forbidding nitrogen should pass
        checker = ConstraintChecker([
            {"type": "smarts", "pattern": "[#7]", "mode": "forbidden"},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True

    def test_invalid_smarts_logs_warning_and_fails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        checker = ConstraintChecker([
            {"type": "smarts", "pattern": "[INVALID", "mode": "required"},
        ])
        with caplog.at_level(logging.WARNING):
            result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        assert result.results[0].passed is False
        assert result.results[0].reason is not None
        assert "Invalid SMARTS" in result.results[0].reason
        assert any("Invalid SMARTS" in rec.message for rec in caplog.records)


class TestAllSatisfiedDerivation:
    """ConstraintResult.all_satisfied must reflect the individual results."""

    def test_all_constraints_satisfied_sets_all_satisfied_true(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 100.0},
            {"type": "numeric", "property": "clogp", "operator": "<=", "value": 5.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True

    def test_one_constraint_failed_sets_all_satisfied_false(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 100.0},
            {"type": "numeric", "property": "clogp", "operator": ">=", "value": 5.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is False
        # Exactly one should fail
        passed_count = sum(1 for r in result.results if r.passed)
        assert passed_count == 1

    def test_empty_constraints_returns_all_satisfied(self) -> None:
        checker = ConstraintChecker([])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True
        assert result.results == []


class TestHbdBoundary:
    """HBD at exact boundary — exercises == behaviour."""

    def test_hbd_constraint_exact_boundary(self) -> None:
        # Aspirin has 1 HBD (carboxylic acid OH) — test at exact boundary
        checker = ConstraintChecker([
            {"type": "numeric", "property": "hbd", "operator": "<=", "value": 1},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True
        assert result.results[0].actual_value == 1


class TestCombinedConstraints:
    """Multiple numeric constraints in a single check."""

    def test_multiple_numeric_constraints_combined(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 100.0},
            {"type": "numeric", "property": "molecular_weight", "operator": "<=", "value": 300.0},
            {"type": "numeric", "property": "clogp", "operator": "<=", "value": 5.0},
            {"type": "numeric", "property": "hbd", "operator": "<=", "value": 5},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.all_satisfied is True
        assert len(result.results) == 4


class TestResultContents:
    """Verify that result objects carry useful actual values."""

    def test_result_contains_actual_values(self) -> None:
        checker = ConstraintChecker([
            {"type": "numeric", "property": "molecular_weight", "operator": ">=", "value": 100.0},
            {"type": "numeric", "property": "clogp", "operator": "<=", "value": 5.0},
        ])
        result = checker.check(ASPIRIN_MOL, ASPIRIN)
        assert result.smiles == ASPIRIN
        for r in result.results:
            assert r.actual_value is not None
            assert isinstance(r.actual_value, (int, float))
            assert r.constraint_name in ("molecular_weight", "clogp")


class TestNullMolGuard:
    """check() must reject None mol with a clear error."""

    def test_none_mol_raises_value_error(self) -> None:
        checker = ConstraintChecker([])
        with pytest.raises(ValueError, match="mol must not be None") as exc_info:
            checker.check(None, "invalid")
        assert "invalid" in str(exc_info.value)
