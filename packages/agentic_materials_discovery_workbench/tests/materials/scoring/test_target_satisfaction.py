"""Tests for target satisfaction scoring."""

import pytest

from agentic_materials_discovery.scoring.target_satisfaction import (
    target_satisfaction_score,
)


# ---------------------------------------------------------------------------
# Empty constraints
# ---------------------------------------------------------------------------


def test_empty_constraints_returns_one():
    """Empty constraints dict means no restrictions — score is 1.0."""
    assert target_satisfaction_score({"x": 1}, {}) == 1.0


def test_empty_constraints_empty_candidate():
    """Both empty: no constraints to fail, score is 1.0."""
    assert target_satisfaction_score({}, {}) == 1.0


# ---------------------------------------------------------------------------
# Categorical (str) constraints
# ---------------------------------------------------------------------------


def test_categorical_exact_match():
    """Exact string match satisfies the constraint."""
    candidate = {"chemistry_scope": "oxide"}
    constraints = {"chemistry_scope": "oxide"}
    assert target_satisfaction_score(candidate, constraints) == 1.0


def test_categorical_mismatch():
    """Different string value fails the constraint."""
    candidate = {"chemistry_scope": "sulfide"}
    constraints = {"chemistry_scope": "oxide"}
    assert target_satisfaction_score(candidate, constraints) == 0.0


def test_categorical_missing_field():
    """Missing field in candidate counts as unsatisfied."""
    candidate = {}
    constraints = {"chemistry_scope": "oxide"}
    assert target_satisfaction_score(candidate, constraints) == 0.0


def test_categorical_case_sensitive():
    """String matching is case-sensitive — 'Oxide' != 'oxide'."""
    candidate = {"chemistry_scope": "Oxide"}
    constraints = {"chemistry_scope": "oxide"}
    assert target_satisfaction_score(candidate, constraints) == 0.0


# ---------------------------------------------------------------------------
# Numeric constraints (candidate value <= limit)
# ---------------------------------------------------------------------------


def test_numeric_within_limit():
    """Candidate value at or below limit satisfies the constraint."""
    candidate = {"structure_size_limit": 50}
    constraints = {"structure_size_limit": 100}
    assert target_satisfaction_score(candidate, constraints) == 1.0


def test_numeric_at_exact_limit():
    """Candidate value exactly equal to limit is satisfied."""
    candidate = {"max_atoms": 200}
    constraints = {"max_atoms": 200}
    assert target_satisfaction_score(candidate, constraints) == 1.0


def test_numeric_exceeds_limit():
    """Candidate value above limit fails the constraint."""
    candidate = {"structure_size_limit": 150}
    constraints = {"structure_size_limit": 100}
    assert target_satisfaction_score(candidate, constraints) == 0.0


def test_numeric_float_constraint():
    """Float constraint values work with <= comparison."""
    candidate = {"energy": 0.05}
    constraints = {"energy": 0.1}
    assert target_satisfaction_score(candidate, constraints) == 1.0


def test_numeric_missing_field():
    """Missing numeric field in candidate counts as unsatisfied."""
    candidate = {}
    constraints = {"max_atoms": 100}
    assert target_satisfaction_score(candidate, constraints) == 0.0


# ---------------------------------------------------------------------------
# Mixed constraints
# ---------------------------------------------------------------------------


def test_mixed_all_met():
    """All constraints met across categorical and numeric types."""
    candidate = {"chemistry_scope": "oxide", "max_atoms": 50, "energy": 0.03}
    constraints = {"chemistry_scope": "oxide", "max_atoms": 100, "energy": 0.1}
    assert target_satisfaction_score(candidate, constraints) == 1.0


def test_mixed_partial():
    """Some constraints met, some not — returns correct fraction."""
    candidate = {"chemistry_scope": "oxide", "max_atoms": 150}
    constraints = {"chemistry_scope": "oxide", "max_atoms": 100}
    assert target_satisfaction_score(candidate, constraints) == 0.5


def test_mixed_none_met():
    """No constraints met scores 0.0."""
    candidate = {"chemistry_scope": "sulfide", "max_atoms": 300}
    constraints = {"chemistry_scope": "oxide", "max_atoms": 100}
    assert target_satisfaction_score(candidate, constraints) == 0.0


def test_fraction_three_of_four():
    """Three of four constraints met gives 0.75."""
    candidate = {
        "scope": "oxide",
        "sg": 225,
        "atoms": 50,
        "density": 5.0,
    }
    constraints = {
        "scope": "oxide",
        "sg": 225,
        "atoms": 100,
        "density": 3.0,  # candidate 5.0 > limit 3.0 — fails
    }
    assert target_satisfaction_score(candidate, constraints) == 0.75


# ---------------------------------------------------------------------------
# Unsupported constraint types
# ---------------------------------------------------------------------------


def test_unsupported_constraint_type_raises():
    """List-typed constraint value raises TypeError with field name."""
    with pytest.raises(TypeError) as exc_info:
        target_satisfaction_score(
            {"tags": ["a"]},
            {"tags": ["a", "b"]},
        )
    assert "tags" in str(exc_info.value)
    assert "list" in str(exc_info.value)


def test_unsupported_constraint_type_dict():
    """Dict-typed constraint value raises TypeError with type name."""
    with pytest.raises(TypeError) as exc_info:
        target_satisfaction_score(
            {"nested": {"a": 1}},
            {"nested": {"a": 1}},
        )
    assert "nested" in str(exc_info.value)
    assert "dict" in str(exc_info.value)
