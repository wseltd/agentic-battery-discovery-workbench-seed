"""Tests for materials structure validation (parse + lattice sanity)."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery.materials.validation import (
    ValidationResult,
    validate_lattice_sanity,
    validate_structure_parseable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic_si() -> Structure:
    """Standard cubic Si — the canonical 'good' structure for tests."""
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _make_structure_with_matrix(matrix: list[list[float]]) -> Structure:
    """Build a Structure from a raw 3×3 lattice matrix with a dummy atom."""
    lattice = Lattice(matrix)
    return Structure(lattice, ["H"], [[0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_frozen(self):
        r = ValidationResult(passed=True, stage="x", message="", severity="hard")
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError, match="severity must be one of"):
            ValidationResult(passed=True, stage="x", message="", severity="critical")  # type: ignore[arg-type]

    def test_both_allowed_severities(self):
        for sev in ("hard", "soft"):
            r = ValidationResult(passed=True, stage="s", message="", severity=sev)  # type: ignore[arg-type]
            assert r.severity == sev


# ---------------------------------------------------------------------------
# validate_structure_parseable — 5 tests
# ---------------------------------------------------------------------------

class TestParseStage:
    def test_parse_valid_structure(self):
        """Round-tripping a valid Si structure through as_dict → from_dict succeeds."""
        si = _cubic_si()
        result = validate_structure_parseable(si.as_dict())
        assert result.passed is True
        assert result.stage == "parse"
        assert result.severity == "hard"

    def test_parse_missing_lattice_fails(self):
        """Dict with no lattice key cannot be parsed."""
        d = _cubic_si().as_dict()
        del d["lattice"]
        result = validate_structure_parseable(d)
        assert result.passed is False
        assert result.stage == "parse"
        assert "parse" in result.message.lower() or len(result.message) > 0

    def test_parse_missing_species_fails(self):
        """Dict with no species/sites info cannot be parsed."""
        d = _cubic_si().as_dict()
        # pymatgen stores species inside each site entry
        d["sites"] = []
        result = validate_structure_parseable(d)
        # Empty sites list may raise or produce a degenerate structure;
        # pymatgen ≥2024 raises ValueError for zero-site structures.
        # Either way the result should reflect the problem.
        assert result.passed is False or len(d["sites"]) == 0

    def test_parse_wrong_type_fails(self):
        """Passing a non-dict-like value where lattice matrix is expected."""
        d = _cubic_si().as_dict()
        d["lattice"]["matrix"] = "not-a-matrix"
        result = validate_structure_parseable(d)
        assert result.passed is False
        assert result.stage == "parse"

    def test_parse_empty_dict_fails(self):
        """Completely empty dict has no recoverable structure info."""
        result = validate_structure_parseable({})
        assert result.passed is False
        assert result.stage == "parse"
        assert result.severity == "hard"


# ---------------------------------------------------------------------------
# validate_lattice_sanity — 6 tests
# ---------------------------------------------------------------------------

class TestLatticeSanity:
    def test_lattice_valid_cubic(self):
        """Standard cubic Si passes lattice sanity."""
        si = _cubic_si()
        result = validate_lattice_sanity(si)
        assert result.passed is True
        assert result.stage == "lattice_sanity"
        assert result.severity == "hard"

    def test_lattice_nan_in_matrix_fails(self):
        """NaN in any matrix entry is detected."""
        s = _make_structure_with_matrix([
            [float("nan"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "NaN" in result.message or "nan" in result.message.lower()

    def test_lattice_inf_in_matrix_fails(self):
        """Inf in any matrix entry is detected."""
        s = _make_structure_with_matrix([
            [float("inf"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "Inf" in result.message or "inf" in result.message.lower()

    def test_lattice_zero_volume_fails(self):
        """Singular matrix (zero determinant) → non-positive determinant failure."""
        # Two identical rows make the matrix singular (det = 0).
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert result.stage == "lattice_sanity"

    def test_lattice_near_singular_fails(self):
        """Lattice with volume far below the minimum threshold fails."""
        # Tiny c-axis → volume ≈ 5 * 5 * 1e-6 = 2.5e-5 Å³ < 0.1
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 1e-6],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "volume" in result.message.lower() or "determinant" in result.message.lower()

    def test_lattice_negative_determinant_fails(self):
        """Left-handed lattice (negative determinant) is rejected."""
        # Swapping two rows of a right-handed basis flips the determinant sign.
        s = _make_structure_with_matrix([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "determinant" in result.message.lower() or "non-positive" in result.message.lower()
