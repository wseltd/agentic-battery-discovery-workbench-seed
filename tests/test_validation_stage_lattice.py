"""Tests for check_lattice — stage 2 lattice validation.

Exercises the numerical edge cases that matter most: NaN, Inf,
zero-determinant (singular), near-singular (volume < 0.1), and
negative-determinant (left-handed) lattices.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from validation.stage_lattice import check_lattice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic_si() -> Structure:
    """Standard cubic Si — canonical 'good' structure."""
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _make_structure_with_matrix(matrix: list[list[float]]) -> Structure:
    """Build a Structure from a raw 3x3 lattice matrix with a dummy atom."""
    return Structure(Lattice(matrix), ["H"], [[0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Valid lattice
# ---------------------------------------------------------------------------

class TestCheckLatticeValid:
    def test_valid_cubic_si(self):
        """Standard cubic Si passes all lattice checks."""
        result = check_lattice(_cubic_si())
        assert result.passed is True
        assert result.stage == "lattice_sanity"
        assert result.severity == "hard"
        assert result.message == ""

    def test_valid_orthorhombic(self):
        """Non-cubic but well-formed orthorhombic lattice passes."""
        s = _make_structure_with_matrix([
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 7.0],
        ])
        result = check_lattice(s)
        assert result.passed is True


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

class TestCheckLatticeNonFinite:
    def test_nan_in_diagonal(self):
        """NaN on a diagonal element is caught."""
        s = _make_structure_with_matrix([
            [float("nan"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False
        assert "nan" in result.message.lower() or "NaN" in result.message

    def test_nan_in_off_diagonal(self):
        """NaN in an off-diagonal position is equally rejected."""
        s = _make_structure_with_matrix([
            [5.0, float("nan"), 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False

    def test_positive_inf(self):
        """Positive Inf in the matrix is rejected."""
        s = _make_structure_with_matrix([
            [float("inf"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False
        assert "inf" in result.message.lower() or "Inf" in result.message

    def test_negative_inf(self):
        """Negative Inf is also non-finite and rejected."""
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [0.0, float("-inf"), 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False


# ---------------------------------------------------------------------------
# Determinant / volume edge cases
# ---------------------------------------------------------------------------

class TestCheckLatticeDeterminant:
    def test_zero_determinant_singular(self):
        """Two identical rows → det = 0 → non-positive determinant rejection."""
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False
        assert result.stage == "lattice_sanity"

    def test_near_singular_tiny_volume(self):
        """Volume ≈ 25 * 1e-6 = 2.5e-5 Å³ — well below the 0.1 threshold."""
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 1e-6],
        ])
        result = check_lattice(s)
        assert result.passed is False
        # Should mention volume or determinant in the message
        assert "volume" in result.message.lower() or "determinant" in result.message.lower()

    def test_volume_just_below_threshold(self):
        """Volume ≈ 0.09 — just under the 0.1 cutoff, must fail."""
        # 0.3 * 0.3 * 1.0 = 0.09
        s = _make_structure_with_matrix([
            [0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = check_lattice(s)
        assert result.passed is False

    def test_volume_just_above_threshold(self):
        """Volume ≈ 0.11 — just above the 0.1 cutoff, must pass."""
        # We need a * b * c >= 0.1 with positive det.
        # 0.32 * 0.32 * 1.0 = 0.1024
        s = _make_structure_with_matrix([
            [0.32, 0.0, 0.0],
            [0.0, 0.32, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = check_lattice(s)
        assert result.passed is True

    def test_negative_determinant_left_handed(self):
        """Swapping two rows flips the determinant sign — must be rejected."""
        s = _make_structure_with_matrix([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False
        assert "determinant" in result.message.lower() or "non-positive" in result.message.lower()

    def test_all_negative_entries_positive_det(self):
        """Matrix with negative entries can still have positive det — volume check applies.

        det([[-5,0,0],[0,-5,0],[0,0,-5]]) = -125 < 0 → rejected for non-positive det.
        """
        s = _make_structure_with_matrix([
            [-5.0, 0.0, 0.0],
            [0.0, -5.0, 0.0],
            [0.0, 0.0, -5.0],
        ])
        result = check_lattice(s)
        assert result.passed is False
