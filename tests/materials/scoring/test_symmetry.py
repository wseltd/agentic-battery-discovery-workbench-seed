"""Tests for space-group symmetry scoring."""

import pytest

from discovery_workbench.materials.scoring.symmetry import (
    PARTIAL_SYSTEM_MATCH_SCORE,
    symmetry_score,
)


# ---------------------------------------------------------------------------
# No constraint — always 1.0
# ---------------------------------------------------------------------------


def test_no_constraint_returns_one():
    """No requested SG or crystal system means no penalty."""
    assert symmetry_score(1) == 1.0


def test_no_constraint_any_sg_returns_one():
    """Any verified SG with no constraint scores 1.0."""
    assert symmetry_score(230) == 1.0


# ---------------------------------------------------------------------------
# Exact space-group match
# ---------------------------------------------------------------------------


def test_exact_sg_match_returns_one():
    """Verified SG equals requested SG — perfect score."""
    assert symmetry_score(225, requested_sg=225) == 1.0


def test_exact_sg_match_low_number():
    """Exact match works for SG 1 (triclinic P1)."""
    assert symmetry_score(1, requested_sg=1) == 1.0


def test_exact_sg_match_high_number():
    """Exact match works for SG 230 (cubic Ia-3d)."""
    assert symmetry_score(230, requested_sg=230) == 1.0


# ---------------------------------------------------------------------------
# Same crystal system, different SG — partial credit
# ---------------------------------------------------------------------------


def test_same_system_partial_credit():
    """Different SG but same crystal system scores 0.5."""
    # SG 225 (Fm-3m) and SG 229 (Im-3m) are both cubic.
    score = symmetry_score(225, requested_sg=229)
    assert score == PARTIAL_SYSTEM_MATCH_SCORE


def test_same_system_monoclinic():
    """Monoclinic SGs 3 and 15 — same system, partial credit."""
    assert symmetry_score(3, requested_sg=15) == PARTIAL_SYSTEM_MATCH_SCORE


def test_same_system_trigonal():
    """Trigonal SGs 143 and 167 — boundary SGs in same system."""
    assert symmetry_score(143, requested_sg=167) == PARTIAL_SYSTEM_MATCH_SCORE


# ---------------------------------------------------------------------------
# Different crystal system — zero
# ---------------------------------------------------------------------------


def test_different_system_returns_zero():
    """Cubic vs triclinic — completely different systems score 0.0."""
    assert symmetry_score(225, requested_sg=1) == 0.0


def test_different_system_adjacent_boundaries():
    """SG 74 (orthorhombic) vs SG 75 (tetragonal) — adjacent but different."""
    assert symmetry_score(74, requested_sg=75) == 0.0


# ---------------------------------------------------------------------------
# Crystal system string matching
# ---------------------------------------------------------------------------


def test_crystal_system_match_returns_partial():
    """Verified SG in requested crystal system scores 0.5."""
    # SG 225 is cubic.
    assert symmetry_score(225, requested_crystal_system="cubic") == PARTIAL_SYSTEM_MATCH_SCORE


def test_crystal_system_mismatch_returns_zero():
    """Verified SG not in requested crystal system scores 0.0."""
    # SG 1 is triclinic, not cubic.
    assert symmetry_score(1, requested_crystal_system="cubic") == 0.0


def test_crystal_system_case_insensitive():
    """Crystal system name matching is case-insensitive."""
    assert symmetry_score(225, requested_crystal_system="Cubic") == PARTIAL_SYSTEM_MATCH_SCORE


def test_crystal_system_with_whitespace():
    """Leading/trailing whitespace is stripped from crystal system name."""
    assert symmetry_score(225, requested_crystal_system="  cubic  ") == PARTIAL_SYSTEM_MATCH_SCORE


# ---------------------------------------------------------------------------
# Both requested_sg and requested_crystal_system provided
# ---------------------------------------------------------------------------


def test_exact_sg_overrides_crystal_system():
    """Exact SG match gives 1.0 even if crystal system also provided."""
    assert symmetry_score(225, requested_sg=225, requested_crystal_system="cubic") == 1.0


def test_sg_mismatch_falls_back_to_crystal_system():
    """SG mismatch with matching crystal system gives partial credit."""
    # SG 229 requested but got 225 — both cubic, crystal_system also "cubic".
    score = symmetry_score(225, requested_sg=229, requested_crystal_system="cubic")
    assert score == PARTIAL_SYSTEM_MATCH_SCORE


# ---------------------------------------------------------------------------
# Boundary validation — verified_sg
# ---------------------------------------------------------------------------


def test_invalid_verified_sg_zero():
    """SG 0 is not valid — must raise with informative message."""
    with pytest.raises(ValueError, match="verified_sg must be between") as exc_info:
        symmetry_score(0)
    assert "got 0" in str(exc_info.value)


def test_invalid_verified_sg_negative():
    """Negative SG is invalid — must raise with informative message."""
    with pytest.raises(ValueError, match="verified_sg must be between") as exc_info:
        symmetry_score(-5)
    assert "got -5" in str(exc_info.value)


def test_invalid_verified_sg_above_230():
    """SG 231 exceeds maximum — must raise with informative message."""
    with pytest.raises(ValueError, match="verified_sg must be between") as exc_info:
        symmetry_score(231)
    assert "got 231" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Boundary validation — requested_sg
# ---------------------------------------------------------------------------


def test_invalid_requested_sg():
    """Invalid requested SG raises ValueError with parameter name."""
    with pytest.raises(ValueError, match="requested_sg must be between") as exc_info:
        symmetry_score(100, requested_sg=999)
    assert "got 999" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Boundary validation — requested_crystal_system
# ---------------------------------------------------------------------------


def test_invalid_crystal_system_name():
    """Unrecognised crystal system name raises with valid options."""
    with pytest.raises(ValueError, match="Unknown crystal system") as exc_info:
        symmetry_score(100, requested_crystal_system="invalid_system")
    assert "invalid_system" in str(exc_info.value)
