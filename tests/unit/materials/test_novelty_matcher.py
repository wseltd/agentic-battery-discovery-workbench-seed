"""Tests for structure matching via match_against_references."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.novelty_matcher import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    match_against_references,
)


# ---------------------------------------------------------------------------
# Helpers — reusable structure builders
# ---------------------------------------------------------------------------


def _nacl_rocksalt() -> Structure:
    """NaCl rock-salt conventional cell (Fm-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ]
    return Structure(lattice, species, coords)


def _fe_bcc() -> Structure:
    """BCC iron primitive cell (Im-3m, 2 atoms)."""
    lattice = Lattice.cubic(2.87)
    return Structure(lattice, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# Core matching behaviour
# ---------------------------------------------------------------------------


class TestMatchBehaviour:
    """Verify match/no-match logic for crystallographic comparison."""

    def test_identical_structure_matches(self) -> None:
        """A structure compared to itself must return the reference ID."""
        nacl = _nacl_rocksalt()
        result = match_against_references(nacl, [("mp-22862", _nacl_rocksalt())])
        assert result == "mp-22862"

    def test_no_match_returns_none(self) -> None:
        """Crystallographically different structures yield no match."""
        nacl = _nacl_rocksalt()
        fe = _fe_bcc()
        result = match_against_references(nacl, [("mp-13", fe)])
        assert result is None
        # Control: same structure does match — proves matcher is active,
        # not silently skipping all comparisons.
        control = match_against_references(nacl, [("mp-22862", _nacl_rocksalt())])
        assert control == "mp-22862"

    def test_empty_references_returns_none(self) -> None:
        """Empty reference list short-circuits to None."""
        nacl = _nacl_rocksalt()
        result = match_against_references(nacl, [])
        assert result is None
        # Verify non-empty list with a real match works (proves empty list
        # is the cause, not a broken matcher).
        with_ref = match_against_references(nacl, [("mp-22862", _nacl_rocksalt())])
        assert with_ref == "mp-22862"

    def test_first_match_wins(self) -> None:
        """When multiple references match, the first one's ID is returned."""
        nacl = _nacl_rocksalt()
        refs = [
            ("first-match", _nacl_rocksalt()),
            ("second-match", _nacl_rocksalt()),
        ]
        result = match_against_references(nacl, refs)
        assert result == "first-match"

    def test_match_skips_non_matching_then_finds_match(self) -> None:
        """Non-matching references are skipped until a match is found."""
        nacl = _nacl_rocksalt()
        refs = [
            ("mp-iron", _fe_bcc()),
            ("mp-nacl", _nacl_rocksalt()),
        ]
        result = match_against_references(nacl, refs)
        assert result == "mp-nacl"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify boundary validation rejects bad inputs early."""

    def test_non_structure_raises_type_error(self) -> None:
        """Non-Structure input raises TypeError with the actual type named."""
        with pytest.raises(TypeError) as exc_info:
            match_against_references("not a structure", [])
        assert "must be a pymatgen Structure" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_negative_ltol_raises_value_error(self) -> None:
        """Negative ltol is rejected before reaching StructureMatcher."""
        nacl = _nacl_rocksalt()
        with pytest.raises(ValueError) as exc_info:
            match_against_references(nacl, [], ltol=-0.1)
        assert "ltol" in str(exc_info.value)
        assert "-0.1" in str(exc_info.value)

    def test_negative_stol_raises_value_error(self) -> None:
        """Negative stol is rejected before reaching StructureMatcher."""
        nacl = _nacl_rocksalt()
        with pytest.raises(ValueError) as exc_info:
            match_against_references(nacl, [], stol=-0.5)
        assert "stol" in str(exc_info.value)
        assert "-0.5" in str(exc_info.value)

    def test_negative_angle_tol_raises_value_error(self) -> None:
        """Negative angle_tol is rejected before reaching StructureMatcher."""
        nacl = _nacl_rocksalt()
        with pytest.raises(ValueError) as exc_info:
            match_against_references(nacl, [], angle_tol=-1.0)
        assert "angle_tol" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)

    def test_zero_tolerances_accepted(self) -> None:
        """Zero is a valid tolerance — only negatives are rejected."""
        nacl = _nacl_rocksalt()
        # Zero tolerance should not raise ValueError.  We pass empty refs
        # so the result is deterministically None regardless of how strict
        # the matcher becomes (zero tol can reject due to float precision).
        result = match_against_references(nacl, [], ltol=0, stol=0, angle_tol=0)
        assert result is None
        # Contrast: negative IS rejected (proves boundary is at zero, not positive)
        with pytest.raises(ValueError) as exc_info:
            match_against_references(nacl, [], ltol=-0.001)
        assert "ltol" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tolerance defaults
# ---------------------------------------------------------------------------


class TestToleranceDefaults:
    """Verify tolerance defaults match pymatgen / T038 values."""

    def test_defaults_match_pymatgen_standard(self) -> None:
        """Module-level defaults must be pymatgen's StructureMatcher defaults."""
        assert DEFAULT_LTOL == pytest.approx(0.2)
        assert DEFAULT_STOL == pytest.approx(0.3)
        assert DEFAULT_ANGLE_TOL == pytest.approx(5.0)

    def test_custom_tolerances_accepted(self) -> None:
        """Custom tolerance values are accepted and propagated to the matcher."""
        nacl = _nacl_rocksalt()
        # Looser-than-default tolerances — identical structures must still match
        result = match_against_references(
            nacl,
            [("mp-22862", _nacl_rocksalt())],
            ltol=0.5,
            stol=0.5,
            angle_tol=10.0,
        )
        assert result == "mp-22862"

        # Different structure type still doesn't match even with loose tolerances
        # (NaCl vs BCC Fe are too different for any reasonable tolerance)
        no_match = match_against_references(
            nacl,
            [("mp-iron", _fe_bcc())],
            ltol=0.5,
            stol=0.5,
            angle_tol=10.0,
        )
        assert no_match is None
