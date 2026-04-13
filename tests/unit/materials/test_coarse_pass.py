"""Tests for coarse pre-relax duplicate grouping and matching."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.coarse_pass import (
    COARSE_ANGLE_TOL,
    COARSE_LTOL,
    COARSE_STOL,
    DEFAULT_VOLUME_TOLERANCE,
    coarse_deduplicate,
)
from agentic_discovery_workbench.materials.duplicate_detector import (
    DuplicateResult,
    PassType,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable structure builders
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


def _zns_zincblende() -> Structure:
    """ZnS zinc-blende conventional cell (F-43m, 8 atoms)."""
    lattice = Lattice.cubic(5.41)
    species = ["Zn", "Zn", "Zn", "Zn", "S", "S", "S", "S"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ]
    return Structure(lattice, species, coords)


def _si_diamond() -> Structure:
    """Si diamond conventional cell (Fd-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.43)
    species = ["Si"] * 8
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ]
    return Structure(lattice, species, coords)


def _fe_bcc() -> Structure:
    """BCC iron primitive cell (Im-3m, 2 atoms)."""
    lattice = Lattice.cubic(2.87)
    species = ["Fe", "Fe"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


# ---------------------------------------------------------------------------
# Empty / degenerate input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Edge case: empty input list."""

    def test_empty_input_returns_empty_list(self) -> None:
        """Empty structure list produces empty result list."""
        results = coarse_deduplicate([])
        assert results == []


# ---------------------------------------------------------------------------
# Composition grouping
# ---------------------------------------------------------------------------


class TestCompositionGrouping:
    """Structures are grouped by reduced formula before comparison."""

    def test_different_composition_never_flagged(self) -> None:
        """NaCl and ZnS have different compositions — neither is duplicate."""
        results = coarse_deduplicate(
            [("nacl", _nacl_rocksalt()), ("zns", _zns_zincblende())]
        )
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl"
        assert results[1].is_duplicate is False
        assert results[1].duplicate_of is None

    def test_same_composition_identical_cell_flagged(self) -> None:
        """Two identical NaCl structures should be flagged as duplicates."""
        results = coarse_deduplicate(
            [("nacl_1", _nacl_rocksalt()), ("nacl_2", _nacl_rocksalt())]
        )
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl_1"
        assert results[1].is_duplicate is True
        assert results[1].query_id == "nacl_2"
        assert results[1].duplicate_of == "nacl_1"


# ---------------------------------------------------------------------------
# Volume sub-grouping
# ---------------------------------------------------------------------------


class TestVolumeSubGrouping:
    """Within a composition group, structures are sub-grouped by volume."""

    def test_large_volume_difference_not_flagged(self) -> None:
        """NaCl at very different cell sizes should not match.

        A 3x scaled lattice parameter gives 27x volume — well outside
        the default 15% tolerance.
        """
        s1 = _nacl_rocksalt()
        # Scale lattice by 3x => volume scales 27x => volume/atom far apart
        s2 = _nacl_rocksalt()
        s2.scale_lattice(s2.volume * 27.0)

        results = coarse_deduplicate([("small", s1), ("huge", s2)])
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False
        assert results[1].duplicate_of is None

    def test_slightly_different_volume_within_tolerance(self) -> None:
        """NaCl with 5% volume perturbation should still match (within 15%)."""
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        s2.scale_lattice(s2.volume * 1.05)

        results = coarse_deduplicate([("orig", s1), ("perturbed", s2)])
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "orig"

    def test_custom_tight_volume_tolerance(self) -> None:
        """With a very tight tolerance (1%), 5% volume diff should not match."""
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        s2.scale_lattice(s2.volume * 1.05)

        results = coarse_deduplicate(
            [("orig", s1), ("perturbed", s2)],
            volume_tolerance=0.01,
        )
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False
        assert results[1].duplicate_of is None


# ---------------------------------------------------------------------------
# StructureMatcher pairwise within volume sub-groups
# ---------------------------------------------------------------------------


class TestPairwiseMatching:
    """After volume filtering, StructureMatcher runs pairwise."""

    def test_supercell_detected_as_duplicate(self) -> None:
        """A 2x1x1 supercell of NaCl should be caught by StructureMatcher.

        StructureMatcher reduces to primitive cell before comparison,
        so the supercell maps back to the same crystal.
        """
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        s2.make_supercell([2, 1, 1])

        results = coarse_deduplicate([("conv", s1), ("super", s2)])
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "conv"
        assert results[1].pass_type == PassType.PRE_RELAX

    def test_batch_multiple_duplicates(self) -> None:
        """In a mixed batch, all later copies reference the first-seen."""
        structures = [
            ("nacl_1", _nacl_rocksalt()),
            ("si_1", _si_diamond()),
            ("nacl_2", _nacl_rocksalt()),
            ("nacl_3", _nacl_rocksalt()),
            ("si_2", _si_diamond()),
        ]
        results = coarse_deduplicate(structures)

        assert len(results) == 5
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False
        assert results[2].is_duplicate is True
        assert results[2].duplicate_of == "nacl_1"
        assert results[3].is_duplicate is True
        assert results[3].duplicate_of == "nacl_1"
        assert results[4].is_duplicate is True
        assert results[4].duplicate_of == "si_1"

        unique_ids = [r.query_id for r in results if not r.is_duplicate]
        assert unique_ids == ["nacl_1", "si_1"]


# ---------------------------------------------------------------------------
# Result shape and field values
# ---------------------------------------------------------------------------


class TestResultFields:
    """Verify DuplicateResult fields carry correct values."""

    def test_result_has_correct_field_values(self) -> None:
        """Result fields must carry the correct tolerances and pass type."""
        results = coarse_deduplicate(
            [("nacl_1", _nacl_rocksalt()), ("nacl_2", _nacl_rocksalt())]
        )

        first = results[0]
        assert first.query_id == "nacl_1"
        assert first.duplicate_of is None
        assert first.pass_type == PassType.PRE_RELAX
        assert first.pass_type == "pre_relax"  # StrEnum str equality  # nosec B105
        assert first.is_duplicate is False
        assert first.matcher_tolerances["ltol"] == COARSE_LTOL
        assert first.matcher_tolerances["stol"] == COARSE_STOL
        assert first.matcher_tolerances["angle_tol"] == COARSE_ANGLE_TOL

        second = results[1]
        assert second.query_id == "nacl_2"
        assert second.duplicate_of == "nacl_1"
        assert second.is_duplicate is True
        assert second.matcher_tolerances == {
            "ltol": 0.3,
            "stol": 0.5,
            "angle_tol": 8.0,
        }

    def test_result_is_duplicate_result_with_expected_values(self) -> None:
        """Each result is a DuplicateResult with correct domain values."""
        results = coarse_deduplicate([("fe", _fe_bcc())])

        result = results[0]
        assert isinstance(result, DuplicateResult)
        # Value assertions — governance requires these beyond type checks
        assert result.query_id == "fe"
        assert result.is_duplicate is False
        assert result.duplicate_of is None
        assert result.pass_type == PassType.PRE_RELAX
        assert result.matcher_tolerances["ltol"] == 0.3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Boundary validation: reject bad inputs early."""

    def test_negative_volume_tolerance_raises_with_message(self) -> None:
        """Negative volume_tolerance must raise ValueError with the bad value."""
        with pytest.raises(ValueError, match=r"-0\.5"):
            coarse_deduplicate([], volume_tolerance=-0.5)

        # Verify the error message includes actionable information
        try:
            coarse_deduplicate([], volume_tolerance=-1.0)
        except ValueError as exc:
            assert "non-negative" in str(exc)
            assert "-1.0" in str(exc)
        else:
            pytest.fail("Expected ValueError for negative volume_tolerance")

    def test_zero_volume_tolerance_accepted(self) -> None:
        """Zero is a valid (very strict) tolerance — should not raise."""
        results = coarse_deduplicate(
            [("nacl", _nacl_rocksalt())], volume_tolerance=0.0
        )
        assert len(results) == 1
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl"


# ---------------------------------------------------------------------------
# Default constant values
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify default constants match documented values."""

    def test_default_volume_tolerance_value(self) -> None:
        """DEFAULT_VOLUME_TOLERANCE should be 0.15."""
        assert DEFAULT_VOLUME_TOLERANCE == pytest.approx(0.15)

    def test_coarse_matcher_tolerances(self) -> None:
        """Coarse pass uses looser tolerances than pymatgen defaults."""
        assert COARSE_LTOL == pytest.approx(0.3)
        assert COARSE_STOL == pytest.approx(0.5)
        assert COARSE_ANGLE_TOL == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Single structure
# ---------------------------------------------------------------------------


class TestSingleStructure:
    """A single structure cannot be a duplicate of anything."""

    def test_single_structure_not_duplicate(self) -> None:
        """Single input must return one non-duplicate result."""
        results = coarse_deduplicate([("only", _si_diamond())])
        assert len(results) == 1
        assert results[0].is_duplicate is False
        assert results[0].query_id == "only"
        assert results[0].duplicate_of is None
