"""Tests for strict post-relax duplicate detection via Niggli reduction."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.duplicate_detector import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    MaterialsDuplicateResult,
    PassType,
)
from agentic_discovery_workbench.materials.strict_pass import strict_deduplicate


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
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Edge case: empty structure list."""

    def test_empty_input_returns_empty_list(self) -> None:
        """Empty input produces empty result — no crash, no sentinel values."""
        results = strict_deduplicate([])
        assert results == []


# ---------------------------------------------------------------------------
# Niggli-equivalent detection
# ---------------------------------------------------------------------------


class TestNiggliEquivalent:
    """Structures that differ only in cell convention should be caught."""

    def test_supercell_detected_as_duplicate(self) -> None:
        """A 2x1x1 supercell of NaCl should be caught after Niggli reduction.

        StructureMatcher reduces to primitive before comparison, so
        the supercell maps back to the same crystal.
        """
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        s2.make_supercell([2, 1, 1])

        results = strict_deduplicate([("nacl_conv", s1), ("nacl_super", s2)])

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl_conv"
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "nacl_conv"
        assert results[1].pass_type == PassType.POST_RELAX

    def test_identical_structures_flagged(self) -> None:
        """Two identical NaCl structures should be flagged as duplicates."""
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()

        results = strict_deduplicate([("nacl_1", s1), ("nacl_2", s2)])

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "nacl_1"


# ---------------------------------------------------------------------------
# Genuinely different structures
# ---------------------------------------------------------------------------


class TestGenuinelyDifferent:
    """Structures with different chemistry or topology must not match."""

    def test_different_chemistry_not_flagged(self) -> None:
        """NaCl and Fe BCC are genuinely different — no false positive."""
        results = strict_deduplicate(
            [("nacl", _nacl_rocksalt()), ("fe", _fe_bcc())]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False
        assert results[1].duplicate_of is None

    def test_different_structure_type_not_flagged(self) -> None:
        """NaCl and ZnS have different compositions — not duplicates."""
        results = strict_deduplicate(
            [("nacl", _nacl_rocksalt()), ("zns", _zns_zincblende())]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


class TestTolerances:
    """Verify that strict_deduplicate uses pymatgen-default tolerances."""

    def test_result_tolerances_match_defaults(self) -> None:
        """Tolerances in results should be the pymatgen standard defaults."""
        results = strict_deduplicate([("si", _si_diamond())])

        assert len(results) == 1
        tols = results[0].matcher_tolerances
        assert tols["ltol"] == pytest.approx(DEFAULT_LTOL)
        assert tols["stol"] == pytest.approx(DEFAULT_STOL)
        assert tols["angle_tol"] == pytest.approx(DEFAULT_ANGLE_TOL)

        # Confirm the actual numeric values
        assert tols["ltol"] == pytest.approx(0.2)
        assert tols["stol"] == pytest.approx(0.3)
        assert tols["angle_tol"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Niggli fallback
# ---------------------------------------------------------------------------


class TestNiggliFallback:
    """When Niggli reduction fails, the fallback must still produce results."""

    def test_degenerate_cell_falls_back_gracefully(self) -> None:
        """When safe_niggli_reduce returns the unreduced structure (its fallback
        behaviour for degenerate cells), strict_deduplicate should still
        correctly detect duplicates via StructureMatcher.

        We patch safe_niggli_reduce to return the input unchanged, simulating
        the fallback path. Patching Lattice.get_niggli_reduced_lattice
        globally would also break StructureMatcher internals, so we mock
        at the safe_niggli_reduce boundary instead.
        """
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()

        # Return structure unchanged — simulates degenerate-cell fallback
        with patch(
            "agentic_discovery_workbench.materials.strict_pass.safe_niggli_reduce",
            side_effect=lambda s: s,
        ):
            results = strict_deduplicate([("s1", s1), ("s2", s2)])

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "s1"


# ---------------------------------------------------------------------------
# Batch deduplication
# ---------------------------------------------------------------------------


class TestBatchDedup:
    """Batch with multiple duplicate groups."""

    def test_all_copies_reference_first_seen(self) -> None:
        """In a batch of 5, later copies all reference the first-seen canonical.

        Only the first of each crystal type survives as non-duplicate.
        """
        structures = [
            ("nacl_1", _nacl_rocksalt()),
            ("si_1", _si_diamond()),
            ("nacl_2", _nacl_rocksalt()),
            ("nacl_3", _nacl_rocksalt()),
            ("si_2", _si_diamond()),
        ]

        results = strict_deduplicate(structures)

        assert len(results) == 5
        # First of each group is canonical
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl_1"
        assert results[1].is_duplicate is False
        assert results[1].query_id == "si_1"
        # Later copies reference the first-seen
        assert results[2].is_duplicate is True
        assert results[2].duplicate_of == "nacl_1"
        assert results[3].is_duplicate is True
        assert results[3].duplicate_of == "nacl_1"
        assert results[4].is_duplicate is True
        assert results[4].duplicate_of == "si_1"

        unique_ids = [r.query_id for r in results if not r.is_duplicate]
        assert unique_ids == ["nacl_1", "si_1"]


# ---------------------------------------------------------------------------
# Result field values
# ---------------------------------------------------------------------------


class TestResultFields:
    """Verify MaterialsDuplicateResult fields carry correct values from strict pass."""

    def test_result_has_correct_pass_type_and_fields(self) -> None:
        """Each result is a MaterialsDuplicateResult with POST_RELAX pass type."""
        results = strict_deduplicate(
            [("nacl_1", _nacl_rocksalt()), ("nacl_2", _nacl_rocksalt())]
        )

        first = results[0]
        assert isinstance(first, MaterialsDuplicateResult)
        assert first.query_id == "nacl_1"
        assert first.duplicate_of is None
        assert first.pass_type == PassType.POST_RELAX
        assert first.pass_type == "post_relax"  # StrEnum str equality  # nosec B105
        assert first.is_duplicate is False

        second = results[1]
        assert second.query_id == "nacl_2"
        assert second.duplicate_of == "nacl_1"
        assert second.is_duplicate is True
        assert second.pass_type == PassType.POST_RELAX

    def test_single_structure_never_duplicate(self) -> None:
        """A single structure in the batch cannot be a duplicate."""
        results = strict_deduplicate([("only", _fe_bcc())])

        assert len(results) == 1
        assert results[0].is_duplicate is False
        assert results[0].query_id == "only"
        assert results[0].duplicate_of is None
        assert results[0].pass_type == PassType.POST_RELAX
