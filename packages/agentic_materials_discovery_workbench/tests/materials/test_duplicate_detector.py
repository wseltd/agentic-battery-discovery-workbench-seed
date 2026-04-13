"""Tests for MaterialsDuplicateDetector and MaterialsDuplicateResult data structures."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_materials_discovery.novelty.duplicate_detector import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    MaterialsDuplicateResult,
    MaterialsDuplicateDetector,
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
# MaterialsDuplicateResult data structure
# ---------------------------------------------------------------------------


class TestMaterialsDuplicateResultFields:
    """Verify MaterialsDuplicateResult dataclass shape and field access."""

    def test_duplicate_result_fields(self) -> None:
        """All declared fields are accessible and hold assigned values."""
        tolerances = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0}
        result = MaterialsDuplicateResult(
            query_id="struct_001",
            duplicate_of="struct_000",
            pass_type=PassType.POST_RELAX,
            matcher_tolerances=tolerances,
            is_duplicate=True,
        )
        assert result.query_id == "struct_001"
        assert result.duplicate_of == "struct_000"
        assert result.pass_type == PassType.POST_RELAX
        assert result.pass_type == "post_relax"  # StrEnum equality with str  # nosec B105
        assert result.matcher_tolerances == tolerances
        assert result.is_duplicate is True

        # Non-duplicate variant
        result_unique = MaterialsDuplicateResult(
            query_id="struct_002",
            duplicate_of=None,
            pass_type=PassType.PRE_RELAX,
            matcher_tolerances=tolerances,
            is_duplicate=False,
        )
        assert result_unique.duplicate_of is None
        assert result_unique.is_duplicate is False
        assert result_unique.pass_type == "pre_relax"  # nosec B105


# ---------------------------------------------------------------------------
# Pre-relax detection
# ---------------------------------------------------------------------------


class TestPreRelax:
    """Pre-relaxation duplicate detection via composition + cell parameters."""

    def test_pre_relax_flags_identical_composition_and_cell(self) -> None:
        """Two NaCl structures with same cell should be flagged as duplicates."""
        detector = MaterialsDuplicateDetector()
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()  # identical copy

        results = detector.detect_duplicates_pre_relax(
            [("nacl_1", s1), ("nacl_2", s2)]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[0].query_id == "nacl_1"
        assert results[1].is_duplicate is True
        assert results[1].query_id == "nacl_2"
        assert results[1].duplicate_of == "nacl_1"
        assert results[1].pass_type == PassType.PRE_RELAX

    def test_pre_relax_passes_different_composition(self) -> None:
        """NaCl and ZnS have different compositions — neither is a duplicate."""
        detector = MaterialsDuplicateDetector()
        s_nacl = _nacl_rocksalt()
        s_zns = _zns_zincblende()

        results = detector.detect_duplicates_pre_relax(
            [("nacl", s_nacl), ("zns", s_zns)]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False
        assert results[1].duplicate_of is None


# ---------------------------------------------------------------------------
# Post-relax detection
# ---------------------------------------------------------------------------


class TestPostRelax:
    """Post-relaxation duplicate detection via StructureMatcher."""

    def test_post_relax_strict_catches_niggli_equivalent(self) -> None:
        """A supercell of NaCl should be caught as a duplicate of the original.

        StructureMatcher reduces both to primitive cells before comparison,
        so a 2x1x1 supercell is identified as the same crystal.
        """
        detector = MaterialsDuplicateDetector()
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()
        s2.make_supercell([2, 1, 1])

        results = detector.detect_duplicates_post_relax(
            [("nacl_conv", s1), ("nacl_super", s2)]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "nacl_conv"
        assert results[1].pass_type == PassType.POST_RELAX

    def test_post_relax_strict_passes_genuinely_different(self) -> None:
        """NaCl and Fe BCC are genuinely different — no false positive."""
        detector = MaterialsDuplicateDetector()
        s_nacl = _nacl_rocksalt()
        s_fe = _fe_bcc()

        results = detector.detect_duplicates_post_relax(
            [("nacl", s_nacl), ("fe", s_fe)]
        )

        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is False

    def test_post_relax_tolerances_match_defaults(self) -> None:
        """Default tolerances should match the module-level constants."""
        detector = MaterialsDuplicateDetector()
        tols = detector.matcher_tolerances

        assert tols["ltol"] == DEFAULT_LTOL
        assert tols["stol"] == DEFAULT_STOL
        assert tols["angle_tol"] == DEFAULT_ANGLE_TOL

        # Confirm the values are pymatgen's standard defaults
        assert tols["ltol"] == pytest.approx(0.2)
        assert tols["stol"] == pytest.approx(0.3)
        assert tols["angle_tol"] == pytest.approx(5.0)

        # Tolerances appear in every result
        results = detector.detect_duplicates_post_relax(
            [("si", _si_diamond())]
        )
        assert results[0].matcher_tolerances == tols


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty input, degenerate cells, batch dedup."""

    def test_empty_input_returns_empty(self) -> None:
        """Both passes return empty list for empty input."""
        detector = MaterialsDuplicateDetector()

        assert detector.detect_duplicates_pre_relax([]) == []
        assert detector.detect_duplicates_post_relax([]) == []

    def test_degenerate_cell_niggli_fallback(self, monkeypatch) -> None:
        """When Niggli reduction fails, the detector falls back gracefully.

        Patches safe_niggli_reduce (as imported by strict_pass) to return
        the structure unchanged, simulating the fallback path for a
        degenerate cell. The detector should still detect duplicates.
        """
        detector = MaterialsDuplicateDetector()
        s1 = _nacl_rocksalt()
        s2 = _nacl_rocksalt()

        # Simulate the fallback: Niggli reduction is a no-op, returning
        # the original structure. Two identical inputs should still match.
        monkeypatch.setattr(
            "agentic_materials_discovery.novelty.strict_pass.safe_niggli_reduce",
            lambda structure: structure,
        )

        results = detector.detect_duplicates_post_relax(
            [("s1", s1), ("s2", s2)]
        )

        # Should still return results — two identical structures should
        # be detected as duplicates even without Niggli reduction
        assert len(results) == 2
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[1].duplicate_of == "s1"

    def test_batch_dedup_removes_all_copies(self) -> None:
        """In a batch of 5, three identical NaCl copies should all be flagged.

        Only the first-seen structure of each duplicate group is kept as
        canonical; all later copies reference it.
        """
        detector = MaterialsDuplicateDetector()
        nacl = _nacl_rocksalt()
        si = _si_diamond()

        structures = [
            ("nacl_1", nacl),
            ("si_1", si),
            ("nacl_2", _nacl_rocksalt()),  # duplicate of nacl_1
            ("nacl_3", _nacl_rocksalt()),  # duplicate of nacl_1
            ("si_2", _si_diamond()),  # duplicate of si_1
        ]

        results = detector.detect_duplicates_post_relax(structures)

        assert len(results) == 5
        # First of each group is canonical
        assert results[0].is_duplicate is False  # nacl_1
        assert results[1].is_duplicate is False  # si_1
        # Later copies reference the first-seen
        assert results[2].is_duplicate is True
        assert results[2].duplicate_of == "nacl_1"
        assert results[3].is_duplicate is True
        assert results[3].duplicate_of == "nacl_1"
        assert results[4].is_duplicate is True
        assert results[4].duplicate_of == "si_1"

        # Unique structures are exactly the first two
        unique_ids = [r.query_id for r in results if not r.is_duplicate]
        assert unique_ids == ["nacl_1", "si_1"]
