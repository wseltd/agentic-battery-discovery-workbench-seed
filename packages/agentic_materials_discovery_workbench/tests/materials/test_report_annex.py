"""Tests for materials report annex builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_materials_discovery.ranking.ranker import RankedCandidate
from agentic_materials_discovery.reporting.report_annex import (
    MaterialsAnnex,
    MaterialsAnnexInput,
    build_materials_annex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candidate(
    candidate_id: str = "cand_001",
    composition: str = "BaTiO3",
    space_group_number: int = 99,
    composite_score: float = 0.85,
) -> RankedCandidate:
    """Create a minimal RankedCandidate for testing."""
    return RankedCandidate(
        candidate_id=candidate_id,
        composition=composition,
        space_group_number=space_group_number,
        stability_score=0.9,
        symmetry_score=1.0,
        complexity_score=0.8,
        target_satisfaction_score=0.7,
        composite_score=composite_score,
        rank=1,
    )


def _make_annex_input(
    candidates: list[RankedCandidate] | None = None,
    dft_paths: list[Path] | None = None,
    generator_config: dict | None = None,
    scope_config: dict | None = None,
    validity_count: int = 8,
    uniqueness_count: int = 6,
    novelty_count: int = 4,
    total_generated: int = 10,
) -> MaterialsAnnexInput:
    """Create a MaterialsAnnexInput with sensible defaults."""
    if candidates is None:
        candidates = [_make_candidate()]
    if dft_paths is None:
        dft_paths = [Path(f"output/dft/{c.candidate_id}") for c in candidates]
    if generator_config is None:
        generator_config = {
            "checkpoint": "mattergen_v2.1",
            "conditioning_mode": "composition",
            "num_samples": 100,
        }
    if scope_config is None:
        scope_config = {
            "max_atoms": 20,
            "excluded_elements": ["Tl", "Pb"],
            "target_space_group": 225,
        }

    return MaterialsAnnexInput(
        generator_config=generator_config,
        scope_config=scope_config,
        relaxer_version="mattersim-0.3.1",
        ranked_candidates=candidates,
        validity_count=validity_count,
        uniqueness_count=uniqueness_count,
        novelty_count=novelty_count,
        total_generated=total_generated,
        matcher_tolerances={"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0},
        reference_db_ids=["mp-2023.11", "alexandria-3d-v2"],
        dft_handoff_paths=dft_paths,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_annex_returns_materials_annex():
    """build_materials_annex must return a MaterialsAnnex with correct relaxer version."""
    annex_input = _make_annex_input()
    result = build_materials_annex(annex_input)
    assert isinstance(result, MaterialsAnnex)
    assert result.relaxer_section["version"] == "mattersim-0.3.1"


def test_generator_section_contains_checkpoint():
    """Generator section must preserve the checkpoint key from config."""
    annex_input = _make_annex_input(
        generator_config={"checkpoint": "mattergen_v2.1", "conditioning_mode": "composition"},
    )
    result = build_materials_annex(annex_input)
    assert result.generator_section["checkpoint"] == "mattergen_v2.1"


def test_generator_section_contains_conditioning_mode():
    """Generator section must preserve the conditioning_mode key."""
    annex_input = _make_annex_input(
        generator_config={"checkpoint": "v1", "conditioning_mode": "space_group"},
    )
    result = build_materials_annex(annex_input)
    assert result.generator_section["conditioning_mode"] == "space_group"


def test_scope_section_contains_max_atoms():
    """Scope section must preserve max_atoms from scope config."""
    annex_input = _make_annex_input(
        scope_config={"max_atoms": 32, "excluded_elements": []},
    )
    result = build_materials_annex(annex_input)
    assert result.scope_section["max_atoms"] == 32


def test_scope_section_contains_excluded_elements():
    """Scope section must preserve excluded_elements list."""
    annex_input = _make_annex_input(
        scope_config={"max_atoms": 20, "excluded_elements": ["Hg", "Cd"]},
    )
    result = build_materials_annex(annex_input)
    assert result.scope_section["excluded_elements"] == ["Hg", "Cd"]


def test_relaxer_section_contains_version():
    """Relaxer section must contain the relaxer version string."""
    annex_input = _make_annex_input()
    result = build_materials_annex(annex_input)
    assert result.relaxer_section["version"] == "mattersim-0.3.1"


def test_validation_stats_keys_present():
    """Validation stats must contain all four required count keys."""
    annex_input = _make_annex_input(
        validity_count=7,
        uniqueness_count=5,
        novelty_count=3,
        total_generated=12,
    )
    result = build_materials_annex(annex_input)
    assert result.validation_stats["validity_count"] == 7
    assert result.validation_stats["uniqueness_count"] == 5
    assert result.validation_stats["novelty_count"] == 3
    assert result.validation_stats["total_generated"] == 12


def test_novelty_details_contains_reference_dbs():
    """Novelty details must list the queried reference databases."""
    annex_input = _make_annex_input()
    result = build_materials_annex(annex_input)
    assert result.novelty_details["reference_dbs"] == [
        "mp-2023.11", "alexandria-3d-v2",
    ]


def test_novelty_details_contains_matcher_tolerances():
    """Novelty details must include the matcher tolerance parameters."""
    annex_input = _make_annex_input()
    result = build_materials_annex(annex_input)
    tolerances = result.novelty_details["matcher_tolerances"]
    assert tolerances["ltol"] == pytest.approx(0.2)
    assert tolerances["stol"] == pytest.approx(0.3)
    assert tolerances["angle_tol"] == pytest.approx(5.0)


def test_novelty_details_novelty_fraction():
    """Novelty fraction must equal novelty_count / total_generated."""
    annex_input = _make_annex_input(novelty_count=3, total_generated=12)
    result = build_materials_annex(annex_input)
    assert result.novelty_details["novelty_fraction"] == pytest.approx(0.25)


def test_dft_handoff_summary_matches_candidates():
    """DFT handoff summary must have one entry per ranked candidate."""
    candidates = [
        _make_candidate(candidate_id="mat_A"),
        _make_candidate(candidate_id="mat_B", composite_score=0.7),
    ]
    dft_paths = [Path("output/dft/mat_A"), Path("output/dft/mat_B")]
    annex_input = _make_annex_input(candidates=candidates, dft_paths=dft_paths)
    result = build_materials_annex(annex_input)

    assert len(result.dft_handoff_summary) == 2
    ids = [entry["candidate_id"] for entry in result.dft_handoff_summary]
    assert "mat_A" in ids
    assert "mat_B" in ids

    # Verify paths are associated with correct candidates
    for entry in result.dft_handoff_summary:
        assert "paths" in entry
        for p in entry["paths"]:
            assert entry["candidate_id"] in p


def test_warnings_list_populated():
    """Warnings should flag when all unique candidates match existing entries."""
    annex_input = _make_annex_input(
        novelty_count=0,
        uniqueness_count=5,
        total_generated=10,
    )
    result = build_materials_annex(annex_input)
    assert any("zero novel" in w for w in result.warnings)


def test_empty_candidates_produces_empty_handoff():
    """An empty candidate list must produce an empty handoff summary."""
    annex_input = _make_annex_input(
        candidates=[],
        dft_paths=[],
        total_generated=0,
        validity_count=0,
        uniqueness_count=0,
        novelty_count=0,
    )
    result = build_materials_annex(annex_input)
    assert result.dft_handoff_summary == []
    # novelty_fraction should be 0.0 when total is 0, not a division error
    assert result.novelty_details["novelty_fraction"] == pytest.approx(0.0)


def test_warnings_on_zero_validity():
    """Warnings should flag when no candidates passed validity checks."""
    annex_input = _make_annex_input(
        validity_count=0,
        total_generated=10,
    )
    result = build_materials_annex(annex_input)
    assert any("validity" in w.lower() for w in result.warnings)


def test_warning_on_zero_total_generated():
    """Zero total_generated is a suspicious condition that must produce a warning."""
    annex_input = _make_annex_input(
        candidates=[],
        dft_paths=[],
        total_generated=0,
        validity_count=0,
        uniqueness_count=0,
        novelty_count=0,
    )
    result = build_materials_annex(annex_input)
    assert any("zero structures generated" in w.lower() for w in result.warnings)


def test_warning_on_empty_dft_paths_with_candidates():
    """Missing DFT paths when candidates exist must produce a warning."""
    candidates = [_make_candidate()]
    annex_input = _make_annex_input(candidates=candidates, dft_paths=[])
    result = build_materials_annex(annex_input)
    assert any("no dft handoff paths" in w.lower() for w in result.warnings)


def test_generator_section_does_not_mutate_input():
    """Generator section must be a copy — mutating it must not affect input."""
    config = {"checkpoint": "v1", "conditioning_mode": "comp"}
    annex_input = _make_annex_input(generator_config=config)
    result = build_materials_annex(annex_input)
    result.generator_section["checkpoint"] = "MUTATED"
    assert annex_input.generator_config["checkpoint"] == "v1"
