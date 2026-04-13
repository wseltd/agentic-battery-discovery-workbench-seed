"""Tests for the Q33 discovery report schema (DiscoveryReport, MoleculeAnnex, ReportMaterialsAnnex)."""

from __future__ import annotations

from amdw.reporting.schema import DiscoveryReport, MoleculeAnnex, ReportMaterialsAnnex
from amdw.shared.evidence import EvidenceLevel


# ---------------------------------------------------------------------------
# Factory helpers — synthetic instances with known values
# ---------------------------------------------------------------------------

def _make_discovery_report(**overrides: object) -> DiscoveryReport:
    """Build a DiscoveryReport with deterministic defaults."""
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "timestamp": "2026-04-13T12:00:00Z",
        "branch": "molecule",
        "tool_versions": {"rdkit": "2024.03.5", "reinvent": "4.1.0"},
        "user_brief": "Find QED > 0.7 molecules with logP < 3",
        "parsed_constraints": {"qed": {"min": 0.7}, "logp": {"max": 3.0}},
        "budget": {"max_cycles": 5, "max_batches": 10, "shortlist_size": 25},
        "evidence_levels_used": [
            EvidenceLevel.GENERATED,
            EvidenceLevel.HEURISTIC_ESTIMATED,
        ],
        "shortlist_summary": [
            {"candidate_id": "mol_a", "rank": 1, "qed": 0.85},
        ],
    }
    defaults.update(overrides)
    return DiscoveryReport(**defaults)  # type: ignore[arg-type]


def _make_molecule_annex(**overrides: object) -> MoleculeAnnex:
    """Build a MoleculeAnnex with deterministic defaults."""
    defaults: dict[str, object] = {
        "generator": "reinvent-4.1.0",
        "validity_counts": {"valid": 80, "invalid": 12, "total": 92},
        "uniqueness_count": 64,
        "novelty_counts": {"exact_known": 5, "close_analogue": 15, "novel_like": 44},
        "constraint_satisfaction": {"qed_gt_0.7": 38, "logp_lt_3": 52},
        "export_bundle_paths": {"sdf": "/out/mols.sdf", "csv": "/out/scores.csv"},
        "warnings": ["3 molecules failed sanitisation"],
    }
    defaults.update(overrides)
    return MoleculeAnnex(**defaults)  # type: ignore[arg-type]


def _make_materials_annex(**overrides: object) -> ReportMaterialsAnnex:
    """Build a ReportMaterialsAnnex with deterministic defaults."""
    defaults: dict[str, object] = {
        "generator": "mattergen-0.2.0",
        "scope_filters": {"max_atoms": 20, "excluded_elements": ["Tl", "Pb"]},
        "relaxation_backend": "mattersim-0.3.1",
        "validation_summary": {"valid": 45, "invalid": 5, "total": 50},
        "dft_handoff_paths": {"cand_001": "/dft/cand_001", "cand_002": "/dft/cand_002"},
        "warnings": ["2 candidates below hull distance threshold"],
    }
    defaults.update(overrides)
    return ReportMaterialsAnnex(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shared_fields_populated() -> None:
    """DiscoveryReport exposes all 9 shared top-level fields as non-None with correct values."""
    report = _make_discovery_report()

    assert report.run_id is not None
    assert report.run_id == "run-001"

    assert report.timestamp is not None
    assert report.timestamp == "2026-04-13T12:00:00Z"

    assert report.branch is not None
    assert report.branch == "molecule"

    assert report.tool_versions is not None
    assert report.tool_versions == {"rdkit": "2024.03.5", "reinvent": "4.1.0"}

    assert report.user_brief is not None
    assert report.user_brief == "Find QED > 0.7 molecules with logP < 3"

    assert report.parsed_constraints is not None
    assert report.parsed_constraints["qed"] == {"min": 0.7}

    assert report.budget is not None
    assert report.budget["max_cycles"] == 5

    assert report.evidence_levels_used is not None
    assert report.evidence_levels_used == [
        EvidenceLevel.GENERATED,
        EvidenceLevel.HEURISTIC_ESTIMATED,
    ]

    assert report.shortlist_summary is not None
    assert report.shortlist_summary[0]["candidate_id"] == "mol_a"


def test_molecule_annex_completeness() -> None:
    """MoleculeAnnex exposes all 7 fields with correct types and values."""
    annex = _make_molecule_annex()

    assert annex.generator is not None
    assert annex.generator == "reinvent-4.1.0"

    assert annex.validity_counts is not None
    assert annex.validity_counts["valid"] == 80
    assert annex.validity_counts["invalid"] == 12
    assert annex.validity_counts["total"] == 92

    assert annex.uniqueness_count is not None
    assert annex.uniqueness_count == 64

    assert annex.novelty_counts is not None
    assert annex.novelty_counts["exact_known"] == 5
    assert annex.novelty_counts["close_analogue"] == 15
    assert annex.novelty_counts["novel_like"] == 44

    assert annex.constraint_satisfaction is not None
    assert annex.constraint_satisfaction["qed_gt_0.7"] == 38

    assert annex.export_bundle_paths is not None
    assert annex.export_bundle_paths["sdf"] == "/out/mols.sdf"

    assert annex.warnings is not None
    assert annex.warnings == ["3 molecules failed sanitisation"]


def test_materials_annex_completeness() -> None:
    """ReportMaterialsAnnex exposes all 6 fields with correct types and values."""
    annex = _make_materials_annex()

    assert annex.generator is not None
    assert annex.generator == "mattergen-0.2.0"

    assert annex.scope_filters is not None
    assert annex.scope_filters["max_atoms"] == 20
    assert annex.scope_filters["excluded_elements"] == ["Tl", "Pb"]

    assert annex.relaxation_backend is not None
    assert annex.relaxation_backend == "mattersim-0.3.1"

    assert annex.validation_summary is not None
    assert annex.validation_summary["valid"] == 45
    assert annex.validation_summary["total"] == 50

    assert annex.dft_handoff_paths is not None
    assert annex.dft_handoff_paths["cand_001"] == "/dft/cand_001"
    assert annex.dft_handoff_paths["cand_002"] == "/dft/cand_002"

    assert annex.warnings is not None
    assert annex.warnings == ["2 candidates below hull distance threshold"]
