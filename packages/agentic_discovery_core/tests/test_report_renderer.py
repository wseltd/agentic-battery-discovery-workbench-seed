"""Tests for report_renderer -- provenance annotation, banned-word caveats, and digest."""

from __future__ import annotations

from agentic_discovery_core.report_constants import APPROVED_WORDING
from agentic_discovery_core.report_renderer import render_report
from agentic_discovery_core.report_schema import BudgetSettings, Report, ShortlistEntry


def _make_report(**overrides):
    defaults = {
        "run_id": "r1",
        "timestamp": "2026-01-01T00:00:00Z",
        "branch": "small_molecule",
        "tool_versions": {},
        "user_brief": "find a stable compound",
        "parsed_constraints": {},
        "budget": BudgetSettings(max_cycles=5, max_batches=8, shortlist_size=25),
        "stop_reason": None,
        "evidence_legend": {},
        "shortlist": [
            ShortlistEntry(
                candidate_id="c1",
                scores={"qed": 0.8},
                evidence_level="heuristic_estimated",
                rank=1,
            ),
        ],
        "warnings": [],
        "annexes": {},
    }
    defaults.update(overrides)
    return Report(**defaults)


def test_render_includes_provenance_metadata():
    out = render_report(_make_report())
    assert "provenance" in out
    assert "sha256_shortlist" in out["provenance"]
    assert isinstance(out["provenance"]["sha256_shortlist"], str)
    assert len(out["provenance"]["sha256_shortlist"]) == 64


def test_render_shortlist_has_provenance_note():
    out = render_report(_make_report())
    for entry in out["shortlist"]:
        assert "provenance_note" in entry
        assert isinstance(entry["provenance_note"], str)
        assert len(entry["provenance_note"]) > 0


def test_render_provenance_note_matches_evidence_level():
    report = _make_report(
        shortlist=[
            ShortlistEntry(candidate_id="c1", scores={"qed": 0.9}, evidence_level="ml_predicted", rank=1),
            ShortlistEntry(candidate_id="c2", scores={"qed": 0.7}, evidence_level="dft_verified", rank=2),
        ],
    )
    out = render_report(report)
    assert out["shortlist"][0]["provenance_note"] == APPROVED_WORDING["ml_predicted"]
    assert out["shortlist"][1]["provenance_note"] == APPROVED_WORDING["dft_verified"]


def test_render_banned_word_in_brief_adds_caveat():
    report = _make_report(user_brief="We discovered a new compound")
    out = render_report(report)
    assert "caveat" in out["provenance"]
    assert "overstate" in out["provenance"]["caveat"].lower()


def test_render_banned_word_in_warnings_adds_caveat():
    report = _make_report(warnings=["This compound is proven effective"])
    out = render_report(report)
    assert "caveat" in out["provenance"]


def test_render_no_banned_word_no_caveat():
    report = _make_report(user_brief="evaluate binding affinity", warnings=[])
    out = render_report(report)
    assert "caveat" not in out["provenance"]


def test_render_sha256_digest_deterministic():
    report = _make_report()
    out1 = render_report(report)
    out2 = render_report(report)
    assert out1["provenance"]["sha256_shortlist"] == out2["provenance"]["sha256_shortlist"]


def test_render_banned_word_boundary_no_false_positive():
    report = _make_report(
        user_brief="check approvals for the predicted compound",
        warnings=["structure is unproven by experiment"],
    )
    out = render_report(report)
    assert "caveat" not in out["provenance"]


def test_render_annexes_passed_through():
    annexes = {"timing": {"wall_s": 42.5}, "notes": "extra data"}
    report = _make_report(annexes=annexes)
    out = render_report(report)
    assert out["annexes"] == annexes


def test_render_all_evidence_levels_have_approved_wording():
    from agentic_discovery_core.evidence import EvidenceLevel

    for member in EvidenceLevel:
        label = member.value[0]
        assert label in APPROVED_WORDING
