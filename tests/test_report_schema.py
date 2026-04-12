"""Tests for discovery_workbench.report_schema."""

from __future__ import annotations

import pytest

from discovery_workbench.report_schema import (
    BudgetSettings,
    Report,
    ShortlistEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _budget() -> BudgetSettings:
    return BudgetSettings(max_cycles=5, max_batches=8, shortlist_size=25)


def _entry(**overrides: object) -> ShortlistEntry:
    defaults: dict[str, object] = {
        "candidate_id": "c1",
        "scores": {"qed": 0.8},
        "evidence_level": "heuristic_estimated",
        "rank": 1,
    }
    defaults.update(overrides)
    return ShortlistEntry(**defaults)  # type: ignore[arg-type]


def _report(**overrides: object) -> Report:
    defaults: dict[str, object] = {
        "run_id": "r1",
        "timestamp": "2026-01-01T00:00:00Z",
        "branch": "small_molecule",
        "tool_versions": {"rdkit": "2024.03"},
        "user_brief": "test",
        "parsed_constraints": {},
        "budget": _budget(),
        "stop_reason": None,
        "evidence_legend": {},
        "shortlist": [_entry()],
        "warnings": [],
        "annexes": {},
    }
    defaults.update(overrides)
    return Report(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BudgetSettings
# ---------------------------------------------------------------------------

def test_budget_settings_fields() -> None:
    """BudgetSettings stores max_cycles, max_batches, and shortlist_size."""
    b = BudgetSettings(max_cycles=10, max_batches=3, shortlist_size=50)
    assert b.max_cycles == 10
    assert b.max_batches == 3
    assert b.shortlist_size == 50


# ---------------------------------------------------------------------------
# ShortlistEntry
# ---------------------------------------------------------------------------

def test_shortlist_entry_stores_evidence_level() -> None:
    """ShortlistEntry preserves the evidence_level string."""
    entry = _entry(evidence_level="ml_predicted")
    assert entry.evidence_level == "ml_predicted"


def test_shortlist_rejects_empty_candidate_id() -> None:
    """Empty candidate_id is rejected at construction time."""
    with pytest.raises(ValueError, match="candidate_id"):
        _entry(candidate_id="")


def test_shortlist_rejects_zero_rank() -> None:
    """rank must be >= 1."""
    with pytest.raises(ValueError, match="rank"):
        _entry(rank=0)


def test_shortlist_rejects_negative_rank() -> None:
    """Negative rank is rejected."""
    with pytest.raises(ValueError, match="rank"):
        _entry(rank=-3)


def test_shortlist_rejects_invalid_evidence_level() -> None:
    """An evidence_level not in EvidenceLevel labels is rejected."""
    with pytest.raises(ValueError, match="evidence_level"):
        _entry(evidence_level="made_up")


# ---------------------------------------------------------------------------
# Report — valid construction
# ---------------------------------------------------------------------------

def test_report_valid_construction() -> None:
    """A Report with all valid fields constructs without error."""
    r = _report()
    assert r.run_id == "r1"
    assert r.branch == "small_molecule"
    assert len(r.shortlist) == 1
    assert r.shortlist[0].candidate_id == "c1"


def test_report_allows_empty_shortlist() -> None:
    """An empty shortlist is valid — the run may have found nothing."""
    r = _report(shortlist=[])
    assert r.shortlist == []


def test_report_allows_none_stop_reason() -> None:
    """stop_reason=None is valid — the run may still be in progress."""
    r = _report(stop_reason=None)
    assert r.stop_reason is None


def test_report_inorganic_branch() -> None:
    """The inorganic_materials branch is accepted."""
    r = _report(branch="inorganic_materials")
    assert r.branch == "inorganic_materials"


# ---------------------------------------------------------------------------
# Report — rejection
# ---------------------------------------------------------------------------

def test_report_rejects_empty_run_id() -> None:
    """Empty run_id is rejected at construction time."""
    with pytest.raises(ValueError, match="run_id"):
        _report(run_id="")


def test_report_rejects_unknown_branch() -> None:
    """A branch not in VALID_BRANCHES is rejected."""
    with pytest.raises(ValueError, match="branch"):
        _report(branch="protein_folding")


def test_report_rejects_unparseable_timestamp() -> None:
    """A non-ISO-8601 timestamp string is rejected."""
    with pytest.raises(ValueError, match="timestamp"):
        _report(timestamp="not-a-date")


def test_report_rejects_numeric_timestamp() -> None:
    """A numeric timestamp (wrong type) is rejected."""
    with pytest.raises(ValueError, match="timestamp"):
        _report(timestamp=12345)  # type: ignore[arg-type]
