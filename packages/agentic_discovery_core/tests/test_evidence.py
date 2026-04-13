"""Tests for agentic_discovery_core.evidence -- ordering gets heavy coverage
because incorrect ordering silently corrupts downstream ranking logic.

Budget-limit and early-stop tests live here alongside the enum tests because
they share the evidence/budget domain."""

from __future__ import annotations

import datetime
import re

from agentic_discovery_core.budget import BudgetConfig, BudgetController
from agentic_discovery_core.evidence import EvidenceLevel, attach_evidence
from agentic_discovery_core.shared.early_stop import CycleStats, evaluate_stop

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def test_all_nine_levels_defined() -> None:
    """EvidenceLevel must expose exactly the 9 specified members."""
    expected = {
        "REQUESTED",
        "GENERATED",
        "HEURISTIC_ESTIMATED",
        "ML_PREDICTED",
        "ML_RELAXED",
        "SEMIEMPIRICAL_QC",
        "DFT_VERIFIED",
        "EXPERIMENTAL_REPORTED",
        "UNKNOWN",
    }
    assert {m.name for m in EvidenceLevel} == expected
    assert len(EvidenceLevel) == 9


def test_ordering_requested_below_generated() -> None:
    """REQUESTED is the lowest credibility tier; GENERATED is one step above."""
    assert EvidenceLevel.REQUESTED < EvidenceLevel.GENERATED


def test_ordering_ml_predicted_below_dft_verified() -> None:
    """ML surrogates rank below first-principles DFT verification."""
    assert EvidenceLevel.ML_PREDICTED < EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.ML_RELAXED < EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.SEMIEMPIRICAL_QC < EvidenceLevel.DFT_VERIFIED


def test_ordering_unknown_not_above_experimental() -> None:
    """UNKNOWN is last in definition order."""
    assert not EvidenceLevel.UNKNOWN < EvidenceLevel.EXPERIMENTAL_REPORTED
    assert EvidenceLevel.UNKNOWN > EvidenceLevel.EXPERIMENTAL_REPORTED


def test_ordering_same_level_equal() -> None:
    """A level must compare equal to itself and support >= correctly."""
    assert EvidenceLevel.DFT_VERIFIED == EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.DFT_VERIFIED >= EvidenceLevel.DFT_VERIFIED
    assert not EvidenceLevel.DFT_VERIFIED < EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.REQUESTED != EvidenceLevel.GENERATED


def test_attach_stamps_level_and_source() -> None:
    """attach_evidence must set _evidence_level and _evidence_source."""
    data: dict[str, object] = {"value": 42}
    result = attach_evidence(data, EvidenceLevel.GENERATED, "xtb-6.6.1")
    assert result is data
    assert data["_evidence_level"] is EvidenceLevel.GENERATED
    assert data["_evidence_source"] == "xtb-6.6.1"


def test_attach_stamps_iso_timestamp() -> None:
    """Timestamp must be a valid ISO-8601 string close to 'now'."""
    before = datetime.datetime.now(datetime.UTC)
    data: dict[str, object] = {}
    attach_evidence(data, EvidenceLevel.DFT_VERIFIED, "vasp")
    after = datetime.datetime.now(datetime.UTC)

    ts_str = data["_evidence_timestamp"]
    assert isinstance(ts_str, str)
    assert _ISO_RE.match(ts_str), f"Timestamp {ts_str!r} is not ISO-8601"
    ts = datetime.datetime.fromisoformat(ts_str)
    assert before <= ts <= after


def test_attach_without_source_sets_none() -> None:
    """When source is omitted, _evidence_source must be None, not missing."""
    data: dict[str, object] = {}
    attach_evidence(data, EvidenceLevel.REQUESTED)
    assert "_evidence_source" in data
    assert data["_evidence_source"] is None


_EXPECTED_NAMES_IN_ORDER = [
    "REQUESTED",
    "GENERATED",
    "HEURISTIC_ESTIMATED",
    "ML_PREDICTED",
    "ML_RELAXED",
    "SEMIEMPIRICAL_QC",
    "DFT_VERIFIED",
    "EXPERIMENTAL_REPORTED",
    "UNKNOWN",
]


def test_evidence_level_enum_completeness() -> None:
    """EvidenceLevel has exactly the 9 specified members, ordered by confidence."""
    actual_names = [m.name for m in EvidenceLevel]
    assert actual_names == _EXPECTED_NAMES_IN_ORDER
    assert len(EvidenceLevel) == 9

    members = list(EvidenceLevel)
    for i in range(len(members) - 1):
        assert members[i] < members[i + 1]

    assert EvidenceLevel.REQUESTED < EvidenceLevel.UNKNOWN


def test_budget_limit_enforcement() -> None:
    """BudgetController correctly distinguishes at-limit vs beyond-limit."""
    cfg = BudgetConfig(max_cycles=5, max_batches=8, shortlist_size=25)
    bc = BudgetController(cfg)

    for i in range(4):
        bc.record_cycle(float(i))
    for _ in range(7):
        bc.record_batch(1.0, 0.0)

    stop, reason = bc.should_stop()
    assert stop is False
    assert reason is None

    remaining = bc.remaining()
    assert remaining["cycles_remaining"] == 1
    assert remaining["batches_remaining"] == 1

    bc.record_cycle(4.0)
    stop, reason = bc.should_stop()
    assert stop is True
    assert "exhausted" in reason
    assert "cycles" in reason

    assert bc.remaining()["cycles_remaining"] == 0

    bc2 = BudgetController(BudgetConfig(max_cycles=100, max_batches=3))
    for _ in range(2):
        bc2.record_batch(1.0, 0.0)
    stop, _ = bc2.should_stop()
    assert stop is False

    bc2.record_batch(1.0, 0.0)
    stop, reason = bc2.should_stop()
    assert stop is True
    assert "batches" in reason


def test_early_stop_plateau_trigger() -> None:
    """evaluate_stop fires on all three early-stop conditions independently."""
    plateau_history = [
        CycleStats(top_10_score=10.0, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=10.05, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=10.08, invalid_fraction=0.0, duplicate_fraction=0.0),
    ]
    decision = evaluate_stop(plateau_history)
    assert decision.should_stop is True
    assert decision.reason == "plateau"

    invalidity_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.10, duplicate_fraction=0.0),
        CycleStats(top_10_score=2.0, invalid_fraction=0.55, duplicate_fraction=0.0),
        CycleStats(top_10_score=3.0, invalid_fraction=0.60, duplicate_fraction=0.0),
    ]
    decision = evaluate_stop(invalidity_history)
    assert decision.should_stop is True
    assert decision.reason == "invalidity_spike"

    duplicate_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=5.0, invalid_fraction=0.0, duplicate_fraction=0.75),
    ]
    decision = evaluate_stop(duplicate_history)
    assert decision.should_stop is True
    assert decision.reason == "duplicate_surge"

    clean_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.05, duplicate_fraction=0.05),
        CycleStats(top_10_score=2.0, invalid_fraction=0.03, duplicate_fraction=0.10),
        CycleStats(top_10_score=3.0, invalid_fraction=0.02, duplicate_fraction=0.08),
    ]
    decision = evaluate_stop(clean_history)
    assert decision.should_stop is False
    assert decision.reason == "none"
