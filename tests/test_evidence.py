"""Tests for discovery_workbench.evidence — ordering gets heavy coverage
because incorrect ordering silently corrupts downstream ranking logic.

Budget-limit and early-stop tests live here alongside the enum tests because
they share the evidence/budget domain and the ticket (T054) groups them."""

from __future__ import annotations

import datetime
import re

from discovery_workbench.budget import BudgetConfig, BudgetController
from discovery_workbench.evidence import EvidenceLevel, attach_evidence
from discovery_workbench.shared.early_stop import CycleStats, evaluate_stop

# ISO-8601 UTC pattern (no timezone suffix required — datetime.isoformat()
# omits it for UTC-aware datetimes produced via datetime.UTC).
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


# -- enum membership ---------------------------------------------------------

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


# -- ordering (5 tests) — the critical contract ------------------------------

def test_ordering_requested_below_generated() -> None:
    """REQUESTED is the lowest credibility tier; GENERATED is one step above."""
    assert EvidenceLevel.REQUESTED < EvidenceLevel.GENERATED


def test_ordering_ml_predicted_below_dft_verified() -> None:
    """ML surrogates rank below first-principles DFT verification."""
    assert EvidenceLevel.ML_PREDICTED < EvidenceLevel.DFT_VERIFIED
    # Also verify the intermediate levels respect the chain.
    assert EvidenceLevel.ML_RELAXED < EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.SEMIEMPIRICAL_QC < EvidenceLevel.DFT_VERIFIED


def test_ordering_unknown_not_above_experimental() -> None:
    """UNKNOWN is last in definition order — it must NOT compare as below
    EXPERIMENTAL_REPORTED (it is *above* it in rank), which is the intended
    semantic: unknown means 'unclassified', placed at end of the enum."""
    assert not EvidenceLevel.UNKNOWN < EvidenceLevel.EXPERIMENTAL_REPORTED
    assert EvidenceLevel.UNKNOWN > EvidenceLevel.EXPERIMENTAL_REPORTED


def test_ordering_same_level_equal() -> None:
    """A level must compare equal to itself and support >= correctly."""
    assert EvidenceLevel.DFT_VERIFIED == EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.DFT_VERIFIED >= EvidenceLevel.DFT_VERIFIED
    assert not EvidenceLevel.DFT_VERIFIED < EvidenceLevel.DFT_VERIFIED
    # Different members must not be equal.
    assert EvidenceLevel.REQUESTED != EvidenceLevel.GENERATED


# -- attach_evidence (3 tests) -----------------------------------------------

def test_attach_stamps_level_and_source() -> None:
    """attach_evidence must set _evidence_level and _evidence_source."""
    data: dict[str, object] = {"value": 42}
    result = attach_evidence(data, EvidenceLevel.GENERATED, "xtb-6.6.1")
    assert result is data  # mutates in place and returns same object
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
    # The stamped time must fall within the before/after window.
    assert before <= ts <= after


def test_attach_without_source_sets_none() -> None:
    """When source is omitted, _evidence_source must be None, not missing."""
    data: dict[str, object] = {}
    attach_evidence(data, EvidenceLevel.REQUESTED)
    assert "_evidence_source" in data
    assert data["_evidence_source"] is None


# -- T054: enum completeness, budget enforcement, early-stop triggers ----------

# Canonical ordering: each member must rank strictly below the next.
# UNKNOWN is intentionally last (highest rank) — it means "unclassified",
# not "low confidence".
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
    """EvidenceLevel has exactly the 9 specified members and they are ordered
    by increasing confidence (definition order)."""
    actual_names = [m.name for m in EvidenceLevel]
    assert actual_names == _EXPECTED_NAMES_IN_ORDER, (
        f"Expected {_EXPECTED_NAMES_IN_ORDER}, got {actual_names}"
    )
    assert len(EvidenceLevel) == 9

    # Verify strict ascending ordering across the full chain.
    members = list(EvidenceLevel)
    for i in range(len(members) - 1):
        assert members[i] < members[i + 1], (
            f"{members[i]} should rank below {members[i + 1]}"
        )

    # Spot-check extremes: lowest < highest.
    assert EvidenceLevel.REQUESTED < EvidenceLevel.UNKNOWN


def test_budget_limit_enforcement() -> None:
    """BudgetController correctly distinguishes at-limit (allowed) from
    beyond-limit (blocked) for both cycles and batches."""
    cfg = BudgetConfig(max_cycles=5, max_batches=8, shortlist_size=25)
    bc = BudgetController(cfg)

    # Use 4 cycles and 7 batches — one below limit for each.
    for i in range(4):
        bc.record_cycle(float(i))
    for _ in range(7):
        bc.record_batch(1.0, 0.0)

    # At-limit (one below max): should NOT stop.
    stop, reason = bc.should_stop()
    assert stop is False, f"Expected no stop at 4/5 cycles, got reason={reason}"
    assert reason is None

    remaining = bc.remaining()
    assert remaining["cycles_remaining"] == 1
    assert remaining["batches_remaining"] == 1

    # Push cycles to exactly max_cycles (5) — beyond limit: must stop.
    bc.record_cycle(4.0)
    stop, reason = bc.should_stop()
    assert stop is True, "Expected stop at 5/5 cycles"
    assert "exhausted" in reason
    assert "cycles" in reason

    # Verify remaining floors at zero.
    assert bc.remaining()["cycles_remaining"] == 0

    # --- Separate controller for batch exhaustion ---
    bc2 = BudgetController(BudgetConfig(max_cycles=100, max_batches=3))
    for _ in range(2):
        bc2.record_batch(1.0, 0.0)
    stop, _ = bc2.should_stop()
    assert stop is False, "2/3 batches should not stop"

    bc2.record_batch(1.0, 0.0)  # batch 3 = max_batches
    stop, reason = bc2.should_stop()
    assert stop is True, "Expected stop at 3/3 batches"
    assert "batches" in reason


def test_early_stop_plateau_trigger() -> None:
    """evaluate_stop fires on all three early-stop conditions independently:
    plateau (<1% relative improvement), invalidity (>50%), and duplicates (>70%).

    Each sub-case builds a standalone cycle history that isolates exactly one
    trigger, verifying that the function returns the correct reason and
    should_stop=True.  A clean history is also tested to confirm no false
    positive when no condition is met.
    """
    # --- Plateau trigger ---
    # Three cycles where the last two transitions have <1% relative improvement.
    # Scores: 10.0 → 10.05 → 10.08.
    # Relative improvements: 0.005 (0.5%) and 0.003 (0.3%) — both below 1%.
    plateau_history = [
        CycleStats(top_10_score=10.0, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=10.05, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=10.08, invalid_fraction=0.0, duplicate_fraction=0.0),
    ]
    decision = evaluate_stop(plateau_history)
    assert decision.should_stop is True, f"Plateau should fire: {decision.detail}"
    assert decision.reason == "plateau"

    # --- Invalidity spike trigger ---
    # Two consecutive cycles with invalid_fraction >= 0.50.
    invalidity_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.10, duplicate_fraction=0.0),
        CycleStats(top_10_score=2.0, invalid_fraction=0.55, duplicate_fraction=0.0),
        CycleStats(top_10_score=3.0, invalid_fraction=0.60, duplicate_fraction=0.0),
    ]
    decision = evaluate_stop(invalidity_history)
    assert decision.should_stop is True, f"Invalidity should fire: {decision.detail}"
    assert decision.reason == "invalidity_spike"

    # --- Duplicate surge trigger ---
    # Latest cycle has duplicate_fraction >= 0.70.
    duplicate_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.0, duplicate_fraction=0.0),
        CycleStats(top_10_score=5.0, invalid_fraction=0.0, duplicate_fraction=0.75),
    ]
    decision = evaluate_stop(duplicate_history)
    assert decision.should_stop is True, f"Duplicate surge should fire: {decision.detail}"
    assert decision.reason == "duplicate_surge"

    # --- No trigger (clean history) ---
    # Good improvement, low invalidity, low duplicates.
    clean_history = [
        CycleStats(top_10_score=1.0, invalid_fraction=0.05, duplicate_fraction=0.05),
        CycleStats(top_10_score=2.0, invalid_fraction=0.03, duplicate_fraction=0.10),
        CycleStats(top_10_score=3.0, invalid_fraction=0.02, duplicate_fraction=0.08),
    ]
    decision = evaluate_stop(clean_history)
    assert decision.should_stop is False, f"Clean history should not stop: {decision.detail}"
    assert decision.reason == "none"
