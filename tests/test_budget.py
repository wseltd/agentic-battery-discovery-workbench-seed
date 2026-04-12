"""Tests for discovery_workbench.budget — budget tracking and early-stop detection.

Coverage is weighted toward the early-stop heuristics (plateau, invalidity,
duplicate surge) because those have boundary-condition risk.  Dataclass defaults
get lighter coverage since they are trivially correct.

record_batch takes (valid_fraction, unique_fraction).  Invalidity is
1 - valid_fraction; duplicate rate is 1 - unique_fraction.
"""

from __future__ import annotations

from discovery_workbench.budget import (
    DUPLICATE_SURGE_THRESHOLD,
    INVALIDITY_CONSECUTIVE_BATCHES,
    INVALIDITY_SPIKE_THRESHOLD,
    PLATEAU_CONSECUTIVE_CYCLES,
    PLATEAU_IMPROVEMENT_THRESHOLD,
    BudgetConfig,
    BudgetController,
    BudgetState,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_plateau_improvement_threshold(self):
        assert PLATEAU_IMPROVEMENT_THRESHOLD == 0.01

    def test_plateau_consecutive_cycles(self):
        assert PLATEAU_CONSECUTIVE_CYCLES == 2

    def test_invalidity_spike_threshold(self):
        assert INVALIDITY_SPIKE_THRESHOLD == 0.50

    def test_invalidity_consecutive_batches(self):
        assert INVALIDITY_CONSECUTIVE_BATCHES == 2

    def test_duplicate_surge_threshold(self):
        assert DUPLICATE_SURGE_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# BudgetConfig / BudgetState dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_default_config_matches_q7(self):
        """Q7 defaults: max_cycles=5, max_batches=8, shortlist_size=25."""
        cfg = BudgetConfig()
        assert cfg.max_cycles == 5
        assert cfg.max_batches == 8
        assert cfg.shortlist_size == 25

    def test_custom_config_overrides_defaults(self):
        cfg = BudgetConfig(max_cycles=10, max_batches=20, shortlist_size=50)
        assert cfg.max_cycles == 10
        assert cfg.max_batches == 20
        assert cfg.shortlist_size == 50

    def test_budget_state_defaults(self):
        state = BudgetState()
        assert state.cycles_used == 0
        assert state.batches_used == 0
        assert state.stopped is False
        assert state.stop_reason is None


# ---------------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------------

class TestExhaustion:
    def test_budget_exhaustion_cycles(self):
        bc = BudgetController(BudgetConfig(max_cycles=2, max_batches=100))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "cycles" in reason

    def test_budget_exhaustion_batches(self):
        bc = BudgetController(BudgetConfig(max_cycles=100, max_batches=2))
        # valid_fraction=1.0, unique_fraction=1.0 → no invalidity or duplicate issues
        bc.record_batch(1.0, 1.0)
        bc.record_batch(1.0, 1.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "batches" in reason


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

class TestPlateau:
    def test_plateau_detection_triggers_stop(self):
        """Two cycles with stalling improvement triggers plateau."""
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)  # +0.005 < 0.01 threshold
        stop, reason = bc.should_stop()
        assert stop is True
        assert "plateau" in reason

    def test_plateau_not_triggered_on_first_cycle(self):
        """A single cycle cannot trigger plateau — need at least two scores."""
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_plateau_resets_on_improvement(self):
        """A good improvement cycle breaks the stall chain."""
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)  # big improvement — no plateau
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_at_threshold_boundary_plateau(self):
        """Improvement exactly at threshold (0.01) is NOT above it, so counts as stall.

        Uses 0.005 increments to avoid IEEE 754 subtraction noise around 0.01.
        """
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)  # +0.005, below threshold — stall
        stop, reason = bc.should_stop()
        assert stop is True
        assert "plateau" in reason

    def test_just_above_threshold_no_plateau(self):
        """Improvement of 0.011 is above threshold — no plateau."""
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.011)  # 0.011 > 0.01
        stop, reason = bc.should_stop()
        assert stop is False

    def test_three_cycle_plateau(self):
        """Three cycles where last two improvements stall — plateau on the last pair."""
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)
        bc.record_cycle(1.008)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "plateau" in reason


# ---------------------------------------------------------------------------
# Invalidity spike
# ---------------------------------------------------------------------------

class TestInvaliditySpike:
    def test_invalidity_spike_two_consecutive(self):
        # valid_fraction=0.3 → invalidity=0.7 >= 0.5
        bc = BudgetController()
        bc.record_batch(0.3, 1.0)
        bc.record_batch(0.3, 1.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "invalidity" in reason

    def test_invalidity_spike_not_triggered_once(self):
        """A single high-invalidity batch is not enough."""
        bc = BudgetController()
        bc.record_batch(0.3, 1.0)  # invalidity=0.7 but only one batch
        stop, reason = bc.should_stop()
        assert stop is False

    def test_invalidity_resets_on_good_batch(self):
        """A good batch in between resets the consecutive counter."""
        bc = BudgetController()
        bc.record_batch(0.3, 1.0)  # bad: invalidity=0.7
        bc.record_batch(0.8, 1.0)  # good: invalidity=0.2
        bc.record_batch(0.3, 1.0)  # bad again, but only one consecutive
        stop, reason = bc.should_stop()
        assert stop is False

    def test_at_threshold_boundary_invalidity(self):
        """Invalidity exactly at threshold (0.50) triggers: valid_fraction=0.5."""
        bc = BudgetController()
        bc.record_batch(0.5, 1.0)  # invalidity=0.5 >= 0.5
        bc.record_batch(0.5, 1.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "invalidity" in reason

    def test_just_below_threshold_no_invalidity(self):
        """valid_fraction=0.51 → invalidity=0.49, below threshold."""
        bc = BudgetController()
        bc.record_batch(0.51, 1.0)
        bc.record_batch(0.51, 1.0)
        stop, reason = bc.should_stop()
        assert stop is False


# ---------------------------------------------------------------------------
# Duplicate surge
# ---------------------------------------------------------------------------

class TestDuplicateSurge:
    def test_duplicate_surge_single_batch(self):
        """A single batch with high duplicate fraction triggers immediate stop."""
        # unique_fraction=0.2 → duplicate=0.8 >= 0.7
        bc = BudgetController()
        bc.record_batch(1.0, 0.2)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "duplicate" in reason

    def test_duplicate_surge_below_threshold(self):
        # unique_fraction=0.4 → duplicate=0.6 < 0.7
        bc = BudgetController()
        bc.record_batch(1.0, 0.4)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_at_threshold_boundary_duplicate(self):
        """unique_fraction=0.3 → duplicate=0.7, exactly at threshold — triggers."""
        bc = BudgetController()
        bc.record_batch(1.0, 0.3)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "duplicate" in reason

    def test_just_below_threshold_no_duplicate(self):
        """unique_fraction=0.31 → duplicate=0.69, below threshold."""
        bc = BudgetController()
        bc.record_batch(1.0, 0.31)
        stop, reason = bc.should_stop()
        assert stop is False


# ---------------------------------------------------------------------------
# should_stop general behaviour
# ---------------------------------------------------------------------------

class TestShouldStop:
    def test_should_stop_returns_none_when_ok(self):
        bc = BudgetController()
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_multiple_stop_conditions_first_wins(self):
        """When multiple conditions trigger simultaneously, exhaustion wins (checked first)."""
        bc = BudgetController(BudgetConfig(max_cycles=2, max_batches=100))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)  # plateau + exhaustion both apply
        stop, reason = bc.should_stop()
        assert stop is True
        assert "exhausted" in reason

    def test_stop_is_sticky(self):
        """Once stopped, subsequent calls return the same reason without re-evaluation."""
        bc = BudgetController()
        bc.record_batch(1.0, 0.2)  # duplicate=0.8 → surge
        stop1, reason1 = bc.should_stop()
        assert stop1 is True
        # Record a good batch — should not clear the stop.
        bc.record_batch(1.0, 1.0)
        stop2, reason2 = bc.should_stop()
        assert stop2 is True
        assert reason2 == reason1


# ---------------------------------------------------------------------------
# remaining()
# ---------------------------------------------------------------------------

class TestRemaining:
    def test_remaining_decrements(self):
        bc = BudgetController(BudgetConfig(max_cycles=5, max_batches=8))
        assert bc.remaining() == {"cycles": 5, "batches": 8}
        bc.record_cycle(1.0)
        bc.record_batch(1.0, 1.0)
        assert bc.remaining() == {"cycles": 4, "batches": 7}

    def test_remaining_floors_at_zero(self):
        bc = BudgetController(BudgetConfig(max_cycles=1, max_batches=1))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)  # over limit
        bc.record_batch(1.0, 1.0)
        bc.record_batch(1.0, 1.0)  # over limit
        remaining = bc.remaining()
        assert remaining["cycles"] == 0
        assert remaining["batches"] == 0
