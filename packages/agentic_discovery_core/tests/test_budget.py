"""Tests for agentic_discovery_core.budget -- budget tracking and early-stop detection."""

from __future__ import annotations

from agentic_discovery_core.budget import (
    DUPLICATE_SURGE_THRESHOLD,
    INVALIDITY_CONSECUTIVE_BATCHES,
    INVALIDITY_SPIKE_THRESHOLD,
    PLATEAU_CONSECUTIVE_CYCLES,
    PLATEAU_IMPROVEMENT_THRESHOLD,
    BudgetConfig,
    BudgetController,
    BudgetState,
)


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


class TestDataclasses:
    def test_default_config_matches_spec(self):
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
        bc.record_batch(1.0, 0.0)
        bc.record_batch(1.0, 0.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "batches" in reason


class TestPlateau:
    def test_plateau_detection_triggers_stop(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "plateau" in reason

    def test_plateau_not_triggered_on_first_cycle(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_plateau_resets_on_improvement(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_at_threshold_boundary_plateau(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(0.0)
        bc.record_cycle(0.01)
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_just_above_threshold_no_plateau(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.011)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_three_cycle_plateau(self):
        bc = BudgetController(BudgetConfig(max_cycles=10))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)
        bc.record_cycle(1.008)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "plateau" in reason

    def test_cycle_score_history_is_trimmed(self):
        bc = BudgetController(BudgetConfig(max_cycles=100))
        max_len = PLATEAU_CONSECUTIVE_CYCLES + 1
        for i in range(max_len + 5):
            bc.record_cycle(float(i))
        assert len(bc._cycle_scores) == max_len


class TestInvaliditySpike:
    def test_invalidity_spike_two_consecutive(self):
        bc = BudgetController()
        bc.record_batch(0.3, 0.0)
        bc.record_batch(0.3, 0.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "invalidity" in reason

    def test_invalidity_spike_not_triggered_once(self):
        bc = BudgetController()
        bc.record_batch(0.3, 0.0)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_invalidity_resets_on_good_batch(self):
        bc = BudgetController()
        bc.record_batch(0.3, 0.0)
        bc.record_batch(0.8, 0.0)
        bc.record_batch(0.3, 0.0)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_at_threshold_boundary_invalidity(self):
        bc = BudgetController()
        bc.record_batch(0.5, 0.0)
        bc.record_batch(0.5, 0.0)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "invalidity" in reason

    def test_just_below_threshold_no_invalidity(self):
        bc = BudgetController()
        bc.record_batch(0.51, 0.0)
        bc.record_batch(0.51, 0.0)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_invalidity_history_is_trimmed(self):
        bc = BudgetController(BudgetConfig(max_batches=100))
        max_len = INVALIDITY_CONSECUTIVE_BATCHES
        for _ in range(max_len + 5):
            bc.record_batch(0.8, 0.0)
        assert len(bc._invalidity_fractions) == max_len


class TestDuplicateSurge:
    def test_duplicate_surge_single_batch(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.8)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "duplicate" in reason

    def test_duplicate_surge_below_threshold(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.6)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_at_threshold_boundary_duplicate(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.7)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "duplicate" in reason

    def test_just_below_threshold_no_duplicate(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.69)
        stop, reason = bc.should_stop()
        assert stop is False

    def test_duplicate_surge_sets_stopped_eagerly(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.8)
        assert bc.state.stopped is True
        assert "duplicate" in bc.state.stop_reason


class TestShouldStop:
    def test_should_stop_returns_none_when_ok(self):
        bc = BudgetController()
        stop, reason = bc.should_stop()
        assert stop is False
        assert reason is None

    def test_multiple_stop_conditions_first_wins(self):
        bc = BudgetController(BudgetConfig(max_cycles=2, max_batches=100))
        bc.record_cycle(1.0)
        bc.record_cycle(1.005)
        stop, reason = bc.should_stop()
        assert stop is True
        assert "exhausted" in reason

    def test_stop_is_sticky(self):
        bc = BudgetController()
        bc.record_batch(1.0, 0.8)
        stop1, reason1 = bc.should_stop()
        assert stop1 is True
        bc.record_batch(1.0, 0.0)
        stop2, reason2 = bc.should_stop()
        assert stop2 is True
        assert reason2 == reason1


class TestRemaining:
    def test_remaining_decrements(self):
        bc = BudgetController(BudgetConfig(max_cycles=5, max_batches=8))
        assert bc.remaining() == {"cycles_remaining": 5, "batches_remaining": 8}
        bc.record_cycle(1.0)
        bc.record_batch(1.0, 0.0)
        assert bc.remaining() == {"cycles_remaining": 4, "batches_remaining": 7}

    def test_remaining_floors_at_zero(self):
        bc = BudgetController(BudgetConfig(max_cycles=1, max_batches=1))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)
        bc.record_batch(1.0, 0.0)
        bc.record_batch(1.0, 0.0)
        remaining = bc.remaining()
        assert remaining["cycles_remaining"] == 0
        assert remaining["batches_remaining"] == 0

    def test_remaining_after_exhaustion(self):
        bc = BudgetController(BudgetConfig(max_cycles=2, max_batches=8))
        bc.record_cycle(1.0)
        bc.record_cycle(2.0)
        remaining = bc.remaining()
        assert remaining["cycles_remaining"] == 0
        assert remaining["batches_remaining"] == 8

    def test_remaining_independent_of_stop(self):
        bc = BudgetController(BudgetConfig(max_cycles=5, max_batches=8))
        bc.record_batch(1.0, 0.8)
        assert bc.state.stopped is True
        remaining = bc.remaining()
        assert remaining["cycles_remaining"] == 5
        assert remaining["batches_remaining"] == 7
