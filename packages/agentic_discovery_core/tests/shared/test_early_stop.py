"""Tests for early-stop evaluation logic."""

from agentic_discovery_core.shared.early_stop import (
    CycleStats,
    DUPLICATE_THRESHOLD,
    INVALIDITY_THRESHOLD,
    INVALIDITY_WINDOW,
    PLATEAU_THRESHOLD,
    PLATEAU_WINDOW,
    StopDecision,
    evaluate_stop,
)


def test_empty_history_no_stop():
    result = evaluate_stop([])
    assert not result.should_stop
    assert result.reason == "none"


def test_single_cycle_no_stop():
    result = evaluate_stop([CycleStats(0.5, 0.1, 0.1)])
    assert not result.should_stop
    assert result.reason == "none"


def test_plateau_detected_after_two_flat_cycles():
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.501, 0.1, 0.1),
        CycleStats(0.502, 0.1, 0.1),
    ]
    result = evaluate_stop(history)
    assert isinstance(result, StopDecision)
    assert result.should_stop
    assert result.reason == "plateau"
    assert "0.00" in result.detail


def test_plateau_not_triggered_with_sufficient_improvement():
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.501, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.1),
    ]
    result = evaluate_stop(history)
    assert not result.should_stop
    assert result.reason == "none"


def test_plateau_with_zero_initial_score():
    history = [
        CycleStats(0.0, 0.1, 0.1),
        CycleStats(0.0, 0.1, 0.1),
        CycleStats(0.0, 0.1, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "plateau"


def test_invalidity_spike_two_consecutive():
    low = CycleStats(0.5, 0.1, 0.1)
    high = CycleStats(0.6, 0.55, 0.1)
    history = [low] + [high] * INVALIDITY_WINDOW
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "invalidity_spike"


def test_invalidity_spike_not_triggered_once():
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.1),
        CycleStats(0.7, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.reason != "invalidity_spike"


def test_invalidity_spike_non_consecutive_no_stop():
    history = [
        CycleStats(0.5, 0.55, 0.1),
        CycleStats(0.6, 0.10, 0.1),
        CycleStats(0.7, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.reason != "invalidity_spike"


def test_duplicate_surge_single_batch():
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.75),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "duplicate_surge"


def test_duplicate_surge_below_threshold():
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, DUPLICATE_THRESHOLD - 0.01),
    ]
    result = evaluate_stop(history)
    assert not result.should_stop


def test_invalidity_takes_priority_over_plateau():
    history = [
        CycleStats(0.500, 0.55, 0.1),
        CycleStats(0.501, 0.55, 0.1),
        CycleStats(0.502, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "invalidity_spike"


def test_duplicate_surge_takes_priority_over_plateau():
    history = [
        CycleStats(0.500, 0.1, 0.1),
        CycleStats(0.501, 0.1, 0.1),
        CycleStats(0.502, 0.1, 0.75),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "duplicate_surge"


def test_boundary_values_at_exact_thresholds():
    history_inv = [
        CycleStats(0.5, INVALIDITY_THRESHOLD, 0.1),
        CycleStats(0.6, INVALIDITY_THRESHOLD, 0.1),
    ]
    assert evaluate_stop(history_inv).reason == "invalidity_spike"

    history_dup = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, DUPLICATE_THRESHOLD),
    ]
    assert evaluate_stop(history_dup).reason == "duplicate_surge"

    base = 1.0
    step = base * PLATEAU_THRESHOLD
    history_plat = [
        CycleStats(base, 0.1, 0.1),
        CycleStats(base + step, 0.1, 0.1),
        CycleStats(base + 2 * step, 0.1, 0.1),
    ]
    result = evaluate_stop(history_plat)
    assert result.reason != "plateau"

    assert PLATEAU_WINDOW == 2
    assert INVALIDITY_WINDOW == 2
