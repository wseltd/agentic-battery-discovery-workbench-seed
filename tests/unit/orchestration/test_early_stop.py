"""Tests for orchestration-level early-stop re-exports.

Verifies that evaluate_stop is accessible through the orchestration
namespace and produces correct decisions.  The exhaustive edge-case
coverage lives in tests/shared/test_early_stop.py; these tests confirm
the re-export wiring and exercise priority ordering through the
orchestration import path.
"""

from agentic_workbench.orchestration.early_stop import (
    DUPLICATE_THRESHOLD,
    INVALIDITY_THRESHOLD,
    INVALIDITY_WINDOW,
    PLATEAU_THRESHOLD,
    PLATEAU_WINDOW,
    CycleStats,
    StopDecision,
    evaluate_stop,
)


def test_empty_history_no_stop():
    """Empty history has no trend data — should never stop."""
    result = evaluate_stop([])
    assert not result.should_stop
    assert result.reason == "none"


def test_single_cycle_no_stop():
    """A single cycle cannot show a trend — no stop."""
    result = evaluate_stop([CycleStats(0.5, 0.1, 0.1)])
    assert not result.should_stop
    assert result.reason == "none"


def test_plateau_detected_after_two_flat_cycles():
    """Plateau fires when PLATEAU_WINDOW transitions show < PLATEAU_THRESHOLD improvement."""
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
    """If any transition within the window improves enough, no plateau."""
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.501, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.1),
    ]
    result = evaluate_stop(history)
    assert not result.should_stop
    assert result.reason == "none"


def test_plateau_with_zero_initial_score():
    """Zero base score: stuck at zero is a plateau."""
    history = [
        CycleStats(0.0, 0.1, 0.1),
        CycleStats(0.0, 0.1, 0.1),
        CycleStats(0.0, 0.1, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "plateau"


def test_invalidity_spike_two_consecutive():
    """INVALIDITY_WINDOW consecutive cycles above threshold triggers stop."""
    low = CycleStats(0.5, 0.1, 0.1)
    high = CycleStats(0.6, 0.55, 0.1)
    history = [low] + [high] * INVALIDITY_WINDOW
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "invalidity_spike"


def test_invalidity_spike_not_triggered_once():
    """A single cycle above invalidity threshold is insufficient."""
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.1),
        CycleStats(0.7, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.reason != "invalidity_spike"


def test_invalidity_spike_non_consecutive_no_stop():
    """High invalidity in non-adjacent cycles does not trigger the spike."""
    history = [
        CycleStats(0.5, 0.55, 0.1),
        CycleStats(0.6, 0.10, 0.1),
        CycleStats(0.7, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.reason != "invalidity_spike"


def test_duplicate_surge_single_batch():
    """A single cycle above DUPLICATE_THRESHOLD triggers immediate stop."""
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, 0.75),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "duplicate_surge"


def test_duplicate_surge_below_threshold():
    """Duplicate fraction just below threshold does not trigger stop."""
    history = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, DUPLICATE_THRESHOLD - 0.01),
    ]
    result = evaluate_stop(history)
    assert not result.should_stop


def test_invalidity_takes_priority_over_plateau():
    """When both invalidity spike and plateau conditions are met, invalidity wins."""
    history = [
        CycleStats(0.500, 0.55, 0.1),
        CycleStats(0.501, 0.55, 0.1),
        CycleStats(0.502, 0.55, 0.1),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "invalidity_spike"


def test_duplicate_surge_takes_priority_over_plateau():
    """When both duplicate surge and plateau conditions are met, duplicate wins."""
    history = [
        CycleStats(0.500, 0.1, 0.1),
        CycleStats(0.501, 0.1, 0.1),
        CycleStats(0.502, 0.1, 0.75),
    ]
    result = evaluate_stop(history)
    assert result.should_stop
    assert result.reason == "duplicate_surge"


def test_boundary_values_at_exact_thresholds():
    """Verify >= semantics: exact threshold values trigger the condition."""
    # Invalidity at exact threshold
    history_inv = [
        CycleStats(0.5, INVALIDITY_THRESHOLD, 0.1),
        CycleStats(0.6, INVALIDITY_THRESHOLD, 0.1),
    ]
    assert evaluate_stop(history_inv).reason == "invalidity_spike"

    # Duplicate at exact threshold
    history_dup = [
        CycleStats(0.5, 0.1, 0.1),
        CycleStats(0.6, 0.1, DUPLICATE_THRESHOLD),
    ]
    assert evaluate_stop(history_dup).reason == "duplicate_surge"

    # Plateau: improvement exactly at threshold should NOT trigger
    # (condition is strictly-less-than)
    base = 1.0
    step = base * PLATEAU_THRESHOLD
    history_plat = [
        CycleStats(base, 0.1, 0.1),
        CycleStats(base + step, 0.1, 0.1),
        CycleStats(base + 2 * step, 0.1, 0.1),
    ]
    result = evaluate_stop(history_plat)
    assert result.reason != "plateau"

    # Verify the window constants have expected values
    assert PLATEAU_WINDOW == 2
    assert INVALIDITY_WINDOW == 2
