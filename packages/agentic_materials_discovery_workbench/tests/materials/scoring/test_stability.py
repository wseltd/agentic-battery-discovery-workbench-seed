"""Tests for ML-relaxation stability scoring."""

import pytest

from agentic_materials_discovery.scoring.stability import (
    DEFAULT_THRESHOLD_EV,
    NONCONVERGED_PENALTY_FACTOR,
    stability_score,
)


# ---------------------------------------------------------------------------
# Core linear decay behaviour
# ---------------------------------------------------------------------------


def test_zero_energy_scores_one():
    """Zero energy per atom is perfectly stable — score 1.0."""
    assert stability_score(0.0, converged=True) == 1.0


def test_at_threshold_scores_zero():
    """Energy equal to threshold scores exactly 0.0."""
    assert stability_score(DEFAULT_THRESHOLD_EV, converged=True) == 0.0


def test_above_threshold_scores_zero():
    """Energy above threshold is clamped to 0.0, not negative."""
    assert stability_score(DEFAULT_THRESHOLD_EV * 3, converged=True) == 0.0


def test_half_threshold_scores_half():
    """Halfway to threshold gives linear midpoint 0.5."""
    score = stability_score(DEFAULT_THRESHOLD_EV / 2, converged=True)
    assert score == pytest.approx(0.5)


def test_linear_decay_intermediate():
    """Verify linear interpolation at an arbitrary point."""
    energy = 0.03
    expected = 1.0 - energy / DEFAULT_THRESHOLD_EV
    assert stability_score(energy, converged=True) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Non-converged penalty
# ---------------------------------------------------------------------------


def test_nonconverged_penalty_at_zero_energy():
    """Non-converged with zero energy gets 1.0 * 0.5 = 0.5."""
    score = stability_score(0.0, converged=False)
    assert score == pytest.approx(NONCONVERGED_PENALTY_FACTOR)


def test_nonconverged_penalty_scales_with_base():
    """Non-converged penalty is proportional to the converged score."""
    energy = DEFAULT_THRESHOLD_EV / 2  # converged score = 0.5
    converged_score = stability_score(energy, converged=True)
    nonconverged_score = stability_score(energy, converged=False)
    assert nonconverged_score == pytest.approx(
        converged_score * NONCONVERGED_PENALTY_FACTOR
    )


def test_nonconverged_above_threshold_still_zero():
    """Non-converged with energy above threshold is still 0.0."""
    score = stability_score(DEFAULT_THRESHOLD_EV * 2, converged=False)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_negative_energy_clamped_to_one():
    """Negative energy per atom (very stable) is clamped to 1.0."""
    score = stability_score(-0.5, converged=True)
    assert score == 1.0


def test_negative_energy_nonconverged_still_clamped():
    """Very negative energy overwhelms the 0.5x penalty — clamp wins."""
    # Base = max(0, 1 - (-0.5)/0.1) = 6.0, * 0.5 = 3.0, clamp -> 1.0
    score = stability_score(-0.5, converged=False)
    assert score == 1.0

    # Mildly negative: base = max(0, 1 - (-0.04)/0.1) = 1.4, * 0.5 = 0.7
    mild = stability_score(-0.04, converged=False)
    assert mild == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


def test_custom_threshold():
    """Custom threshold changes the normalisation denominator."""
    # With threshold=0.2, energy=0.1 gives score = 1 - 0.1/0.2 = 0.5
    score = stability_score(0.1, converged=True, threshold=0.2)
    assert score == pytest.approx(0.5)


def test_custom_threshold_nonconverged():
    """Custom threshold works with non-converged penalty."""
    score = stability_score(0.1, converged=False, threshold=0.2)
    assert score == pytest.approx(0.5 * NONCONVERGED_PENALTY_FACTOR)


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def test_nonpositive_threshold_raises():
    """Zero or negative threshold raises ValueError."""
    with pytest.raises(ValueError, match="threshold must be positive") as exc_info:
        stability_score(0.05, converged=True, threshold=0.0)
    assert "threshold must be positive" in str(exc_info.value)

    with pytest.raises(ValueError, match="threshold must be positive") as exc_info:
        stability_score(0.05, converged=True, threshold=-0.1)
    assert "threshold must be positive" in str(exc_info.value)
