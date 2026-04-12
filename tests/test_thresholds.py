"""Tests for apply_thresholds — the pure threshold-gating function.

Threshold boundaries are the core risk surface: off-by-one in >= vs >
silently misroutes requests.  Heavy coverage on exact boundary values.
"""

from __future__ import annotations

from discovery_workbench.routing.thresholds import apply_thresholds


class TestUnsupportedFlag:
    """is_unsupported takes highest priority."""

    def test_unsupported_overrides_high_confidence(self) -> None:
        """Even with confidence=1.0, unsupported flag wins."""
        assert apply_thresholds(1.0, is_ambiguous=False, is_unsupported=True) == "unsupported"

    def test_unsupported_overrides_ambiguous(self) -> None:
        """Unsupported beats ambiguous when both are set."""
        assert apply_thresholds(0.90, is_ambiguous=True, is_unsupported=True) == "unsupported"

    def test_unsupported_at_zero_confidence(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=False, is_unsupported=True) == "unsupported"


class TestAmbiguousFlag:
    """is_ambiguous forces clarify regardless of confidence."""

    def test_ambiguous_high_confidence(self) -> None:
        """Ambiguity forces clarify even at confidence=1.0."""
        assert apply_thresholds(1.0, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_low_confidence(self) -> None:
        """Ambiguity forces clarify even below clarify threshold."""
        assert apply_thresholds(0.10, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_at_auto_boundary(self) -> None:
        """Ambiguity overrides auto threshold."""
        assert apply_thresholds(0.80, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_at_zero(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=True, is_unsupported=False) == "clarify"


class TestAutoThresholdBoundary:
    """Exact boundary tests at CONFIDENCE_AUTO_THRESHOLD = 0.80."""

    def test_exactly_at_auto_threshold(self) -> None:
        """0.80 → auto (boundary is inclusive: >=)."""
        assert apply_thresholds(0.80, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_just_below_auto_threshold(self) -> None:
        """0.7999 → clarify (just under the auto boundary)."""
        assert apply_thresholds(0.7999, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_just_above_auto_threshold(self) -> None:
        """0.8001 → auto."""
        assert apply_thresholds(0.8001, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_well_above_auto(self) -> None:
        assert apply_thresholds(0.95, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_confidence_one(self) -> None:
        assert apply_thresholds(1.0, is_ambiguous=False, is_unsupported=False) == "auto"


class TestClarifyThresholdBoundary:
    """Exact boundary tests at CONFIDENCE_CLARIFY_THRESHOLD = 0.55."""

    def test_exactly_at_clarify_threshold(self) -> None:
        """0.55 → clarify (boundary is inclusive: >=)."""
        assert apply_thresholds(0.55, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_just_below_clarify_threshold(self) -> None:
        """0.5499 → unsupported (just under the clarify boundary)."""
        assert apply_thresholds(0.5499, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_just_above_clarify_threshold(self) -> None:
        """0.5501 → clarify."""
        assert apply_thresholds(0.5501, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_midrange_clarify(self) -> None:
        """0.70 — clearly in clarify range."""
        assert apply_thresholds(0.70, is_ambiguous=False, is_unsupported=False) == "clarify"


class TestBelowClarify:
    """Confidence below clarify threshold → unsupported."""

    def test_low_confidence(self) -> None:
        assert apply_thresholds(0.10, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_zero_confidence(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_just_under_clarify(self) -> None:
        """0.54 → unsupported."""
        assert apply_thresholds(0.54, is_ambiguous=False, is_unsupported=False) == "unsupported"


class TestFlagCombinations:
    """Verify priority ordering when multiple conditions overlap."""

    def test_both_flags_false_high_confidence(self) -> None:
        """No flags, high confidence → auto."""
        assert apply_thresholds(0.90, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_both_flags_false_mid_confidence(self) -> None:
        """No flags, mid confidence → clarify."""
        assert apply_thresholds(0.60, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_both_flags_false_low_confidence(self) -> None:
        """No flags, low confidence → unsupported."""
        assert apply_thresholds(0.20, is_ambiguous=False, is_unsupported=False) == "unsupported"
