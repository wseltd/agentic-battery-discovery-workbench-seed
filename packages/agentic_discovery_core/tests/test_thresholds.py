"""Tests for apply_thresholds -- the pure threshold-gating function."""

from __future__ import annotations

from agentic_discovery_core.routing.thresholds import apply_thresholds


class TestUnsupportedFlag:
    def test_unsupported_overrides_high_confidence(self) -> None:
        assert apply_thresholds(1.0, is_ambiguous=False, is_unsupported=True) == "unsupported"

    def test_unsupported_overrides_ambiguous(self) -> None:
        assert apply_thresholds(0.90, is_ambiguous=True, is_unsupported=True) == "unsupported"

    def test_unsupported_at_zero_confidence(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=False, is_unsupported=True) == "unsupported"


class TestAmbiguousFlag:
    def test_ambiguous_high_confidence(self) -> None:
        assert apply_thresholds(1.0, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_low_confidence(self) -> None:
        assert apply_thresholds(0.10, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_at_auto_boundary(self) -> None:
        assert apply_thresholds(0.80, is_ambiguous=True, is_unsupported=False) == "clarify"

    def test_ambiguous_at_zero(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=True, is_unsupported=False) == "clarify"


class TestAutoThresholdBoundary:
    def test_exactly_at_auto_threshold(self) -> None:
        assert apply_thresholds(0.80, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_just_below_auto_threshold(self) -> None:
        assert apply_thresholds(0.7999, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_just_above_auto_threshold(self) -> None:
        assert apply_thresholds(0.8001, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_well_above_auto(self) -> None:
        assert apply_thresholds(0.95, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_confidence_one(self) -> None:
        assert apply_thresholds(1.0, is_ambiguous=False, is_unsupported=False) == "auto"


class TestClarifyThresholdBoundary:
    def test_exactly_at_clarify_threshold(self) -> None:
        assert apply_thresholds(0.55, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_just_below_clarify_threshold(self) -> None:
        assert apply_thresholds(0.5499, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_just_above_clarify_threshold(self) -> None:
        assert apply_thresholds(0.5501, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_midrange_clarify(self) -> None:
        assert apply_thresholds(0.70, is_ambiguous=False, is_unsupported=False) == "clarify"


class TestBelowClarify:
    def test_low_confidence(self) -> None:
        assert apply_thresholds(0.10, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_zero_confidence(self) -> None:
        assert apply_thresholds(0.0, is_ambiguous=False, is_unsupported=False) == "unsupported"

    def test_just_under_clarify(self) -> None:
        assert apply_thresholds(0.54, is_ambiguous=False, is_unsupported=False) == "unsupported"


class TestFlagCombinations:
    def test_both_flags_false_high_confidence(self) -> None:
        assert apply_thresholds(0.90, is_ambiguous=False, is_unsupported=False) == "auto"

    def test_both_flags_false_mid_confidence(self) -> None:
        assert apply_thresholds(0.60, is_ambiguous=False, is_unsupported=False) == "clarify"

    def test_both_flags_false_low_confidence(self) -> None:
        assert apply_thresholds(0.20, is_ambiguous=False, is_unsupported=False) == "unsupported"
