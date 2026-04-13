"""Tests for compute_confidence -- the keyword-density ratio function."""

from __future__ import annotations

from agentic_discovery_core.routing.confidence import compute_confidence


class TestZeroDenominator:
    def test_zero_tokens_returns_zero(self) -> None:
        assert compute_confidence(0, 0) == 0.0

    def test_zero_tokens_with_nonzero_hits_returns_zero(self) -> None:
        assert compute_confidence(5, 0) == 0.0


class TestExactRatios:
    def test_return_type_is_float(self) -> None:
        result = compute_confidence(3, 10)
        assert isinstance(result, float)
        assert result == 0.3

    def test_half_match(self) -> None:
        assert compute_confidence(5, 10) == 0.5

    def test_full_match(self) -> None:
        assert compute_confidence(10, 10) == 1.0

    def test_no_hits(self) -> None:
        assert compute_confidence(0, 10) == 0.0

    def test_one_out_of_three(self) -> None:
        result = compute_confidence(1, 3)
        assert abs(result - 1 / 3) < 1e-12

    def test_single_hit_single_token(self) -> None:
        assert compute_confidence(1, 1) == 1.0


class TestClamping:
    def test_clamp_above_one(self) -> None:
        result = compute_confidence(15, 10)
        assert result == 1.0

    def test_clamp_preserves_normal_values(self) -> None:
        assert compute_confidence(3, 10) == 0.3

    def test_negative_hits_clamped_to_zero(self) -> None:
        result = compute_confidence(-1, 10)
        assert result == 0.0


class TestPurity:
    def test_repeated_calls_same_result(self) -> None:
        results = [compute_confidence(7, 20) for _ in range(100)]
        assert all(r == results[0] for r in results)
        assert results[0] == 0.35
