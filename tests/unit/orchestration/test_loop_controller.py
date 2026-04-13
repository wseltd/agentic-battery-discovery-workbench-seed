"""Tests for the orchestration-level loop controller.

Heaviest coverage on: budget-before-generate ordering, cross-cycle
deduplication via deduplicate_across_cycles, and early-stop evaluator
integration — these are the orchestration concerns that differ from the
lower-level agent_loop module.
"""

from __future__ import annotations

from typing import Any

import pytest

from discovery_workbench.budget import BudgetConfig, BudgetController
from discovery_workbench.shared.agent_loop import (
    CycleResult,
    DiscoveryPipeline,
)

from agentic_workbench.orchestration.loop_controller import (
    AgentLoopController,
    LoopRequest,
)


# ---------------------------------------------------------------------------
# Controllable mock pipeline
# ---------------------------------------------------------------------------


class _MockPipeline:
    """Pipeline with preconfigured per-cycle results.

    Args:
        cycle_candidates: List of candidate lists, one per cycle.
            Cycles beyond the list length reuse the last entry.
        valid_fraction: Fraction that passes validation (sliced).
        score_value: Base score for all candidates.
        score_increment: Added to base score per cycle for non-plateau runs.
        filter_pass: Whether filter_candidates passes everything through.
    """

    def __init__(
        self,
        cycle_candidates: list[list[str]] | None = None,
        valid_fraction: float = 1.0,
        score_value: float = 0.5,
        score_increment: float = 0.0,
        filter_pass: bool = True,
    ) -> None:
        self._cycle_candidates = cycle_candidates or [["mol_a", "mol_b", "mol_c"]]
        self._valid_fraction = valid_fraction
        self._score_value = score_value
        self._score_increment = score_increment
        self._filter_pass = filter_pass
        self._call_count = 0

    def generate(self, batch_size: int) -> list[Any]:
        idx = min(self._call_count, len(self._cycle_candidates) - 1)
        self._call_count += 1
        return self._cycle_candidates[idx][:batch_size]

    def validate(self, candidates: list[Any]) -> list[Any]:
        if self._valid_fraction == 0.0:
            return []
        keep = max(1, int(len(candidates) * self._valid_fraction))
        return candidates[:keep]

    def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
        cycle_score = self._score_value + (self._call_count - 1) * self._score_increment
        return [(c, cycle_score) for c in valid]

    def filter_candidates(
        self, scored: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        if self._filter_pass:
            return scored
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget(
    max_cycles: int = 5,
    max_batches: int = 8,
    shortlist_size: int = 25,
) -> BudgetController:
    return BudgetController(
        BudgetConfig(
            max_cycles=max_cycles,
            max_batches=max_batches,
            shortlist_size=shortlist_size,
        ),
    )


def _default_request(
    batch_size: int = 10,
    output_count: int = 25,
    evidence_level: str = "generated",
) -> LoopRequest:
    return LoopRequest(
        batch_size=batch_size,
        output_count=output_count,
        evidence_level=evidence_level,
    )


# ---------------------------------------------------------------------------
# Core loop behaviour
# ---------------------------------------------------------------------------


class TestSingleCycle:
    def test_single_cycle_produces_shortlist(self) -> None:
        """One cycle with valid candidates produces a non-empty shortlist."""
        pipeline = _MockPipeline(
            cycle_candidates=[["a", "b", "c"]],
            score_value=0.8,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=1),
        )

        assert result.cycles_run == 1
        assert len(result.shortlist) > 0
        assert "budget_exhausted" in result.stopped_reason


class TestAccumulation:
    def test_accumulation_across_multiple_cycles(self) -> None:
        """Results from all cycles accumulate into the final shortlist."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                ["cycle1_a", "cycle1_b"],
                ["cycle2_a", "cycle2_b"],
                ["cycle3_a", "cycle3_b"],
            ],
            score_value=0.5,
            score_increment=0.1,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(output_count=10),
            pipeline,
            _make_budget(max_cycles=3),
        )

        assert result.cycles_run == 3
        assert len(result.shortlist) == 6
        assert "cycle1_a" in result.shortlist
        assert "cycle2_a" in result.shortlist
        assert "cycle3_a" in result.shortlist


class TestShortlist:
    def test_shortlist_respects_output_count(self) -> None:
        """Shortlist is truncated to output_count even when more candidates exist."""
        pipeline = _MockPipeline(
            cycle_candidates=[["a", "b", "c", "d", "e"]],
            score_value=0.5,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(batch_size=10, output_count=3),
            pipeline,
            _make_budget(max_cycles=1),
        )

        assert len(result.shortlist) <= 3


# ---------------------------------------------------------------------------
# Budget enforcement — checked BEFORE generate
# ---------------------------------------------------------------------------


class TestBudgetMaxCycles:
    def test_budget_stops_loop_at_max_cycles(self) -> None:
        """Loop runs exactly max_cycles before budget exhaustion stops it."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(10)
            ],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=3, max_batches=20),
        )

        assert result.cycles_run == 3
        assert "budget_exhausted" in result.stopped_reason
        assert "cycles" in result.stopped_reason


class TestBudgetMaxBatches:
    def test_budget_stops_loop_at_max_batches(self) -> None:
        """Loop stops when batch limit is reached before cycle limit."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(10)
            ],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=10, max_batches=2),
        )

        assert result.cycles_run == 2
        assert "budget_exhausted" in result.stopped_reason
        assert "batches" in result.stopped_reason


class TestBudgetBeforeGenerate:
    def test_budget_checked_before_first_generate(self) -> None:
        """When budget is already exhausted, no generate call is made."""
        call_tracker: list[str] = []

        class _TrackingPipeline:
            def generate(self, batch_size: int) -> list[Any]:
                call_tracker.append("generate")
                return ["x"]

            def validate(self, candidates: list[Any]) -> list[Any]:
                return candidates

            def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
                return [(c, 0.5) for c in valid]

            def filter_candidates(
                self, scored: list[tuple[Any, float]],
            ) -> list[tuple[Any, float]]:
                return scored

        budget = _make_budget(max_cycles=0)
        ctrl = AgentLoopController()
        result = ctrl.run(_default_request(), _TrackingPipeline(), budget)

        assert result.cycles_run == 0
        assert "generate" not in call_tracker


# ---------------------------------------------------------------------------
# Early-stop evaluator
# ---------------------------------------------------------------------------


class _CountingEarlyStop:
    """Early-stop evaluator that triggers after N cycles."""

    def __init__(self, trigger_after: int) -> None:
        self._trigger_after = trigger_after
        self._calls = 0

    def should_stop(
        self,
        cycle_result: CycleResult,
        accumulated: list[tuple[Any, float]],
    ) -> bool:
        self._calls += 1
        return self._calls >= self._trigger_after


class TestEarlyStopEvaluator:
    def test_early_stop_halts_loop(self) -> None:
        """Early-stop evaluator triggers after the configured cycle count."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}"] for i in range(10)
            ],
            score_value=1.0,
            score_increment=1.0,
        )
        evaluator = _CountingEarlyStop(trigger_after=2)
        ctrl = AgentLoopController(early_stop_evaluator=evaluator)
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=10, max_batches=20),
        )

        assert result.cycles_run == 2
        assert result.stopped_reason == "early_stop"

    def test_no_early_stop_without_evaluator(self) -> None:
        """Without an evaluator, loop runs until budget exhaustion."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(5)
            ],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController(early_stop_evaluator=None)
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=3, max_batches=20),
        )

        assert result.cycles_run == 3
        assert "budget_exhausted" in result.stopped_reason

    def test_early_stop_receives_cycle_result_and_accumulated(self) -> None:
        """The evaluator receives the correct cycle_result and accumulated state."""
        captured: list[tuple[CycleResult, list]] = []

        class _CapturingEvaluator:
            def should_stop(
                self,
                cycle_result: CycleResult,
                accumulated: list[tuple[Any, float]],
            ) -> bool:
                captured.append((cycle_result, list(accumulated)))
                return len(captured) >= 2

        pipeline = _MockPipeline(
            cycle_candidates=[["a", "b"], ["c", "d"]],
            score_value=0.7,
            score_increment=0.1,
        )
        ctrl = AgentLoopController(early_stop_evaluator=_CapturingEvaluator())
        ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=5, max_batches=20),
        )

        assert len(captured) == 2
        # First call: cycle 1 result, accumulated has cycle 1 items
        cr1, acc1 = captured[0]
        assert cr1.cycle_number == 1
        assert len(acc1) == 2

        # Second call: cycle 2 result, accumulated has both cycles' items
        cr2, acc2 = captured[1]
        assert cr2.cycle_number == 2
        assert len(acc2) == 4


# ---------------------------------------------------------------------------
# Early-stop via budget heuristics (plateau, invalidity, duplicate surge)
# ---------------------------------------------------------------------------


class TestEarlyStopPlateau:
    def test_early_stop_halts_on_plateau(self) -> None:
        """Plateau detection stops loop when scores stop improving."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(10)
            ],
            score_value=0.5,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=10, max_batches=20),
        )

        assert result.cycles_run == 2
        assert "plateau" in result.stopped_reason


class TestEarlyStopInvaliditySpike:
    def test_early_stop_halts_on_invalidity_spike(self) -> None:
        """Invalidity spike stops loop after consecutive high-invalidity batches."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(4)] for i in range(10)
            ],
            valid_fraction=0.25,
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=10, max_batches=20),
        )

        assert result.cycles_run == 2
        assert "invalidity_spike" in result.stopped_reason


class TestEarlyStopDuplicateSurge:
    def test_early_stop_halts_on_duplicate_surge(self) -> None:
        """Duplicate surge stops loop when most generated candidates are repeats.

        Cycle 1: all new.  Cycle 2: same candidates again — dedup removes
        them all, giving a high dup fraction that triggers the surge.
        """
        same = ["dup_a", "dup_b", "dup_c"]
        pipeline = _MockPipeline(
            cycle_candidates=[same, same, same],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=10, max_batches=20),
        )

        assert result.cycles_run == 2
        assert "duplicate_surge" in result.stopped_reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmptyGeneration:
    def test_empty_generation_cycle_does_not_crash(self) -> None:
        """A cycle where generate returns [] is handled without error."""
        pipeline = _MockPipeline(cycle_candidates=[[]])
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=2),
        )

        assert result.cycles_run == 2
        for cr in result.cycle_results:
            assert cr.generated_count == 0
            assert cr.top_score is None


class TestZeroValidCandidates:
    def test_zero_valid_candidates_cycle(self) -> None:
        """A cycle where all candidates fail validation does not crash."""
        pipeline = _MockPipeline(
            cycle_candidates=[["a", "b", "c"]],
            valid_fraction=0.0,
            score_value=float("inf"),
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(),
            pipeline,
            _make_budget(max_cycles=1),
        )

        assert result.cycles_run == 1
        cr = result.cycle_results[0]
        assert cr.generated_count == 3
        assert cr.valid_count == 0
        assert cr.scored_count == 0
        assert cr.filtered_count == 0
        assert cr.top_score is None


# ---------------------------------------------------------------------------
# CycleResult and evidence
# ---------------------------------------------------------------------------


class TestCycleResultFields:
    def test_cycle_result_fields_populated(self) -> None:
        """All CycleResult fields are set to correct values after a cycle."""
        pipeline = _MockPipeline(
            cycle_candidates=[["x", "y", "z"]],
            score_value=0.75,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(evidence_level="ml_predicted"),
            pipeline,
            _make_budget(max_cycles=1),
        )

        assert result.cycles_run == 1
        cr = result.cycle_results[0]
        assert cr.cycle_number == 1
        assert cr.generated_count == 3
        assert cr.valid_count == 3
        assert cr.scored_count == 3
        assert cr.filtered_count == 3
        assert cr.top_score == pytest.approx(0.75)
        assert cr.evidence_level == "ml_predicted"


class TestEvidenceLevelPropagation:
    def test_evidence_level_propagated_to_cycle_result(self) -> None:
        """Custom evidence_level flows through to every CycleResult."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}"] for i in range(3)
            ],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(evidence_level="dft_verified"),
            pipeline,
            _make_budget(max_cycles=3, max_batches=20),
        )

        for cr in result.cycle_results:
            assert cr.evidence_level == "dft_verified"


# ---------------------------------------------------------------------------
# Protocol enforcement
# ---------------------------------------------------------------------------


class TestPipelineProtocol:
    def test_pipeline_protocol_enforced(self) -> None:
        """DiscoveryPipeline isinstance check accepts complete and rejects incomplete."""

        class _Complete:
            def generate(self, batch_size: int) -> list[Any]:
                return []

            def validate(self, candidates: list[Any]) -> list[Any]:
                return candidates

            def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
                return []

            def filter_candidates(
                self, scored: list[tuple[Any, float]],
            ) -> list[tuple[Any, float]]:
                return scored

        class _MissingFilter:
            def generate(self, batch_size: int) -> list[Any]:
                return []

            def validate(self, candidates: list[Any]) -> list[Any]:
                return candidates

            def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
                return []

        assert isinstance(_Complete(), DiscoveryPipeline)
        assert not isinstance(_MissingFilter(), DiscoveryPipeline)


# ---------------------------------------------------------------------------
# Cross-cycle deduplication integration
# ---------------------------------------------------------------------------


class TestCrossCycleDedup:
    def test_duplicates_across_cycles_are_removed(self) -> None:
        """Same candidate appearing in multiple cycles is accumulated only once."""
        pipeline = _MockPipeline(
            cycle_candidates=[["shared", "unique1"], ["shared", "unique2"]],
            score_value=0.5,
            score_increment=0.1,
        )
        ctrl = AgentLoopController()
        result = ctrl.run(
            _default_request(output_count=10),
            pipeline,
            _make_budget(max_cycles=2, max_batches=20),
        )

        assert result.shortlist.count("shared") == 1
        assert "unique1" in result.shortlist
        assert "unique2" in result.shortlist


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_batch_size_zero_raises(self) -> None:
        """batch_size < 1 is rejected at the boundary."""
        ctrl = AgentLoopController()
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ctrl.run(
                LoopRequest(batch_size=0),
                _MockPipeline(),
                _make_budget(),
            )

    def test_output_count_zero_raises(self) -> None:
        """output_count < 1 is rejected at the boundary."""
        ctrl = AgentLoopController()
        with pytest.raises(ValueError, match="output_count must be >= 1"):
            ctrl.run(
                LoopRequest(output_count=0),
                _MockPipeline(),
                _make_budget(),
            )
