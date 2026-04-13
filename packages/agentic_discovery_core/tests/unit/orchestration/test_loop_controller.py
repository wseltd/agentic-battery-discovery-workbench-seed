"""Tests for the orchestration-level loop controller."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_discovery_core.budget import BudgetConfig, BudgetController
from agentic_discovery_core.shared.agent_loop import (
    CycleResult,
    DiscoveryPipeline,
)

from agentic_discovery_core.orchestration.loop_controller import (
    AgentLoopController,
    LoopRequest,
)


class _MockPipeline:
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


class TestSingleCycle:
    def test_single_cycle_produces_shortlist(self) -> None:
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


class TestBudgetMaxCycles:
    def test_budget_stops_loop_at_max_cycles(self) -> None:
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


class _CountingEarlyStop:
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
        cr1, acc1 = captured[0]
        assert cr1.cycle_number == 1
        assert len(acc1) == 2

        cr2, acc2 = captured[1]
        assert cr2.cycle_number == 2
        assert len(acc2) == 4


class TestEarlyStopPlateau:
    def test_early_stop_halts_on_plateau(self) -> None:
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


class TestEmptyGeneration:
    def test_empty_generation_cycle_does_not_crash(self) -> None:
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


class TestCycleResultFields:
    def test_cycle_result_fields_populated(self) -> None:
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


class TestPipelineProtocol:
    def test_pipeline_protocol_enforced(self) -> None:
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


class TestCrossCycleDedup:
    def test_duplicates_across_cycles_are_removed(self) -> None:
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


class TestInputValidation:
    def test_batch_size_zero_raises(self) -> None:
        ctrl = AgentLoopController()
        with pytest.raises(ValueError, match="batch_size must be >= 1") as exc_info:
            ctrl.run(
                LoopRequest(batch_size=0),
                _MockPipeline(),
                _make_budget(),
            )
        assert "0" in str(exc_info.value)

    def test_output_count_zero_raises(self) -> None:
        ctrl = AgentLoopController()
        with pytest.raises(ValueError, match="output_count must be >= 1") as exc_info:
            ctrl.run(
                LoopRequest(output_count=0),
                _MockPipeline(),
                _make_budget(),
            )
        assert "0" in str(exc_info.value)
