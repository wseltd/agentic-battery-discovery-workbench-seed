"""Tests for the agent loop controller.

Uses a controllable mock pipeline to verify budget enforcement,
early-stop heuristics, accumulation, shortlisting, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from discovery_workbench.budget import BudgetConfig, BudgetController
from discovery_workbench.shared.agent_loop import (
    AgentLoopController,
    DiscoveryPipeline,
    assemble_shortlist,
)


# ---------------------------------------------------------------------------
# Controllable mock pipeline
# ---------------------------------------------------------------------------

class _MockPipeline:
    """Pipeline that returns preconfigured results per cycle.

    Args:
        cycle_candidates: List of candidate lists, one per cycle.
            Cycles beyond the list length reuse the last entry.
        valid_fraction: Fraction of candidates that pass validation.
            Applied by slicing: keep first N where N = int(len * fraction).
        score_value: Fixed score assigned to every valid candidate.
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
        candidates = self._cycle_candidates[idx]
        return candidates[:batch_size]

    def validate(self, candidates: list[Any]) -> list[Any]:
        keep = max(1, int(len(candidates) * self._valid_fraction))
        if self._valid_fraction == 0.0:
            return []
        return candidates[:keep]

    def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
        # Cycle index is 0-based (call_count already incremented in generate)
        cycle_score = self._score_value + (self._call_count - 1) * self._score_increment
        return [(c, cycle_score) for c in valid]

    def filter_candidates(
        self, scored: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        if self._filter_pass:
            return scored
        return []


def _make_controller(
    pipeline: _MockPipeline | None = None,
    max_cycles: int = 5,
    max_batches: int = 8,
    batch_size: int = 10,
    output_count: int = 25,
    evidence_level: str = "generated",
    shortlist_size: int = 25,
) -> AgentLoopController:
    """Build an AgentLoopController with convenient defaults."""
    budget = BudgetController(
        BudgetConfig(
            max_cycles=max_cycles,
            max_batches=max_batches,
            shortlist_size=shortlist_size,
        ),
    )
    return AgentLoopController(
        pipeline=pipeline or _MockPipeline(),
        budget=budget,
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
        ctrl = _make_controller(pipeline=pipeline, max_cycles=1)
        result = ctrl.run()

        assert result.cycles_run == 1
        assert len(result.shortlist) > 0
        assert result.stopped_reason == "budget_exhausted: cycles"


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
            score_increment=0.1,  # avoid plateau
        )
        ctrl = _make_controller(pipeline=pipeline, max_cycles=3, output_count=10)
        result = ctrl.run()

        assert result.cycles_run == 3
        # All 6 unique candidates accumulated
        assert len(result.shortlist) == 6
        # Candidates from every cycle present
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
        ctrl = _make_controller(
            pipeline=pipeline, max_cycles=1, output_count=3, batch_size=10,
        )
        result = ctrl.run()

        assert len(result.shortlist) <= 3

    def test_assemble_shortlist_ranks_by_score(self) -> None:
        """Module-level assemble_shortlist returns top items by descending score."""
        items = [("low", 0.1), ("mid", 0.5), ("high", 0.9), ("med", 0.3)]
        top2 = assemble_shortlist(items, output_count=2)

        assert len(top2) == 2
        assert top2[0] == ("high", 0.9)
        assert top2[1] == ("mid", 0.5)


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetMaxCycles:
    def test_budget_stops_loop_at_max_cycles(self) -> None:
        """Loop runs exactly max_cycles before budget exhaustion stops it."""
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(10)
            ],
            score_value=1.0,
            score_increment=1.0,  # strictly increasing, no plateau
        )
        ctrl = _make_controller(pipeline=pipeline, max_cycles=3, max_batches=20)
        result = ctrl.run()

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
        ctrl = _make_controller(
            pipeline=pipeline, max_cycles=10, max_batches=2,
        )
        result = ctrl.run()

        assert result.cycles_run == 2
        assert "budget_exhausted" in result.stopped_reason
        assert "batches" in result.stopped_reason


# ---------------------------------------------------------------------------
# Early-stop heuristics
# ---------------------------------------------------------------------------

class TestEarlyStopPlateau:
    def test_early_stop_halts_on_plateau(self) -> None:
        """Plateau detection stops loop when scores stop improving.

        With PLATEAU_CONSECUTIVE_CYCLES=2, two cycles with identical scores
        trigger the plateau check on the third iteration's budget check.
        """
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(3)] for i in range(10)
            ],
            score_value=0.5,  # constant score → zero improvement
        )
        ctrl = _make_controller(pipeline=pipeline, max_cycles=10, max_batches=20)
        result = ctrl.run()

        assert result.cycles_run == 2
        assert "plateau" in result.stopped_reason


class TestEarlyStopInvaliditySpike:
    def test_early_stop_halts_on_invalidity_spike(self) -> None:
        """Invalidity spike stops loop after consecutive high-invalidity batches.

        With INVALIDITY_CONSECUTIVE_BATCHES=2 and threshold=0.50, two
        consecutive batches where >50% are invalid triggers the stop.
        """
        pipeline = _MockPipeline(
            cycle_candidates=[
                [f"c{i}_{j}" for j in range(4)] for i in range(10)
            ],
            valid_fraction=0.25,  # 75% invalid, above the 50% threshold
            score_value=1.0,
            score_increment=1.0,  # strictly increasing, prevent plateau
        )
        ctrl = _make_controller(pipeline=pipeline, max_cycles=10, max_batches=20)
        result = ctrl.run()

        assert result.cycles_run == 2
        assert "invalidity_spike" in result.stopped_reason


class TestEarlyStopDuplicateSurge:
    def test_early_stop_halts_on_duplicate_surge(self) -> None:
        """Duplicate surge stops loop when most generated candidates are repeats.

        The pipeline returns the same candidates every cycle.  Cycle 1: all new
        (dup_fraction=0).  Cycle 2: all duplicates (dup_fraction=1.0 >= 0.70
        threshold).  The surge is detected eagerly in record_batch, so the loop
        stops at the budget check before cycle 3.
        """
        same_candidates = ["dup_a", "dup_b", "dup_c"]
        pipeline = _MockPipeline(
            cycle_candidates=[same_candidates, same_candidates, same_candidates],
            score_value=1.0,
            score_increment=1.0,
        )
        ctrl = _make_controller(pipeline=pipeline, max_cycles=10, max_batches=20)
        result = ctrl.run()

        assert result.cycles_run == 2
        assert "duplicate_surge" in result.stopped_reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEmptyGeneration:
    def test_empty_generation_cycle_does_not_crash(self) -> None:
        """A cycle where generate returns [] is handled without error."""
        pipeline = _MockPipeline(cycle_candidates=[[]])
        ctrl = _make_controller(pipeline=pipeline, max_cycles=2)
        result = ctrl.run()

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
        ctrl = _make_controller(pipeline=pipeline, max_cycles=1)
        result = ctrl.run()

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
        ctrl = _make_controller(
            pipeline=pipeline, max_cycles=1, evidence_level="ml_predicted",
        )
        result = ctrl.run()

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
        ctrl = _make_controller(
            pipeline=pipeline, max_cycles=3, max_batches=20,
            evidence_level="dft_verified",
        )
        result = ctrl.run()

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
