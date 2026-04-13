"""Agent loop controller for iterative discovery workflows.

Drives generate-validate-score-filter cycles with budget enforcement
and early-stop heuristics delegated to BudgetController.

Design choice: duplicate tracking uses a set at the generation level,
which requires hashable candidates.  For unhashable types (e.g. dicts),
duplicates are silently untracked — the loop still runs but the
duplicate-surge heuristic will not fire.  This is acceptable because
the primary candidate types (SMILES strings, composition strings) are
hashable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from discovery_workbench.budget import BudgetController

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Immutable record of one generate-validate-score-filter cycle.

    Args:
        cycle_number: 1-based cycle index within the loop run.
        generated_count: Candidates returned by generate.
        valid_count: Candidates that passed validation.
        scored_count: Candidates that received scores.
        filtered_count: Candidates remaining after filtering.
        top_score: Highest score in this cycle, or None if no candidates scored.
        evidence_level: Evidence credibility tier for this cycle's outputs.
    """

    cycle_number: int
    generated_count: int
    valid_count: int
    scored_count: int
    filtered_count: int
    top_score: float | None
    evidence_level: str


@dataclass(frozen=True, slots=True)
class LoopResult:
    """Final output of a completed agent loop run.

    Args:
        shortlist: Top candidates after all cycles, ranked by score.
        cycles_run: Number of cycles completed before stopping.
        cycle_results: Per-cycle metrics for every completed cycle.
        stopped_reason: Why the loop terminated.
    """

    shortlist: list[Any]
    cycles_run: int
    cycle_results: list[CycleResult]
    stopped_reason: str


@runtime_checkable
class DiscoveryPipeline(Protocol):
    """Protocol for the four stages a discovery pipeline must implement."""

    def generate(self, batch_size: int) -> list[Any]:
        """Produce a batch of candidate structures or molecules."""
        ...  # Protocol method — implementors provide the body

    def validate(self, candidates: list[Any]) -> list[Any]:
        """Filter candidates that fail domain-specific validity checks."""
        ...  # Protocol method — implementors provide the body

    def score(self, valid: list[Any]) -> list[tuple[Any, float]]:
        """Assign numeric scores to valid candidates."""
        ...  # Protocol method — implementors provide the body

    def filter_candidates(
        self, scored: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        """Apply post-scoring filters (e.g. diversity, constraints)."""
        ...  # Protocol method — implementors provide the body

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def assemble_shortlist(
    accumulated: list[tuple[Any, float]],
    output_count: int,
) -> list[tuple[Any, float]]:
    """Select the top-scoring candidates from accumulated results.

    Args:
        accumulated: All (candidate, score) pairs accumulated across cycles.
        output_count: Maximum number of candidates to return.

    Returns:
        Top candidates sorted by descending score, truncated to output_count.
    """
    sorted_items = sorted(accumulated, key=lambda pair: pair[1], reverse=True)
    return sorted_items[:output_count]


class AgentLoopController:
    """Drives iterative generate-validate-score-filter loops with budget control.

    Wraps a DiscoveryPipeline and a BudgetController to run cycles until the
    budget is exhausted, an early-stop heuristic triggers, or all cycles
    complete.  Budget is checked before every generate call; early-stop
    conditions are evaluated after every cycle via BudgetController.should_stop.

    Args:
        pipeline: Discovery pipeline implementing the four required stages.
        budget: Budget controller tracking limits and early-stop conditions.
        batch_size: Number of candidates to request per generate call.
        output_count: Target shortlist size for final ranking.
        evidence_level: Evidence tier label attached to cycle results.
    """

    def __init__(
        self,
        pipeline: DiscoveryPipeline,
        budget: BudgetController,
        batch_size: int = 50,
        output_count: int = 25,
        evidence_level: str = "generated",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if output_count < 1:
            raise ValueError(f"output_count must be >= 1, got {output_count}")

        self._pipeline = pipeline
        self._budget = budget
        self._batch_size = batch_size
        self._output_count = output_count
        self._evidence_level = evidence_level

    def run(self) -> LoopResult:
        """Execute the discovery loop until a stop condition triggers.

        Each iteration: check budget -> generate -> validate -> score ->
        filter -> record metrics.  Accumulated results are assembled into
        a ranked shortlist at the end.

        Returns:
            LoopResult with shortlist, cycle count, per-cycle metrics,
            and stop reason.
        """
        logger.info(
            "Starting agent loop batch_size=%d output_count=%d max_cycles=%d",
            self._batch_size,
            self._output_count,
            self._budget.config.max_cycles,
        )

        accumulated: list[tuple[Any, float]] = []
        cycle_results: list[CycleResult] = []
        stopped_reason = "completed"
        seen: set = set()

        cycle_num = 0
        while True:
            # Budget checked before each generate call
            should_stop, reason = self._budget.should_stop()
            if should_stop:
                stopped_reason = reason or "unknown"
                break

            cycle_num += 1
            cycle_result = self._run_one_cycle(
                cycle_num, seen, accumulated,
            )
            cycle_results.append(cycle_result)

        shortlist_pairs = self.assemble_shortlist(accumulated, self._output_count)

        result = LoopResult(
            shortlist=[item for item, _ in shortlist_pairs],
            cycles_run=len(cycle_results),
            cycle_results=cycle_results,
            stopped_reason=stopped_reason,
        )
        logger.info(
            "Loop finished cycles=%d stopped=%s shortlist=%d",
            result.cycles_run,
            result.stopped_reason,
            len(result.shortlist),
        )
        return result

    def _run_one_cycle(
        self,
        cycle_num: int,
        seen: set,
        accumulated: list[tuple[Any, float]],
    ) -> CycleResult:
        """Execute a single generate-validate-score-filter cycle.

        Args:
            cycle_num: 1-based cycle index.
            seen: Mutable set tracking previously generated candidates.
            accumulated: Mutable list collecting filtered (candidate, score) pairs.

        Returns:
            CycleResult with metrics for this cycle.
        """
        candidates = self._pipeline.generate(self._batch_size)
        generated_count = len(candidates)

        if generated_count == 0:
            self._budget.record_batch(1.0, 0.0)
            self._budget.record_cycle(0.0)
            logger.info("Cycle %d: empty generation", cycle_num)
            return CycleResult(
                cycle_number=cycle_num,
                generated_count=0,
                valid_count=0,
                scored_count=0,
                filtered_count=0,
                top_score=None,
                evidence_level=self._evidence_level,
            )

        # Track duplicates at generation level
        dup_count = 0
        for candidate in candidates:
            try:
                if candidate in seen:
                    dup_count += 1
                else:
                    seen.add(candidate)
            except TypeError:
                pass
        dup_fraction = dup_count / generated_count

        valid = self._pipeline.validate(candidates)
        valid_count = len(valid)
        valid_fraction = valid_count / generated_count

        scored = self._pipeline.score(valid)
        scored_count = len(scored)

        filtered = self._pipeline.filter_candidates(scored)
        filtered_count = len(filtered)

        top = max((s for _, s in filtered), default=None)

        self._budget.record_batch(valid_fraction, dup_fraction)
        self._budget.record_cycle(top if top is not None else 0.0)

        accumulated.extend(filtered)

        logger.info(
            "Cycle %d: generated=%d valid=%d scored=%d filtered=%d top=%s",
            cycle_num,
            generated_count,
            valid_count,
            scored_count,
            filtered_count,
            f"{top:.4f}" if top is not None else "None",
        )

        return CycleResult(
            cycle_number=cycle_num,
            generated_count=generated_count,
            valid_count=valid_count,
            scored_count=scored_count,
            filtered_count=filtered_count,
            top_score=top,
            evidence_level=self._evidence_level,
        )

    @staticmethod
    def assemble_shortlist(
        accumulated: list[tuple[Any, float]],
        output_count: int,
    ) -> list[tuple[Any, float]]:
        """Select top-scoring candidates from accumulated results.

        Delegates to the module-level assemble_shortlist function.

        Args:
            accumulated: All (candidate, score) pairs across cycles.
            output_count: Maximum candidates to return.

        Returns:
            Top candidates by descending score, truncated to output_count.
        """
        return assemble_shortlist(accumulated, output_count)
