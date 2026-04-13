"""Loop controller for iterative discovery workflows.

Orchestrates generate-validate-score-filter cycles with budget enforcement,
cross-cycle deduplication via deduplicate_across_cycles, and optional
early-stop heuristics.

Design choice: early stopping is decoupled from budget enforcement.
Budget handles hard limits (max cycles, max batches) via should_stop();
the optional early_stop_evaluator handles softer heuristics (plateau,
invalidity, duplicate surge).  This separation keeps budget logic
testable in isolation and lets callers compose different stop strategies.

Design choice: deduplication is delegated to deduplicate_across_cycles
rather than inline tracking.  This keeps the loop body focused on
orchestration and lets the dedup logic be tested and evolved separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentic_workbench.orchestration.dedup import deduplicate_across_cycles
from agentic_workbench.orchestration.shortlist import assemble_shortlist
from discovery_workbench.shared.agent_loop import (
    CycleResult,
    DiscoveryPipeline,
    LoopResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoopRequest:
    """Parameters for a single agent loop run.

    Args:
        batch_size: Candidates to request per generate call.
        output_count: Target shortlist size for final ranking.
        evidence_level: Evidence tier label for cycle results.
    """

    batch_size: int = 50
    output_count: int = 25
    evidence_level: str = "generated"


class AgentLoopController:
    """Drives generate-validate-score-filter loops with budget and early-stop control.

    Budget is checked BEFORE each generate call.  Early-stop evaluation
    happens AFTER each completed cycle.  This ordering ensures no
    generation work is wasted when the budget is already exhausted.

    Args:
        early_stop_evaluator: Optional heuristic evaluator with a
            should_stop(cycle_result, accumulated) -> bool method.
            When None, only budget limits can stop the loop.
    """

    def __init__(self, early_stop_evaluator: Any | None = None) -> None:
        self._early_stop_evaluator = early_stop_evaluator

    def run(
        self,
        request: LoopRequest,
        pipeline: DiscoveryPipeline,
        budget: Any,
    ) -> LoopResult:
        """Execute the discovery loop until a stop condition triggers.

        Each iteration: check budget -> generate -> validate -> score ->
        filter -> deduplicate -> accumulate -> early-stop check.

        Args:
            request: Loop parameters (batch size, output count, evidence level).
            pipeline: Discovery pipeline implementing generate/validate/score/filter.
            budget: Budget controller with should_stop() -> (bool, str | None),
                record_batch(valid_fraction, dup_fraction), and
                record_cycle(best_score) methods.

        Returns:
            LoopResult with shortlist, cycle count, per-cycle metrics,
            and the reason the loop stopped.

        Raises:
            ValueError: If batch_size or output_count is less than 1.
        """
        if request.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {request.batch_size}")
        if request.output_count < 1:
            raise ValueError(f"output_count must be >= 1, got {request.output_count}")

        logger.info(
            "Starting agent loop batch_size=%d output_count=%d",
            request.batch_size,
            request.output_count,
        )

        accumulated: list[tuple[Any, float]] = []
        cycle_results: list[CycleResult] = []
        seen: set = set()
        stopped_reason = "completed"

        cycle_num = 0
        while True:
            # Budget checked BEFORE generate — never waste a generation call
            should_stop, reason = budget.should_stop()
            if should_stop:
                stopped_reason = reason or "budget_exhausted"
                break

            cycle_num += 1

            # Generate -> validate -> score -> filter
            candidates = pipeline.generate(request.batch_size)
            generated_count = len(candidates)

            if generated_count == 0:
                budget.record_batch(1.0, 0.0)
                budget.record_cycle(0.0)
                cycle_result = CycleResult(
                    cycle_number=cycle_num,
                    generated_count=0,
                    valid_count=0,
                    scored_count=0,
                    filtered_count=0,
                    top_score=None,
                    evidence_level=request.evidence_level,
                )
                cycle_results.append(cycle_result)

                if self._should_early_stop(cycle_result, accumulated):
                    stopped_reason = "early_stop"
                    break
                continue

            valid = pipeline.validate(candidates)
            valid_count = len(valid)
            valid_fraction = valid_count / generated_count

            scored = pipeline.score(valid)
            scored_count = len(scored)

            filtered = pipeline.filter_candidates(scored)

            # Deduplicate against accumulated results from prior cycles
            deduped, seen = deduplicate_across_cycles(filtered, seen)
            deduped_count = len(deduped)

            top = max((s for _, s in deduped), default=None)

            # Dup fraction relative to generated count for budget heuristics
            removed_count = len(filtered) - deduped_count
            dup_fraction = removed_count / generated_count

            budget.record_batch(valid_fraction, dup_fraction)
            budget.record_cycle(top if top is not None else 0.0)

            # Accumulate deduplicated results
            accumulated.extend(deduped)

            cycle_result = CycleResult(
                cycle_number=cycle_num,
                generated_count=generated_count,
                valid_count=valid_count,
                scored_count=scored_count,
                filtered_count=deduped_count,
                top_score=top,
                evidence_level=request.evidence_level,
            )
            cycle_results.append(cycle_result)

            logger.info(
                "Cycle %d: generated=%d valid=%d scored=%d filtered=%d top=%s",
                cycle_num,
                generated_count,
                valid_count,
                scored_count,
                deduped_count,
                f"{top:.4f}" if top is not None else "None",
            )

            # Early-stop evaluation AFTER cycle completion
            if self._should_early_stop(cycle_result, accumulated):
                stopped_reason = "early_stop"
                break

        shortlist_pairs = assemble_shortlist(accumulated, request.output_count)

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

    def _should_early_stop(
        self,
        cycle_result: CycleResult,
        accumulated: list[tuple[Any, float]],
    ) -> bool:
        """Check the early-stop evaluator if one is configured.

        Args:
            cycle_result: Metrics from the just-completed cycle.
            accumulated: All (candidate, score) pairs accumulated so far.

        Returns:
            True if the evaluator says to stop, False otherwise.
        """
        if self._early_stop_evaluator is None:
            return False
        return self._early_stop_evaluator.should_stop(cycle_result, accumulated)
