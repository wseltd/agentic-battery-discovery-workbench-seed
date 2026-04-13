"""Early-stop evaluation for discovery cycles.

Checks three degradation signals and returns a stop decision.
Priority order: invalidity spike > duplicate surge > plateau.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Relative improvement below this over PLATEAU_WINDOW consecutive
# transitions triggers a plateau stop.
PLATEAU_THRESHOLD = 0.01

# Number of consecutive below-threshold transitions required.
PLATEAU_WINDOW = 2

# Fraction of invalid candidates above this for INVALIDITY_WINDOW
# consecutive cycles triggers an invalidity-spike stop.
INVALIDITY_THRESHOLD = 0.50

# Number of consecutive above-threshold cycles required.
INVALIDITY_WINDOW = 2

# Fraction of duplicates above this in a single cycle triggers stop.
DUPLICATE_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class CycleStats:
    """Statistics for a single discovery cycle.

    Args:
        top_10_score: Mean score of the top-10 candidates.
        invalid_fraction: Fraction of candidates that failed validation.
        duplicate_fraction: Fraction of candidates that were duplicates.
    """

    top_10_score: float
    invalid_fraction: float
    duplicate_fraction: float


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Result of early-stop evaluation.

    Args:
        should_stop: Whether the loop should terminate.
        reason: One of 'plateau', 'invalidity_spike', 'duplicate_surge', 'none'.
        detail: Human-readable explanation with numeric values.
    """

    should_stop: bool
    reason: str
    detail: str


def evaluate_stop(history: list[CycleStats]) -> StopDecision:
    """Evaluate whether the discovery loop should stop early.

    Checks three conditions in priority order:
    1. Invalidity spike -- INVALIDITY_WINDOW consecutive cycles >= INVALIDITY_THRESHOLD.
    2. Duplicate surge -- current cycle >= DUPLICATE_THRESHOLD.
    3. Plateau -- PLATEAU_WINDOW consecutive transitions with < PLATEAU_THRESHOLD
       relative improvement in top-10 score.

    Args:
        history: Ordered list of cycle statistics, oldest first.

    Returns:
        StopDecision indicating whether to stop and why.
    """
    logger.info("Evaluating early stop with %s cycles", len(history))

    if len(history) < 2:
        return StopDecision(
            should_stop=False,
            reason="none",
            detail="Not enough cycles to evaluate",
        )

    # Invalidity spike (highest priority)
    if len(history) >= INVALIDITY_WINDOW:
        recent = history[-INVALIDITY_WINDOW:]
        if all(c.invalid_fraction >= INVALIDITY_THRESHOLD for c in recent):
            fractions = [f"{c.invalid_fraction:.2f}" for c in recent]
            return StopDecision(
                should_stop=True,
                reason="invalidity_spike",
                detail=(
                    f"Invalid fraction >= {INVALIDITY_THRESHOLD:.2f} for "
                    f"{INVALIDITY_WINDOW} consecutive cycles: "
                    f"{', '.join(fractions)}"
                ),
            )

    # Duplicate surge (second priority)
    current = history[-1]
    if current.duplicate_fraction >= DUPLICATE_THRESHOLD:
        return StopDecision(
            should_stop=True,
            reason="duplicate_surge",
            detail=(
                f"Duplicate fraction {current.duplicate_fraction:.2f} "
                f">= threshold {DUPLICATE_THRESHOLD:.2f}"
            ),
        )

    # Plateau (lowest priority)
    if len(history) >= PLATEAU_WINDOW + 1:
        improvements: list[float] = []
        for i in range(-PLATEAU_WINDOW, 0):
            prev_score = history[i - 1].top_10_score
            curr_score = history[i].top_10_score
            if prev_score == 0.0:
                improvement = curr_score
            else:
                improvement = (curr_score - prev_score) / abs(prev_score)
            improvements.append(improvement)

        if all(imp < PLATEAU_THRESHOLD for imp in improvements):
            formatted = [f"{imp:.4f}" for imp in improvements]
            return StopDecision(
                should_stop=True,
                reason="plateau",
                detail=(
                    f"Relative improvement < {PLATEAU_THRESHOLD:.2f} for "
                    f"{PLATEAU_WINDOW} consecutive transitions: "
                    f"{', '.join(formatted)}"
                ),
            )

    return StopDecision(
        should_stop=False,
        reason="none",
        detail="No stop condition met",
    )
