"""Budget tracking and early-stop detection for discovery loops.

Provides BudgetConfig (immutable limits), BudgetState (mutable counters),
and BudgetController (stateful tracker that evaluates stop conditions).

Early-stop heuristics detect three failure modes:
- Plateau: score improvement stalls across consecutive cycles.
- Invalidity spike: too many invalid candidates in consecutive batches.
- Duplicate surge: too many duplicates in a single batch.

These are heuristic thresholds, not guarantees.  They trade recall for
cost savings by terminating loops that are unlikely to improve.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Early-stop thresholds ---------------------------------------------------
# Plateau: improvement below this fraction for consecutive cycles triggers stop.
PLATEAU_IMPROVEMENT_THRESHOLD: float = 0.01
PLATEAU_CONSECUTIVE_CYCLES: int = 2

# Invalidity: fraction of invalid candidates in a batch; consecutive hits stop.
INVALIDITY_SPIKE_THRESHOLD: float = 0.50
INVALIDITY_CONSECUTIVE_BATCHES: int = 2

# Duplicates: fraction of duplicates in a single batch triggers immediate stop.
DUPLICATE_SURGE_THRESHOLD: float = 0.70


@dataclass
class BudgetConfig:
    """Immutable budget limits for a discovery loop.

    Args:
        max_cycles: Maximum optimisation cycles before hard stop.
        max_batches: Maximum generation batches before hard stop.
        shortlist_size: Target shortlist size for final ranking.
    """

    max_cycles: int = 5
    max_batches: int = 8
    shortlist_size: int = 25


@dataclass
class BudgetState:
    """Mutable counters tracking consumption against a BudgetConfig."""

    cycles_used: int = 0
    batches_used: int = 0
    stopped: bool = False
    stop_reason: str | None = None


class BudgetController:
    """Tracks budget consumption and evaluates early-stop conditions.

    Holds a BudgetConfig (limits) and a BudgetState (counters), plus bounded
    history lists for heuristic stop detection.

    Args:
        config: Budget limits.  Defaults to BudgetConfig() if omitted.
    """

    # Max entries kept in each history list.  Sized to the minimum window
    # each heuristic needs — no point storing older values.
    _MAX_SCORE_HISTORY = PLATEAU_CONSECUTIVE_CYCLES + 1
    _MAX_INVALIDITY_HISTORY = INVALIDITY_CONSECUTIVE_BATCHES

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
        self.state = BudgetState()

        # Plateau tracking: bounded list of recent best scores per cycle.
        self._cycle_scores: list[float] = []
        # Invalidity tracking: bounded list of recent invalidity fractions.
        self._invalidity_fractions: list[float] = []

    def record_cycle(self, best_score: float) -> None:
        """Record one completed optimisation cycle with its best score.

        Args:
            best_score: Best candidate score observed in this cycle.
        """
        self.state.cycles_used += 1
        self._cycle_scores.append(best_score)
        if len(self._cycle_scores) > self._MAX_SCORE_HISTORY:
            self._cycle_scores = self._cycle_scores[-self._MAX_SCORE_HISTORY:]

    def record_batch(self, valid_fraction: float, duplicate_fraction: float) -> None:
        """Record one completed generation batch.

        Args:
            valid_fraction: Fraction of candidates that passed validation (0-1).
                Invalidity rate is computed as 1 - valid_fraction.
            duplicate_fraction: Fraction of candidates that were duplicates (0-1).
        """
        self.state.batches_used += 1

        invalidity = 1.0 - valid_fraction
        self._invalidity_fractions.append(invalidity)
        if len(self._invalidity_fractions) > self._MAX_INVALIDITY_HISTORY:
            self._invalidity_fractions = self._invalidity_fractions[
                -self._MAX_INVALIDITY_HISTORY:
            ]

        # Duplicate surge is an immediate stop — checked eagerly here rather
        # than deferring to should_stop, because a single bad batch is enough.
        if duplicate_fraction >= DUPLICATE_SURGE_THRESHOLD and not self.state.stopped:
            self.state.stopped = True
            self.state.stop_reason = (
                "duplicate_surge: high duplicate fraction in batch"
            )

    def should_stop(self) -> tuple[bool, str | None]:
        """Evaluate all stop conditions.  First triggered condition wins.

        Returns:
            (should_stop, reason) where reason is None when should_stop is False.

        Order: exhaustion (cycles, batches), plateau, invalidity spike.
        Duplicate surge is checked eagerly in record_batch.
        Order is fixed so that results are deterministic.
        """
        if self.state.stopped:
            return True, self.state.stop_reason

        # Duplicate surge is checked eagerly in record_batch (immediate stop),
        # so it is not re-evaluated here.
        reason = (
            self._check_exhaustion()
            or self._check_plateau()
            or self._check_invalidity()
        )

        if reason is not None:
            self.state.stopped = True
            self.state.stop_reason = reason
            return True, reason

        return False, None

    def remaining(self) -> dict:
        """Return remaining budget as a dict.

        Returns:
            Dict with keys ``cycles`` and ``batches``, each an int >= 0.
        """
        return {
            "cycles": max(0, self.config.max_cycles - self.state.cycles_used),
            "batches": max(0, self.config.max_batches - self.state.batches_used),
        }

    # --- Private stop-condition checkers --------------------------------------

    def _check_exhaustion(self) -> str | None:
        if self.state.cycles_used >= self.config.max_cycles:
            return "budget_exhausted: cycles"
        if self.state.batches_used >= self.config.max_batches:
            return "budget_exhausted: batches"
        return None

    def _check_plateau(self) -> str | None:
        n = len(self._cycle_scores)
        if n < PLATEAU_CONSECUTIVE_CYCLES:
            # Need at least PLATEAU_CONSECUTIVE_CYCLES scores to form a plateau.
            return None

        # Check the last (PLATEAU_CONSECUTIVE_CYCLES - 1) improvements.
        # Two consecutive plateau cycles = one improvement window to check.
        for i in range(n - PLATEAU_CONSECUTIVE_CYCLES + 1, n):
            improvement = self._cycle_scores[i] - self._cycle_scores[i - 1]
            if improvement > PLATEAU_IMPROVEMENT_THRESHOLD:
                return None

        return "plateau: score improvement below threshold for consecutive cycles"

    def _check_invalidity(self) -> str | None:
        if len(self._invalidity_fractions) < INVALIDITY_CONSECUTIVE_BATCHES:
            return None
        # Check that the last N fractions are all at or above the threshold.
        tail = self._invalidity_fractions[-INVALIDITY_CONSECUTIVE_BATCHES:]
        if all(f >= INVALIDITY_SPIKE_THRESHOLD for f in tail):
            return "invalidity_spike: high invalid fraction in consecutive batches"
        return None
