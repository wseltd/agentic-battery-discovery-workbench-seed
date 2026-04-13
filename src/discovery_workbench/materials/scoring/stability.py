"""ML-relaxation stability scoring.

Scores candidate materials based on ML-relaxed energy per atom.  This is
a pre-screening heuristic used before convex-hull analysis is available —
it normalises raw energy/atom against a configurable threshold rather than
computing energy above hull.

Chose a multiplicative penalty for non-convergence (0.5x) rather than a
fixed low score because the base energy still carries information — a
near-zero energy/atom that didn't converge is more promising than a
high-energy non-converged result.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Multiplicative factor applied when the ML relaxation did not converge.
# Halves the score rather than zeroing it — non-converged results still
# carry partial information about the energy landscape.
NONCONVERGED_PENALTY_FACTOR = 0.5

# Default energy-per-atom threshold in eV.  Candidates with energy/atom
# at or above this value score 0.
DEFAULT_THRESHOLD_EV = 0.1


def stability_score(
    energy_per_atom: float,
    converged: bool,
    threshold: float = DEFAULT_THRESHOLD_EV,
) -> float:
    """Score thermodynamic plausibility from ML-relaxed energy per atom.

    Normalises energy/atom linearly: 0 eV/atom -> 1.0, threshold -> 0.0.
    Non-converged relaxations receive a 0.5x penalty.  Result is clamped
    to [0.0, 1.0].

    Args:
        energy_per_atom: ML-relaxed energy per atom in eV.
        converged: Whether the ML relaxation converged.
        threshold: Energy per atom at which the score reaches zero.

    Returns:
        Stability score in [0.0, 1.0].

    Raises:
        ValueError: If threshold is not positive.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    logger.info(
        "Scoring stability energy_per_atom=%s converged=%s threshold=%s",
        energy_per_atom, converged, threshold,
    )

    score = max(0.0, 1.0 - energy_per_atom / threshold)

    if not converged:
        score *= NONCONVERGED_PENALTY_FACTOR

    # Clamp to [0, 1] — energy_per_atom < 0 would push score above 1.0
    return min(1.0, max(0.0, score))
