"""ML-relaxation stability scoring.

Scores candidate materials based on ML-relaxed energy per atom.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NONCONVERGED_PENALTY_FACTOR = 0.5
DEFAULT_THRESHOLD_EV = 0.1


def stability_score(
    energy_per_atom: float,
    converged: bool,
    threshold: float = DEFAULT_THRESHOLD_EV,
) -> float:
    """Score thermodynamic plausibility from ML-relaxed energy per atom.

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

    return min(1.0, max(0.0, score))
