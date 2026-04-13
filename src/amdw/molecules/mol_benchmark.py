"""Molecular benchmark metric aggregation over pre-computed result dicts.

Computes GuacaMol/MOSES-style metrics (validity, uniqueness, novelty,
diversity, target satisfaction) from a batch of candidate dicts carrying
boolean flags set by earlier pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MolecularMetrics:
    """Aggregate benchmark metrics for a generated molecular library.

    All fields are fractions in [0.0, 1.0].

    Args:
        validity: Fraction of candidates passing structural validation.
        uniqueness: Fraction of candidates that are not duplicates.
        novelty: Fraction of candidates flagged as novel.
        diversity: Mean per-candidate diversity score, or 0.0 if absent.
        target_satisfaction: Fraction meeting user-specified property targets.
    """

    validity: float
    uniqueness: float
    novelty: float
    diversity: float
    target_satisfaction: float


def compute_molecular_metrics(
    candidates: list[dict[str, object]],
) -> MolecularMetrics:
    """Compute molecular benchmark metrics from pre-computed result dicts.

    Each dict carries boolean flags from earlier pipeline stages.
    Missing flags are treated as ``False``.  Missing ``diversity_score``
    values are excluded from the diversity mean.

    Expected keys per candidate dict:
        ``is_valid``, ``is_duplicate``, ``is_novel``, ``meets_target``.
        Optional: ``diversity_score`` (float in [0.0, 1.0]).

    Args:
        candidates: List of candidate dicts with boolean flag values.

    Returns:
        MolecularMetrics with all fields populated as fractions.
    """
    logger.info(
        "compute_molecular_metrics called with %d candidates",
        len(candidates),
    )
    n = len(candidates)
    if n == 0:
        return MolecularMetrics(
            validity=0.0,
            uniqueness=0.0,
            novelty=0.0,
            diversity=0.0,
            target_satisfaction=0.0,
        )

    valid_count = sum(1 for c in candidates if bool(c.get("is_valid", False)))
    unique_count = sum(
        1 for c in candidates if not bool(c.get("is_duplicate", False))
    )
    novel_count = sum(1 for c in candidates if bool(c.get("is_novel", False)))
    target_count = sum(
        1 for c in candidates if bool(c.get("meets_target", False))
    )

    # Diversity: mean of per-candidate diversity_score when provided.
    diversity_scores = [
        float(c["diversity_score"])
        for c in candidates
        if "diversity_score" in c
    ]
    diversity = (
        sum(diversity_scores) / len(diversity_scores)
        if diversity_scores
        else 0.0
    )

    return MolecularMetrics(
        validity=valid_count / n,
        uniqueness=unique_count / n,
        novelty=novel_count / n,
        diversity=diversity,
        target_satisfaction=target_count / n,
    )
