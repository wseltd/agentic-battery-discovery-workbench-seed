"""Materials benchmark metric aggregation over pre-computed result dicts.

Computes S.U.N.-style metrics (validity, uniqueness, novelty,
stability proxy, target satisfaction) from a batch of candidate dicts
carrying boolean flags set by earlier pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaterialsMetrics:
    """Aggregate benchmark metrics for a generated materials library.

    All fields are fractions in [0.0, 1.0].

    Args:
        validity: Fraction of candidates passing structural validation.
        uniqueness: Fraction of candidates that are not duplicates.
        novelty: Fraction of candidates flagged as novel.
        stability_proxy: Fraction of candidates passing stability proxy.
        target_satisfaction: Fraction meeting user-specified targets.
    """

    validity: float
    uniqueness: float
    novelty: float
    stability_proxy: float
    target_satisfaction: float


def compute_materials_metrics(
    candidates: list[dict[str, object]],
) -> MaterialsMetrics:
    """Compute materials benchmark metrics from pre-computed result dicts.

    Each dict carries boolean flags from earlier pipeline stages.
    Missing flags are treated as ``False``.

    Expected keys per candidate dict:
        ``is_valid``, ``is_duplicate``, ``is_novel``,
        ``meets_stability_threshold``, ``meets_target``.

    Args:
        candidates: List of candidate dicts with boolean flag values.

    Returns:
        MaterialsMetrics with all fields populated as fractions.
    """
    logger.info(
        "compute_materials_metrics called with %d candidates",
        len(candidates),
    )
    n = len(candidates)
    if n == 0:
        return MaterialsMetrics(
            validity=0.0,
            uniqueness=0.0,
            novelty=0.0,
            stability_proxy=0.0,
            target_satisfaction=0.0,
        )

    valid_count = sum(1 for c in candidates if bool(c.get("is_valid", False)))
    unique_count = sum(
        1 for c in candidates if not bool(c.get("is_duplicate", False))
    )
    novel_count = sum(1 for c in candidates if bool(c.get("is_novel", False)))
    stable_count = sum(
        1 for c in candidates
        if bool(c.get("meets_stability_threshold", False))
    )
    target_count = sum(
        1 for c in candidates if bool(c.get("meets_target", False))
    )

    return MaterialsMetrics(
        validity=valid_count / n,
        uniqueness=unique_count / n,
        novelty=novel_count / n,
        stability_proxy=stable_count / n,
        target_satisfaction=target_count / n,
    )
