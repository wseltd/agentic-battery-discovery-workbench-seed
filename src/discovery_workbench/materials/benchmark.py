"""Materials benchmark metrics for evaluating generated inorganic candidates.

Computes validity, uniqueness, novelty, stability proxy, target satisfaction,
shortlist usefulness, and DFT conversion rate from a list of candidate dicts
carrying boolean flags set by earlier pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MaterialsBenchmarkReport:
    """Aggregate benchmark metrics for a generated materials library.

    All percentage fields are fractions in [0.0, 1.0].  ``__post_init__``
    clamps every ``float`` field to that range; ``dft_conversion_rate``
    is excluded from clamping when ``None``.

    Args:
        validity_pct: Fraction of candidates passing structural validation.
        uniqueness_pct: Fraction of candidates that are not duplicates.
        novelty_pct: Fraction of candidates flagged as novel.
        stability_proxy_pct: Fraction of candidates passing stability proxy.
        target_satisfaction_pct: Fraction meeting user-specified targets.
        shortlist_usefulness: Fraction passing all five filters above.
        dft_conversion_rate: Reserved for future DFT success tracking.
    """

    validity_pct: float
    uniqueness_pct: float
    novelty_pct: float
    stability_proxy_pct: float
    target_satisfaction_pct: float
    shortlist_usefulness: float
    dft_conversion_rate: float | None

    def __post_init__(self) -> None:
        """Clamp float fields to [0.0, 1.0]."""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float):
                clamped = max(0.0, min(1.0, value))
                object.__setattr__(self, f.name, clamped)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compute_materials_benchmark(
    candidates: list[dict[str, object]],
) -> MaterialsBenchmarkReport:
    """Compute benchmark metrics from a list of candidate result dicts.

    Each dict is expected to carry boolean flags from earlier pipeline
    stages.  Missing flags are treated as ``False``.

    Expected keys per candidate dict:
        ``is_valid``, ``is_duplicate``, ``is_novel``, ``is_stable``,
        ``satisfies_target``.

    Shortlist usefulness counts candidates that pass all five criteria
    (valid, not duplicate, novel, stable, satisfies target).

    Args:
        candidates: List of candidate dicts with boolean flag values.

    Returns:
        MaterialsBenchmarkReport with all metrics populated.
        dft_conversion_rate is always None (not yet implemented).
    """
    logger.info(
        "compute_materials_benchmark called with %d candidates",
        len(candidates),
    )
    n = len(candidates)
    if n == 0:
        return MaterialsBenchmarkReport(
            validity_pct=0.0,
            uniqueness_pct=0.0,
            novelty_pct=0.0,
            stability_proxy_pct=0.0,
            target_satisfaction_pct=0.0,
            shortlist_usefulness=0.0,
            dft_conversion_rate=None,
        )

    valid_count = 0
    unique_count = 0
    novel_count = 0
    stable_count = 0
    target_count = 0
    shortlist_count = 0

    for c in candidates:
        is_valid = bool(c.get("is_valid", False))
        is_duplicate = bool(c.get("is_duplicate", False))
        is_novel = bool(c.get("is_novel", False))
        is_stable = bool(c.get("is_stable", False))
        satisfies_target = bool(c.get("satisfies_target", False))

        if is_valid:
            valid_count += 1
        if not is_duplicate:
            unique_count += 1
        if is_novel:
            novel_count += 1
        if is_stable:
            stable_count += 1
        if satisfies_target:
            target_count += 1
        if is_valid and not is_duplicate and is_novel and is_stable and satisfies_target:
            shortlist_count += 1

    return MaterialsBenchmarkReport(
        validity_pct=valid_count / n,
        uniqueness_pct=unique_count / n,
        novelty_pct=novel_count / n,
        stability_proxy_pct=stable_count / n,
        target_satisfaction_pct=target_count / n,
        shortlist_usefulness=shortlist_count / n,
        dft_conversion_rate=None,
    )
