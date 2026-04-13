"""Shortlist assembler for iterative discovery pipelines.

Pure function that takes accumulated (candidate, score) pairs from multiple
cycles and produces a ranked, deduplicated shortlist.

Deduplication uses candidate identity via hash/equality.  Unhashable
candidates (e.g. dicts) skip dedup tracking silently — acceptable because
the primary candidate types (SMILES strings, composition strings) are
hashable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def assemble_shortlist(
    accumulated: list[tuple[Any, float]],
    output_count: int,
) -> list[tuple[Any, float]]:
    """Select top-scoring, deduplicated candidates from accumulated results.

    Sorts by score descending, removes duplicate candidates (keeping the
    highest-scored occurrence), and returns the top output_count entries.
    Uses a stable sort so ties are resolved deterministically by insertion
    order.

    Args:
        accumulated: All (candidate, score) pairs accumulated across cycles.
        output_count: Maximum number of candidates to return.

    Returns:
        Top unique candidates sorted by descending score, truncated to
        output_count.  Returns fewer than output_count when not enough
        unique candidates exist.

    Raises:
        ValueError: If output_count is negative.
    """
    if output_count < 0:
        raise ValueError(f"output_count must be >= 0, got {output_count}")

    if not accumulated:
        return []

    # Stable sort — ties preserve insertion order for deterministic results
    sorted_items = sorted(accumulated, key=lambda pair: pair[1], reverse=True)

    # Deduplicate by candidate identity, keeping highest-scored occurrence
    # (which comes first after descending sort)
    seen: set = set()
    unique: list[tuple[Any, float]] = []
    for candidate, score in sorted_items:
        try:
            if candidate in seen:
                continue
            seen.add(candidate)
        except TypeError:
            # Unhashable candidates cannot be dedup-tracked; include them all
            logger.warning(
                "Unhashable candidate type %s — skipping dedup",
                type(candidate).__name__,
            )
        unique.append((candidate, score))

    return unique[:output_count]
