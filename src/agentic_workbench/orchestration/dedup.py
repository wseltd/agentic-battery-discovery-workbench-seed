"""Cycle-level duplicate filtering for iterative discovery loops.

Removes candidates whose identity (via hash) is already in a seen set,
preventing duplicates from leaking between iterations.  The controller
calls this each cycle before accumulating results.

Design choice: only hashable candidates are tracked.  Unhashable types
(e.g. dicts) raise TypeError at the boundary rather than silently
passing through — unlike the shortlist assembler, duplicate filtering
is a correctness requirement, not a best-effort convenience.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def deduplicate_across_cycles(
    new_candidates: list[tuple[Any, float]],
    seen: set,
) -> tuple[list[tuple[Any, float]], set]:
    """Remove candidates already encountered in previous cycles.

    Filters new_candidates by checking each candidate's identity against
    the seen set, then adds novel candidates to seen.  Returns the
    filtered list and the updated seen set.

    Args:
        new_candidates: Scored (candidate, score) pairs from the current cycle.
        seen: Mutable set of previously encountered candidate identities.

    Returns:
        Tuple of (filtered candidates not in seen, updated seen set with
        new entries added).

    Raises:
        TypeError: If seen is not a mutable set (e.g. frozenset or list).
    """
    if not isinstance(seen, set):
        raise TypeError(
            f"seen must be a mutable set, got {type(seen).__name__}"
        )

    if not new_candidates:
        return [], seen

    filtered: list[tuple[Any, float]] = []
    for candidate, score in new_candidates:
        if candidate in seen:
            logger.info("Duplicate candidate filtered: %s", candidate)
            continue
        seen.add(candidate)
        filtered.append((candidate, score))

    logger.info(
        "Dedup: %d candidates in, %d novel, %d duplicates removed",
        len(new_candidates),
        len(filtered),
        len(new_candidates) - len(filtered),
    )

    return filtered, seen
