"""Multi-objective Pareto ranker with NSGA-II-style front assignment.

Assigns candidates to successive Pareto fronts, computes crowding
distance within each front (weighted by objective weights), and
produces a ranked shortlist with a full audit log.

Higher scores are better (maximisation on all objectives).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RankAuditEntry:
    """Audit record for a single ranked candidate.

    Parameters
    ----------
    candidate_id:
        Unique identifier of the candidate.
    front_index:
        Zero-based Pareto front the candidate was assigned to.
    crowding_distance:
        Crowding distance within its front (higher = more isolated).
    final_rank:
        One-based rank in the final shortlist (1 = best).
    """

    candidate_id: str
    front_index: int
    crowding_distance: float
    final_rank: int


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Result of Pareto ranking.

    Parameters
    ----------
    shortlist:
        Candidates in rank order, capped at the requested shortlist size.
        Each entry is the original candidate dict.
    audit_log:
        One :class:`RankAuditEntry` per candidate in *shortlist* order.
    """

    shortlist: list[dict[str, Any]] = field(default_factory=list)
    audit_log: list[RankAuditEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def _dominates(scores_a: dict[str, float], scores_b: dict[str, float]) -> bool:
    """Return True if *scores_a* Pareto-dominates *scores_b* (maximisation)."""
    dominated = False
    for key in scores_a:
        if scores_a[key] < scores_b[key]:
            return False
        if scores_a[key] > scores_b[key]:
            dominated = True
    return dominated


def _assign_fronts(
    candidates: list[dict[str, Any]],
) -> list[list[int]]:
    """Assign candidates to Pareto fronts.  Returns list of fronts, each a
    list of indices into *candidates*.  Front 0 is the non-dominated set."""
    n = len(candidates)
    remaining = set(range(n))
    fronts: list[list[int]] = []

    while remaining:
        front: list[int] = []
        for i in remaining:
            is_dominated = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(candidates[j]["scores"], candidates[i]["scores"]):
                    is_dominated = True
                    break
            if not is_dominated:
                front.append(i)
        # Deterministic ordering within a front for reproducibility.
        front.sort()
        fronts.append(front)
        remaining -= set(front)
    return fronts


def _crowding_distances(
    candidates: list[dict[str, Any]],
    indices: list[int],
    weights: dict[str, float],
) -> dict[int, float]:
    """Compute crowding distance for candidates within a single front.

    Uses the NSGA-II algorithm: for each objective, sort the front members,
    assign infinity to boundary points, and accumulate normalised neighbour
    gaps scaled by the objective weight.
    """
    distances: dict[int, float] = {i: 0.0 for i in indices}

    if len(indices) <= 2:
        # With 1-2 members every candidate gets infinite distance.
        for i in indices:
            distances[i] = float("inf")
        return distances

    objectives = list(weights.keys())
    for obj in objectives:
        # Sort indices by this objective's score.
        sorted_idx = sorted(indices, key=lambda i: candidates[i]["scores"][obj])
        obj_min = candidates[sorted_idx[0]]["scores"][obj]
        obj_max = candidates[sorted_idx[-1]]["scores"][obj]
        span = obj_max - obj_min

        # Boundary points get infinite distance.
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        if span == 0.0:
            continue

        w = weights[obj]
        for k in range(1, len(sorted_idx) - 1):
            gap = (
                candidates[sorted_idx[k + 1]]["scores"][obj]
                - candidates[sorted_idx[k - 1]]["scores"][obj]
            )
            distances[sorted_idx[k]] += w * (gap / span)

    return distances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pareto_ranking(
    candidates: list[dict[str, Any]],
    weights: dict[str, float],
    shortlist_size: int,
) -> RankingResult:
    """Rank candidates using multi-objective Pareto sorting.

    Parameters
    ----------
    candidates:
        Each dict must have ``candidate_id`` (str) and ``scores``
        (dict mapping objective name to float).
    weights:
        Positive weight per objective, used for crowding distance scaling.
    shortlist_size:
        Maximum number of candidates to include in the shortlist.

    Returns
    -------
    RankingResult
        Contains ``shortlist`` (list of candidate dicts, length at most
        *shortlist_size*) and ``audit_log`` (one :class:`RankAuditEntry`
        per shortlisted candidate).

    Warns
    -----
    UserWarning
        If any candidate has NaN scores — those candidates are dropped.
    """
    if not candidates:
        return RankingResult()

    # Filter out candidates with NaN scores.
    valid: list[dict[str, Any]] = []
    for cand in candidates:
        has_nan = False
        for val in cand["scores"].values():
            # NaN != NaN is the standard NaN check.
            if val != val:  # noqa: PLR0124
                has_nan = True
                break
        if has_nan:
            warnings.warn(
                f"Candidate {cand['candidate_id']!r} has NaN scores and was dropped",
                UserWarning,
                stacklevel=2,
            )
        else:
            valid.append(cand)

    if not valid:
        return RankingResult()

    fronts = _assign_fronts(valid)

    # Build (front_index, -crowding_distance, candidate_index) tuples
    # for sorting.  Lower front_index is better; higher crowding is better.
    ranked_tuples: list[tuple[int, float, int]] = []
    for front_idx, front in enumerate(fronts):
        cd = _crowding_distances(valid, front, weights)
        for cand_idx in front:
            ranked_tuples.append((front_idx, -cd[cand_idx], cand_idx))

    # Sort: ascending front_index, then ascending negative crowding distance
    # (= descending crowding distance).
    ranked_tuples.sort()

    capped = ranked_tuples[:shortlist_size]

    shortlist: list[dict[str, Any]] = []
    audit_log: list[RankAuditEntry] = []
    for rank_pos, (front_idx, neg_cd, cand_idx) in enumerate(capped, start=1):
        shortlist.append(valid[cand_idx])
        audit_log.append(
            RankAuditEntry(
                candidate_id=valid[cand_idx]["candidate_id"],
                front_index=front_idx,
                crowding_distance=-neg_cd,
                final_rank=rank_pos,
            )
        )

    return RankingResult(shortlist=shortlist, audit_log=audit_log)
