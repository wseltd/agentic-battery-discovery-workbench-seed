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

from discovery_workbench.pareto import non_dominated_sort

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


def _crowding_distances(
    front: list[dict[str, Any]],
    weights: dict[str, float],
) -> list[float]:
    """Compute crowding distance for each candidate in a single front.

    Uses the NSGA-II algorithm: for each objective, sort the front members,
    assign infinity to boundary points, and accumulate normalised neighbour
    gaps scaled by the objective weight.

    Returns a list aligned with *front* — distances[i] is the crowding
    distance for front[i].
    """
    n = len(front)
    distances = [0.0] * n

    if n <= 2:
        return [float("inf")] * n

    objectives = list(weights.keys())
    for obj in objectives:
        # Sort by this objective's score, tracking original position.
        order = sorted(range(n), key=lambda i: front[i]["scores"][obj])
        obj_min = front[order[0]]["scores"][obj]
        obj_max = front[order[-1]]["scores"][obj]
        span = obj_max - obj_min

        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")

        if span == 0.0:
            continue

        w = weights[obj]
        for k in range(1, len(order) - 1):
            gap = (
                front[order[k + 1]]["scores"][obj]
                - front[order[k - 1]]["scores"][obj]
            )
            distances[order[k]] += w * (gap / span)

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

    objective_names = list(weights.keys())
    fronts = non_dominated_sort(valid, objective_names)

    # Build (front_index, -crowding_distance, candidate) triples for
    # sorting.  Lower front_index is better; higher crowding is better.
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for front_idx, front in enumerate(fronts):
        cd = _crowding_distances(front, weights)
        for i, cand in enumerate(front):
            ranked.append((front_idx, -cd[i], cand))

    # Sort: ascending front_index, then ascending negative crowding
    # (= descending crowding distance).  Ties broken by candidate_id
    # for determinism.
    ranked.sort(key=lambda t: (t[0], t[1], t[2]["candidate_id"]))

    capped = ranked[:shortlist_size]

    shortlist: list[dict[str, Any]] = []
    audit_log: list[RankAuditEntry] = []
    for rank_pos, (front_idx, neg_cd, cand) in enumerate(capped, start=1):
        shortlist.append(cand)
        audit_log.append(
            RankAuditEntry(
                candidate_id=cand["candidate_id"],
                front_index=front_idx,
                crowding_distance=-neg_cd,
                final_rank=rank_pos,
            )
        )

    return RankingResult(shortlist=shortlist, audit_log=audit_log)
