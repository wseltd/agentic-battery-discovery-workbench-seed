"""Deb 2002 fast non-dominated sort (NSGA-II).

Implements the O(MN^2) algorithm from:
    Deb, Pratap, Agarwal, Meyarivan -- "A Fast and Elitist Multiobjective
    Genetic Algorithm: NSGA-II", IEEE TEC 6(2), 2002.

All objectives are maximised: candidate A dominates B iff A.scores[obj] >= B.scores[obj]
for every objective and strictly greater for at least one.
"""

from __future__ import annotations

from typing import Any

# Same dict shape as ranker_validation.ScoredCandidate:
# {"candidate_id": str, "scores": dict[str, float]}
ScoredCandidate = dict[str, Any]


def _dominates(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    objectives: list[str],
) -> bool:
    """Return True if *scores_a* Pareto-dominates *scores_b* on *objectives*."""
    at_least_one_strict = False
    for obj in objectives:
        if scores_a[obj] < scores_b[obj]:
            return False
        if scores_a[obj] > scores_b[obj]:
            at_least_one_strict = True
    return at_least_one_strict


def non_dominated_sort(
    candidates: list[ScoredCandidate],
    objective_names: list[str],
) -> list[list[ScoredCandidate]]:
    """Partition *candidates* into successive Pareto fronts.

    Uses the Deb 2002 fast non-dominated sort: precompute domination
    counts and dominated-by sets in O(MN^2), then peel fronts in O(N)
    per front.

    Parameters
    ----------
    candidates:
        Each dict must have ``candidate_id`` (str) and ``scores``
        (dict mapping objective name to float).  Only valid candidates
        should be passed -- upstream validation is assumed.
    objective_names:
        Objective keys to compare across candidates.

    Returns
    -------
    list[list[ScoredCandidate]]
        Front 0 is the Pareto-optimal set, front 1 is optimal after
        removing front 0, etc.  Order within each front is stable
        (preserves input order).
    """
    n = len(candidates)
    if n == 0:
        return []

    # Phase 1: for every pair, compute domination relationships.
    # dominated_by[p] = indices of candidates that p dominates.
    # domination_count[p] = how many candidates dominate p.
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count: list[int] = [0] * n

    for p in range(n):
        for q in range(p + 1, n):
            p_scores = candidates[p]["scores"]
            q_scores = candidates[q]["scores"]
            if _dominates(p_scores, q_scores, objective_names):
                dominated_by[p].append(q)
                domination_count[q] += 1
            elif _dominates(q_scores, p_scores, objective_names):
                dominated_by[q].append(p)
                domination_count[p] += 1

    # Phase 2: peel fronts.  Front 0 = candidates with domination_count == 0.
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]
    fronts: list[list[ScoredCandidate]] = []

    while current_front_indices:
        fronts.append([candidates[i] for i in current_front_indices])
        next_front: list[int] = []
        for p in current_front_indices:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        # Stable ordering: sort by original index so output order within
        # a front matches input order.
        next_front.sort()
        current_front_indices = next_front

    return fronts
