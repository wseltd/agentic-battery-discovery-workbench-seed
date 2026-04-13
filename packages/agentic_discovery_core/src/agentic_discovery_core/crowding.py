"""NSGA-II crowding distance computation.

Computes per-candidate crowding distance within a single Pareto front,
using the standard NSGA-II algorithm: for each objective, sort by score,
assign infinity to boundary candidates, and accumulate normalised
neighbour gaps for interior candidates.  Each objective's contribution
is scaled by its weight to allow tiebreak weighting.

Higher crowding distance means the candidate lives in a less crowded
region of objective space and should be preferred during selection.
"""

from __future__ import annotations

from agentic_discovery_core.pareto import ScoredCandidate


def compute_crowding_distance(
    front: list[ScoredCandidate],
    objective_names: list[str],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute weighted crowding distance for each candidate in a front.

    Parameters
    ----------
    front:
        Candidates on a single Pareto front.  Each dict must have
        ``candidate_id`` (str) and ``scores`` (dict[str, float]).
    objective_names:
        Objective keys to use for distance computation.
    weights:
        Positive weight per objective -- each objective's normalised gap
        contribution is multiplied by ``weights[objective]``.

    Returns
    -------
    dict[str, float]
        Mapping from ``candidate_id`` to crowding distance.  Boundary
        candidates (best or worst on any single objective) get
        ``float('inf')``.
    """
    n = len(front)
    if n == 0:
        return {}

    # Trivial fronts: all members are boundary points.
    if n <= 2:
        return {cand["candidate_id"]: float("inf") for cand in front}

    distances = [0.0] * n

    for obj in objective_names:
        # Sort indices by this objective's score.
        obj_name = obj  # bind for lambda closure
        order = sorted(range(n), key=lambda i: front[i]["scores"][obj_name])

        obj_min = front[order[0]]["scores"][obj]
        obj_max = front[order[-1]]["scores"][obj]
        span = obj_max - obj_min

        # Boundary candidates always get infinity.
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")

        if span == 0.0:
            # All candidates have the same score on this objective --
            # no discriminating information to add.
            continue

        w = weights[obj]
        for k in range(1, len(order) - 1):
            gap = (
                front[order[k + 1]]["scores"][obj]
                - front[order[k - 1]]["scores"][obj]
            )
            distances[order[k]] += w * (gap / span)

    return {
        front[i]["candidate_id"]: distances[i]
        for i in range(n)
    }
