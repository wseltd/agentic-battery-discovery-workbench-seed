"""Tests for discovery_workbench.crowding — crowding distance computation.

Focuses on correctness of the NSGA-II crowding distance algorithm:
boundary infinity assignment, normalised gap accumulation, weight
scaling, and edge cases (empty front, single candidate, identical scores).
"""

from __future__ import annotations

import math

from discovery_workbench.crowding import compute_crowding_distance


def _cand(cid: str, **scores: float) -> dict:
    """Shorthand to build a candidate dict."""
    return {"candidate_id": cid, "scores": scores}


# -- Boundary candidates get infinity ----------------------------------------

def test_two_candidates_both_get_infinity() -> None:
    """A front with only two members has no interior — both are boundary."""
    front = [_cand("a", x=0.0, y=1.0), _cand("b", x=1.0, y=0.0)]
    result = compute_crowding_distance(front, ["x", "y"], {"x": 1.0, "y": 1.0})
    assert result["a"] == float("inf")
    assert result["b"] == float("inf")


def test_single_candidate_gets_infinity() -> None:
    """Degenerate front with one member — it is its own boundary."""
    front = [_cand("solo", x=5.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert result["solo"] == float("inf")


def test_boundary_candidates_in_larger_front() -> None:
    """Extreme-value candidates on any objective must get infinite distance."""
    front = [
        _cand("low_x", x=0.0, y=0.5),
        _cand("mid", x=0.5, y=0.5),
        _cand("high_x", x=1.0, y=0.5),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    # low_x and high_x are boundary on x; mid is interior on x but all
    # share the same y, so y boundaries also include low_x and high_x
    # (they are first/last when sorted by y — ties broken by sort order,
    # but boundary assignment goes to order[0] and order[-1]).
    assert result["low_x"] == float("inf")
    assert result["high_x"] == float("inf")


# -- Interior distance accumulation ------------------------------------------

def test_interior_distance_is_normalised_gap() -> None:
    """For a single objective with unit weight, the interior distance
    equals the normalised gap between neighbours."""
    # Three candidates on one objective: 0, 0.5, 1.0
    front = [_cand("lo", x=0.0), _cand("mid", x=0.5), _cand("hi", x=1.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    # mid's neighbours span the full range [0, 1], gap = 1.0, span = 1.0
    assert result["mid"] == 1.0


def test_uneven_spacing_gives_different_distances() -> None:
    """Interior candidates with different neighbour gaps get different
    crowding distances."""
    front = [
        _cand("a", x=0.0),
        _cand("b", x=0.1),  # close to a, far from c
        _cand("c", x=0.9),  # far from b, close to d
        _cand("d", x=1.0),
    ]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    # b: gap = (0.9 - 0.0) / 1.0 = 0.9
    # c: gap = (1.0 - 0.1) / 1.0 = 0.9
    assert math.isclose(result["b"], 0.9)
    assert math.isclose(result["c"], 0.9)


def test_multi_objective_distances_accumulate() -> None:
    """Distances from multiple objectives add up for interior candidates."""
    front = [
        _cand("a", x=0.0, y=1.0),
        _cand("b", x=0.5, y=0.5),
        _cand("c", x=1.0, y=0.0),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    # b is interior on both objectives.
    # x: gap = (1.0 - 0.0) / 1.0 = 1.0
    # y: gap = (1.0 - 0.0) / 1.0 = 1.0  (neighbours are a=1.0, c=0.0)
    # Total = 2.0
    assert math.isclose(result["b"], 2.0)


# -- Weight scaling -----------------------------------------------------------

def test_weight_scales_contribution() -> None:
    """Doubling a weight doubles that objective's contribution to the
    interior candidate's crowding distance."""
    front = [_cand("a", x=0.0), _cand("b", x=0.5), _cand("c", x=1.0)]
    d_w1 = compute_crowding_distance(front, ["x"], {"x": 1.0})
    d_w3 = compute_crowding_distance(front, ["x"], {"x": 3.0})
    assert math.isclose(d_w3["b"], 3.0 * d_w1["b"])


def test_tiebreak_weighting_changes_relative_order() -> None:
    """With two objectives and four non-dominated candidates, adjusting
    weights changes which interior candidate has higher crowding distance.

    Placement is chosen so that x_spread has a larger normalised gap on x
    (0.9) but smaller on y (0.6), while y_spread has the mirror pattern.
    Shifting weight emphasis flips which interior candidate wins.
    """
    front = [
        _cand("left", x=0.0, y=1.0),
        _cand("right", x=1.0, y=0.0),
        # x_spread: x-sorted neighbours are left(0.0) and y_spread(0.9) → gap 0.9
        #           y-sorted neighbours are y_spread(0.4) and left(1.0) → gap 0.6
        _cand("x_spread", x=0.4, y=0.9),
        # y_spread: x-sorted neighbours are x_spread(0.4) and right(1.0) → gap 0.6
        #           y-sorted neighbours are right(0.0) and x_spread(0.9) → gap 0.9
        _cand("y_spread", x=0.9, y=0.4),
    ]
    d_x_heavy = compute_crowding_distance(
        front, ["x", "y"], {"x": 10.0, "y": 0.1},
    )
    d_y_heavy = compute_crowding_distance(
        front, ["x", "y"], {"x": 0.1, "y": 10.0},
    )
    # x-heavy: x_spread's large x-gap dominates → x_spread wins.
    assert d_x_heavy["x_spread"] > d_x_heavy["y_spread"]
    # y-heavy: y_spread's large y-gap dominates → y_spread wins.
    assert d_y_heavy["y_spread"] > d_y_heavy["x_spread"]


# -- Edge cases ---------------------------------------------------------------

def test_empty_front_returns_empty_dict() -> None:
    """Empty input must not crash — return an empty mapping."""
    result = compute_crowding_distance([], ["x"], {"x": 1.0})
    assert result == {}


def test_identical_scores_all_boundary() -> None:
    """When all candidates have the same scores, span is zero and
    boundary candidates still get infinity (no interior accumulation)."""
    front = [
        _cand("a", x=1.0, y=1.0),
        _cand("b", x=1.0, y=1.0),
        _cand("c", x=1.0, y=1.0),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    # First and last in sort order get infinity; middle gets 0.
    inf_count = sum(1 for v in result.values() if v == float("inf"))
    assert inf_count >= 2  # at least boundary points
    # Interior candidate(s) get 0 since span is 0 on both objectives.
    finite_vals = [v for v in result.values() if v != float("inf")]
    assert all(v == 0.0 for v in finite_vals)


def test_return_keys_match_candidate_ids() -> None:
    """Returned dict keys must be exactly the candidate_id values."""
    front = [_cand("alpha", x=1.0), _cand("beta", x=2.0), _cand("gamma", x=3.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert set(result.keys()) == {"alpha", "beta", "gamma"}
