"""Tests for agentic_discovery_core.crowding -- crowding distance computation."""

from __future__ import annotations

import math

from agentic_discovery_core.crowding import compute_crowding_distance


def _cand(cid: str, **scores: float) -> dict:
    return {"candidate_id": cid, "scores": scores}


def test_two_candidates_both_get_infinity() -> None:
    front = [_cand("a", x=0.0, y=1.0), _cand("b", x=1.0, y=0.0)]
    result = compute_crowding_distance(front, ["x", "y"], {"x": 1.0, "y": 1.0})
    assert result["a"] == float("inf")
    assert result["b"] == float("inf")


def test_single_candidate_gets_infinity() -> None:
    front = [_cand("solo", x=5.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert result["solo"] == float("inf")


def test_boundary_candidates_in_larger_front() -> None:
    front = [
        _cand("low_x", x=0.0, y=0.5),
        _cand("mid", x=0.5, y=0.5),
        _cand("high_x", x=1.0, y=0.5),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    assert result["low_x"] == float("inf")
    assert result["high_x"] == float("inf")


def test_interior_distance_is_normalised_gap() -> None:
    front = [_cand("lo", x=0.0), _cand("mid", x=0.5), _cand("hi", x=1.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert result["mid"] == 1.0


def test_uneven_spacing_gives_different_distances() -> None:
    front = [
        _cand("a", x=0.0),
        _cand("b", x=0.1),
        _cand("c", x=0.9),
        _cand("d", x=1.0),
    ]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert math.isclose(result["b"], 0.9)
    assert math.isclose(result["c"], 0.9)


def test_multi_objective_distances_accumulate() -> None:
    front = [
        _cand("a", x=0.0, y=1.0),
        _cand("b", x=0.5, y=0.5),
        _cand("c", x=1.0, y=0.0),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    assert math.isclose(result["b"], 2.0)


def test_weight_scales_contribution() -> None:
    front = [_cand("a", x=0.0), _cand("b", x=0.5), _cand("c", x=1.0)]
    d_w1 = compute_crowding_distance(front, ["x"], {"x": 1.0})
    d_w3 = compute_crowding_distance(front, ["x"], {"x": 3.0})
    assert math.isclose(d_w3["b"], 3.0 * d_w1["b"])


def test_tiebreak_weighting_changes_relative_order() -> None:
    front = [
        _cand("left", x=0.0, y=1.0),
        _cand("right", x=1.0, y=0.0),
        _cand("x_spread", x=0.4, y=0.9),
        _cand("y_spread", x=0.9, y=0.4),
    ]
    d_x_heavy = compute_crowding_distance(
        front, ["x", "y"], {"x": 10.0, "y": 0.1},
    )
    d_y_heavy = compute_crowding_distance(
        front, ["x", "y"], {"x": 0.1, "y": 10.0},
    )
    assert d_x_heavy["x_spread"] > d_x_heavy["y_spread"]
    assert d_y_heavy["y_spread"] > d_y_heavy["x_spread"]


def test_empty_front_returns_empty_dict() -> None:
    result = compute_crowding_distance([], ["x"], {"x": 1.0})
    assert result == {}


def test_identical_scores_all_boundary() -> None:
    front = [
        _cand("a", x=1.0, y=1.0),
        _cand("b", x=1.0, y=1.0),
        _cand("c", x=1.0, y=1.0),
    ]
    result = compute_crowding_distance(
        front, ["x", "y"], {"x": 1.0, "y": 1.0},
    )
    inf_count = sum(1 for v in result.values() if v == float("inf"))
    assert inf_count >= 2
    finite_vals = [v for v in result.values() if v != float("inf")]
    assert all(v == 0.0 for v in finite_vals)


def test_return_keys_match_candidate_ids() -> None:
    front = [_cand("alpha", x=1.0), _cand("beta", x=2.0), _cand("gamma", x=3.0)]
    result = compute_crowding_distance(front, ["x"], {"x": 1.0})
    assert set(result.keys()) == {"alpha", "beta", "gamma"}
