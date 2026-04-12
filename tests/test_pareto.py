"""Tests for discovery_workbench.pareto — Deb 2002 fast non-dominated sort.

Focuses on front assignment correctness: domination semantics, multi-front
chains, ties, single-objective degeneration, and empty input.
"""

from __future__ import annotations

from discovery_workbench.pareto import non_dominated_sort


def _cand(cid: str, **scores: float) -> dict:
    """Shorthand to build a candidate dict."""
    return {"candidate_id": cid, "scores": scores}


# -- Basic front assignment ---------------------------------------------------

def test_empty_input_returns_empty() -> None:
    """No candidates means no fronts."""
    assert non_dominated_sort([], ["x", "y"]) == []


def test_single_candidate_on_front_zero() -> None:
    """A lone candidate is always on front 0."""
    fronts = non_dominated_sort([_cand("a", x=1.0)], ["x"])
    assert len(fronts) == 1
    assert len(fronts[0]) == 1
    assert fronts[0][0]["candidate_id"] == "a"


def test_two_nondominated_share_front_zero() -> None:
    """Two candidates that don't dominate each other share front 0."""
    fronts = non_dominated_sort(
        [_cand("a", x=1.0, y=0.0), _cand("b", x=0.0, y=1.0)],
        ["x", "y"],
    )
    assert len(fronts) == 1
    ids = {c["candidate_id"] for c in fronts[0]}
    assert ids == {"a", "b"}


def test_strict_domination_creates_two_fronts() -> None:
    """A dominates B → front 0 = {A}, front 1 = {B}."""
    fronts = non_dominated_sort(
        [_cand("a", x=2.0, y=2.0), _cand("b", x=1.0, y=1.0)],
        ["x", "y"],
    )
    assert len(fronts) == 2
    assert fronts[0][0]["candidate_id"] == "a"
    assert fronts[1][0]["candidate_id"] == "b"


def test_three_front_dominance_chain() -> None:
    """Three candidates in a strict dominance chain produce three fronts."""
    fronts = non_dominated_sort(
        [
            _cand("top", x=3.0, y=3.0),
            _cand("mid", x=2.0, y=2.0),
            _cand("low", x=1.0, y=1.0),
        ],
        ["x", "y"],
    )
    assert len(fronts) == 3
    assert fronts[0][0]["candidate_id"] == "top"
    assert fronts[1][0]["candidate_id"] == "mid"
    assert fronts[2][0]["candidate_id"] == "low"


# -- Edge cases ---------------------------------------------------------------

def test_equal_scores_share_front() -> None:
    """Candidates with identical scores don't dominate each other."""
    fronts = non_dominated_sort(
        [_cand("a", x=1.0, y=1.0), _cand("b", x=1.0, y=1.0)],
        ["x", "y"],
    )
    assert len(fronts) == 1
    assert len(fronts[0]) == 2


def test_partial_domination_no_strict_advantage() -> None:
    """A >= B on all objectives but equal on all → no domination."""
    fronts = non_dominated_sort(
        [_cand("a", x=5.0, y=5.0), _cand("b", x=5.0, y=5.0)],
        ["x", "y"],
    )
    assert len(fronts) == 1


def test_domination_requires_strict_on_at_least_one() -> None:
    """A >= B on all but strictly > on only one objective → A dominates B."""
    fronts = non_dominated_sort(
        [_cand("a", x=1.0, y=2.0), _cand("b", x=1.0, y=1.0)],
        ["x", "y"],
    )
    assert len(fronts) == 2
    assert fronts[0][0]["candidate_id"] == "a"
    assert fronts[1][0]["candidate_id"] == "b"


def test_single_objective_degenerates_to_chain() -> None:
    """With one objective, each unique score level is its own front."""
    fronts = non_dominated_sort(
        [_cand("low", x=1.0), _cand("mid", x=5.0), _cand("hi", x=10.0)],
        ["x"],
    )
    assert len(fronts) == 3
    ids_in_order = [fronts[i][0]["candidate_id"] for i in range(3)]
    assert ids_in_order == ["hi", "mid", "low"]


def test_stable_order_within_front() -> None:
    """Candidates on the same front preserve input order."""
    cands = [
        _cand("c", x=0.0, y=1.0),
        _cand("a", x=0.5, y=0.5),
        _cand("b", x=1.0, y=0.0),
    ]
    fronts = non_dominated_sort(cands, ["x", "y"])
    assert len(fronts) == 1
    ids = [c["candidate_id"] for c in fronts[0]]
    assert ids == ["c", "a", "b"]


def test_mixed_fronts_larger_population() -> None:
    """Five candidates with a mix of domination relationships."""
    cands = [
        _cand("p1", x=5.0, y=1.0),   # front 0 (non-dominated with p2)
        _cand("p2", x=1.0, y=5.0),   # front 0
        _cand("p3", x=3.0, y=3.0),   # front 0 (not dominated by p1 or p2)
        _cand("p4", x=2.0, y=2.0),   # front 1 (dominated by p3)
        _cand("p5", x=0.5, y=0.5),   # front 2 (dominated by p4)
    ]
    fronts = non_dominated_sort(cands, ["x", "y"])
    front_ids = [{c["candidate_id"] for c in f} for f in fronts]
    assert front_ids[0] == {"p1", "p2", "p3"}
    assert front_ids[1] == {"p4"}
    assert front_ids[2] == {"p5"}


def test_subset_of_objectives_used() -> None:
    """Sorting uses only the listed objectives, ignoring extra score keys."""
    cands = [
        _cand("a", x=1.0, y=10.0, z=0.0),
        _cand("b", x=2.0, y=5.0, z=100.0),
    ]
    # Only consider x and y — neither dominates the other.
    fronts = non_dominated_sort(cands, ["x", "y"])
    assert len(fronts) == 1
    assert len(fronts[0]) == 2
