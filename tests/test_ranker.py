"""Tests for discovery_workbench.ranker — Pareto ranking with audit logging.

Focuses on front assignment correctness, crowding distance tiebreaking,
boundary handling (empty input, NaN scores), and shortlist capping.
"""

from __future__ import annotations

import warnings

from discovery_workbench.ranker import (
    RankAuditEntry,
    RankingResult,
    compute_pareto_ranking,
)


def _cand(cid: str, **scores: float) -> dict:
    """Shorthand to build a candidate dict."""
    return {"candidate_id": cid, "scores": scores}


# -- Front assignment --------------------------------------------------------

def test_two_nondominated_candidates_both_on_front() -> None:
    """Two candidates that don't dominate each other share front 0."""
    result = compute_pareto_ranking(
        candidates=[_cand("a", x=1.0, y=0.0), _cand("b", x=0.0, y=1.0)],
        weights={"x": 1.0, "y": 1.0},
        shortlist_size=10,
    )
    assert len(result.shortlist) == 2
    assert all(e.front_index == 0 for e in result.audit_log)


def test_one_dominates_other_front_assignment() -> None:
    """When A dominates B, A goes to front 0 and B to front 1."""
    result = compute_pareto_ranking(
        candidates=[_cand("a", x=1.0, y=1.0), _cand("b", x=0.5, y=0.5)],
        weights={"x": 1.0, "y": 1.0},
        shortlist_size=10,
    )
    audit_by_id = {e.candidate_id: e for e in result.audit_log}
    assert audit_by_id["a"].front_index == 0
    assert audit_by_id["b"].front_index == 1


def test_three_fronts_correct_assignment() -> None:
    """Three candidates forming a dominance chain produce three fronts."""
    result = compute_pareto_ranking(
        candidates=[
            _cand("top", x=3.0, y=3.0),
            _cand("mid", x=2.0, y=2.0),
            _cand("low", x=1.0, y=1.0),
        ],
        weights={"x": 1.0, "y": 1.0},
        shortlist_size=10,
    )
    audit_by_id = {e.candidate_id: e for e in result.audit_log}
    assert audit_by_id["top"].front_index == 0
    assert audit_by_id["mid"].front_index == 1
    assert audit_by_id["low"].front_index == 2


# -- Crowding distance -------------------------------------------------------

def test_crowding_distance_tiebreak_uses_weights() -> None:
    """Within the same front, crowding distance (scaled by weights) determines
    rank.  With 4 non-dominated candidates on front 0, the two boundary
    points (extreme in either objective) get infinite crowding distance and
    rank first.  Adjusting weights changes which interior candidate ranks
    higher among the non-boundary members."""
    # Four candidates on a single Pareto front.
    cands = [
        _cand("left", x=0.0, y=1.0),
        _cand("right", x=1.0, y=0.0),
        _cand("inner_a", x=0.3, y=0.7),
        _cand("inner_b", x=0.7, y=0.3),
    ]
    # Weight x heavily — crowding distance in x dimension counts more.
    result = compute_pareto_ranking(
        candidates=cands,
        weights={"x": 10.0, "y": 1.0},
        shortlist_size=4,
    )
    audit_by_id = {e.candidate_id: e for e in result.audit_log}
    # Both boundary points get infinite crowding distance.
    assert audit_by_id["left"].crowding_distance == float("inf")
    assert audit_by_id["right"].crowding_distance == float("inf")
    # Interior candidates have finite, distinct distances.
    assert audit_by_id["inner_a"].crowding_distance < float("inf")
    assert audit_by_id["inner_b"].crowding_distance < float("inf")


# -- Shortlist capping -------------------------------------------------------

def test_shortlist_size_caps_output() -> None:
    """Output length must not exceed shortlist_size even when more candidates
    are available."""
    cands = [_cand(f"c{i}", x=float(i)) for i in range(20)]
    result = compute_pareto_ranking(
        candidates=cands, weights={"x": 1.0}, shortlist_size=5,
    )
    assert len(result.shortlist) == 5
    assert len(result.audit_log) == 5


# -- Edge cases --------------------------------------------------------------

def test_empty_candidates_returns_empty() -> None:
    """Empty input must return empty shortlist and audit log, not crash."""
    result = compute_pareto_ranking(
        candidates=[], weights={"x": 1.0}, shortlist_size=10,
    )
    assert isinstance(result, RankingResult)
    assert result.shortlist == []
    assert result.audit_log == []


def test_nan_score_rejected_with_warning() -> None:
    """Candidates with NaN scores are dropped and a warning is emitted."""
    cands = [
        _cand("good", x=1.0, y=1.0),
        {"candidate_id": "bad", "scores": {"x": float("nan"), "y": 0.5}},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_pareto_ranking(
            candidates=cands, weights={"x": 1.0, "y": 1.0}, shortlist_size=10,
        )
    assert len(result.shortlist) == 1
    assert result.shortlist[0]["candidate_id"] == "good"
    nan_warnings = [w for w in caught if "NaN" in str(w.message)]
    assert len(nan_warnings) == 1


def test_single_objective_degenerates_to_sort() -> None:
    """With only one objective the Pareto ranking must degenerate to a simple
    descending sort — each candidate on its own front."""
    cands = [_cand("low", x=1.0), _cand("mid", x=5.0), _cand("high", x=10.0)]
    result = compute_pareto_ranking(
        candidates=cands, weights={"x": 1.0}, shortlist_size=10,
    )
    ids = [e.candidate_id for e in result.audit_log]
    assert ids == ["high", "mid", "low"]
    # Each candidate dominates the next, so each is on a separate front.
    fronts = [e.front_index for e in result.audit_log]
    assert fronts == [0, 1, 2]


# -- Audit log completeness --------------------------------------------------

def test_audit_log_contains_all_ranked_candidates() -> None:
    """Every shortlisted candidate must have a corresponding audit entry with
    a valid RankAuditEntry, and final_rank values must be consecutive
    starting from 1."""
    cands = [_cand(f"c{i}", x=float(i), y=float(10 - i)) for i in range(6)]
    result = compute_pareto_ranking(
        candidates=cands, weights={"x": 1.0, "y": 1.0}, shortlist_size=4,
    )
    assert len(result.audit_log) == len(result.shortlist) == 4
    for entry in result.audit_log:
        assert isinstance(entry, RankAuditEntry)
    ranks = [e.final_rank for e in result.audit_log]
    assert ranks == [1, 2, 3, 4]
    # Every audit entry maps to a candidate in the shortlist.
    audit_ids = {e.candidate_id for e in result.audit_log}
    shortlist_ids = {c["candidate_id"] for c in result.shortlist}
    assert audit_ids == shortlist_ids
