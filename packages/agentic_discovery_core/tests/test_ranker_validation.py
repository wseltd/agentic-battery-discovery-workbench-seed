"""Tests for agentic_discovery_core.ranker_validation -- pre-ranking candidate filter."""

from __future__ import annotations

from agentic_discovery_core.ranker_validation import validate_candidates


def _cand(cid: str, **scores: float) -> dict:
    return {"candidate_id": cid, "scores": scores}


def test_nan_score_rejected_with_reason() -> None:
    cands = [_cand("ok", x=1.0, y=2.0), {"candidate_id": "bad", "scores": {"x": float("nan"), "y": 0.5}}]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 1
    assert valid[0]["candidate_id"] == "ok"
    assert len(warns) == 1
    assert "bad" in warns[0]
    assert "NaN" in warns[0]


def test_nan_on_multiple_objectives_reports_all() -> None:
    cand = {"candidate_id": "multi", "scores": {"x": float("nan"), "y": float("nan"), "z": 1.0}}
    valid, warns = validate_candidates([cand], ["x", "y", "z"])
    assert valid == []
    assert len(warns) == 1
    assert "x" in warns[0]
    assert "y" in warns[0]


def test_missing_objective_rejected_with_reason() -> None:
    cands = [_cand("ok", x=1.0, y=2.0), _cand("partial", x=1.0)]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 1
    assert valid[0]["candidate_id"] == "ok"
    assert "partial" in warns[0]
    assert "missing" in warns[0].lower()


def test_missing_multiple_keys_reports_all() -> None:
    cand = _cand("sparse", z=1.0)
    valid, warns = validate_candidates([cand], ["x", "y", "z"])
    assert valid == []
    assert "x" in warns[0]
    assert "y" in warns[0]


def test_empty_candidates_returns_empty() -> None:
    valid, warns = validate_candidates([], ["x", "y"])
    assert valid == []
    assert warns == []


def test_all_invalid_returns_empty_valid_list() -> None:
    cands = [
        {"candidate_id": "nan_one", "scores": {"x": float("nan")}},
        _cand("missing_one"),
    ]
    valid, warns = validate_candidates(cands, ["x"])
    assert valid == []
    assert len(warns) == 2


def test_extra_score_keys_are_ignored() -> None:
    cand = _cand("extra", x=1.0, y=2.0, bonus=99.0)
    valid, warns = validate_candidates([cand], ["x", "y"])
    assert len(valid) == 1
    assert warns == []


def test_all_valid_candidates_pass_through() -> None:
    cands = [_cand("a", x=1.0, y=2.0), _cand("b", x=3.0, y=4.0)]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 2
    assert warns == []


def test_preserves_candidate_order() -> None:
    cands = [
        _cand("third", x=3.0),
        {"candidate_id": "reject", "scores": {"x": float("nan")}},
        _cand("first", x=1.0),
        _cand("second", x=2.0),
    ]
    valid, warns = validate_candidates(cands, ["x"])
    ids = [c["candidate_id"] for c in valid]
    assert ids == ["third", "first", "second"]
    assert len(warns) == 1
