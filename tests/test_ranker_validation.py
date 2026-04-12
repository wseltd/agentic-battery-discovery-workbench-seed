"""Tests for discovery_workbench.ranker_validation — pre-ranking candidate filter.

Exercises NaN detection, missing-key rejection, mixed valid/invalid batches,
and edge cases (empty input, all-invalid input, extra keys ignored).
"""

from __future__ import annotations

from discovery_workbench.ranker_validation import validate_candidates


def _cand(cid: str, **scores: float) -> dict:
    """Shorthand to build a candidate dict."""
    return {"candidate_id": cid, "scores": scores}


# -- NaN rejection -----------------------------------------------------------

def test_nan_score_rejected_with_reason() -> None:
    """A candidate with a NaN value in a required objective is rejected."""
    cands = [_cand("ok", x=1.0, y=2.0), {"candidate_id": "bad", "scores": {"x": float("nan"), "y": 0.5}}]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 1
    assert valid[0]["candidate_id"] == "ok"
    assert len(warns) == 1
    assert "bad" in warns[0]
    assert "NaN" in warns[0]


def test_nan_on_multiple_objectives_reports_all() -> None:
    """When multiple objectives have NaN, the warning lists all of them."""
    cand = {"candidate_id": "multi", "scores": {"x": float("nan"), "y": float("nan"), "z": 1.0}}
    valid, warns = validate_candidates([cand], ["x", "y", "z"])
    assert valid == []
    assert len(warns) == 1
    assert "x" in warns[0]
    assert "y" in warns[0]


# -- Missing key rejection ---------------------------------------------------

def test_missing_objective_rejected_with_reason() -> None:
    """A candidate missing a required objective key is rejected."""
    cands = [_cand("ok", x=1.0, y=2.0), _cand("partial", x=1.0)]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 1
    assert valid[0]["candidate_id"] == "ok"
    assert "partial" in warns[0]
    assert "missing" in warns[0].lower()


def test_missing_multiple_keys_reports_all() -> None:
    """When several objectives are missing, the warning lists them."""
    cand = _cand("sparse", z=1.0)
    valid, warns = validate_candidates([cand], ["x", "y", "z"])
    assert valid == []
    assert "x" in warns[0]
    assert "y" in warns[0]


# -- Edge cases --------------------------------------------------------------

def test_empty_candidates_returns_empty() -> None:
    """Empty input produces empty output and no warnings."""
    valid, warns = validate_candidates([], ["x", "y"])
    assert valid == []
    assert warns == []


def test_all_invalid_returns_empty_valid_list() -> None:
    """When every candidate is invalid, valid list is empty but warnings exist."""
    cands = [
        {"candidate_id": "nan_one", "scores": {"x": float("nan")}},
        _cand("missing_one"),
    ]
    valid, warns = validate_candidates(cands, ["x"])
    assert valid == []
    assert len(warns) == 2


def test_extra_score_keys_are_ignored() -> None:
    """Candidates with extra keys beyond objective_names pass validation."""
    cand = _cand("extra", x=1.0, y=2.0, bonus=99.0)
    valid, warns = validate_candidates([cand], ["x", "y"])
    assert len(valid) == 1
    assert warns == []


def test_all_valid_candidates_pass_through() -> None:
    """When all candidates are valid, they all pass and no warnings are emitted."""
    cands = [_cand("a", x=1.0, y=2.0), _cand("b", x=3.0, y=4.0)]
    valid, warns = validate_candidates(cands, ["x", "y"])
    assert len(valid) == 2
    assert warns == []


def test_preserves_candidate_order() -> None:
    """Valid candidates are returned in the same order they were provided."""
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
