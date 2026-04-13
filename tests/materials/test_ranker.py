"""Tests for materials property ranker."""

import pytest

from discovery_workbench.materials.ranker import (
    NONCONVERGED_STABILITY_SCORE,
    PARTIAL_SYMMETRY_SCORE,
    STABILITY_THRESHOLD_EV,
    MaterialsPropertyRanker,
    RankedCandidate,
    compute_complexity_score,
    compute_stability_score,
    compute_symmetry_score,
    compute_target_satisfaction_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(**overrides):
    """Build a RankedCandidate with sensible defaults for field tests."""
    defaults = dict(
        candidate_id="test",
        composition="NaCl",
        space_group_number=225,
        stability_score=0.9,
        symmetry_score=1.0,
        complexity_score=0.8,
        target_satisfaction_score=1.0,
        composite_score=0.9,
    )
    defaults.update(overrides)
    return RankedCandidate(**defaults)


def _make_input(
    candidate_id,
    energy_above_hull=0.01,
    space_group_number=225,
    num_atoms=5,
    volume=60.0,
    composition="BaTiO3",
    **kwargs,
):
    """Build a candidate input dict for MaterialsPropertyRanker."""
    cand = {
        "candidate_id": candidate_id,
        "composition": composition,
        "space_group_number": space_group_number,
        "energy_above_hull": energy_above_hull,
        "num_atoms": num_atoms,
        "volume": volume,
    }
    cand.update(kwargs)
    return cand


# ---------------------------------------------------------------------------
# Stability scoring
# ---------------------------------------------------------------------------


def test_stability_score_penalises_above_threshold():
    """Energy above threshold yields zero score."""
    score = compute_stability_score(STABILITY_THRESHOLD_EV * 2)
    assert score == 0.0


def test_stability_score_rewards_below_threshold():
    """Energy well below threshold yields high score via linear decay."""
    score = compute_stability_score(0.02)
    expected = 1.0 - 0.02 / STABILITY_THRESHOLD_EV
    assert score == pytest.approx(expected)
    assert score > 0.7


def test_stability_score_handles_nonconverged():
    """Non-converged relaxation gets fixed penalty score."""
    score = compute_stability_score(0.0, converged=False)
    assert score == NONCONVERGED_STABILITY_SCORE


def test_stability_score_on_hull():
    """Zero energy above hull gives perfect score."""
    assert compute_stability_score(0.0) == 1.0


def test_stability_score_rejects_negative():
    """Negative energy above hull raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        compute_stability_score(-0.01)
    assert "energy_above_hull must be >= 0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Symmetry scoring
# ---------------------------------------------------------------------------


def test_symmetry_score_exact_match():
    """Exact space group match scores 1.0."""
    assert compute_symmetry_score(225, 225) == 1.0


def test_symmetry_score_same_system_partial():
    """Same crystal system but different space group scores partial credit."""
    # 225 (Fm-3m) and 229 (Im-3m) are both cubic (195-230)
    score = compute_symmetry_score(225, 229)
    assert score == PARTIAL_SYMMETRY_SCORE


def test_symmetry_score_mismatch_zero():
    """Different crystal systems score 0.0."""
    # 225 is cubic, 62 is orthorhombic
    assert compute_symmetry_score(225, 62) == 0.0


def test_symmetry_score_no_target():
    """No target preference yields 1.0 (no penalty)."""
    assert compute_symmetry_score(225, None) == 1.0


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------


def test_complexity_score_typical_structure():
    """Typical perovskite cell (5 atoms, ~60 A^3) scores 1.0."""
    # 60 / 5 = 12 A^3/atom — well within 7-40 range
    score = compute_complexity_score(5, 60.0)
    assert score == 1.0


def test_complexity_score_extreme_density():
    """Unrealistically dense structure is penalised below 1.0."""
    # 5.0 / 5 = 1.0 A^3/atom — far below MIN_VOLUME_PER_ATOM (7)
    score = compute_complexity_score(5, 5.0)
    assert score == pytest.approx(1.0 / 7.0, rel=1e-6)
    assert score < 0.2


def test_complexity_score_rejects_nonpositive_volume():
    """Non-positive volume raises ValueError with descriptive message."""
    with pytest.raises(ValueError) as exc_info:
        compute_complexity_score(4, 0.0)
    assert "volume must be positive" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        compute_complexity_score(4, -10.0)
    assert "volume must be positive" in str(exc_info.value)


def test_complexity_score_rejects_nonpositive_atoms():
    """Non-positive atom count raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        compute_complexity_score(0, 60.0)
    assert "num_atoms must be positive" in str(exc_info.value)


def test_complexity_score_sparse_structure():
    """Unrealistically sparse structure is penalised."""
    # 500 / 2 = 250 A^3/atom — far above MAX_VOLUME_PER_ATOM (40)
    score = compute_complexity_score(2, 500.0)
    assert score == pytest.approx(40.0 / 250.0, rel=1e-6)
    assert score < 0.2


# ---------------------------------------------------------------------------
# Target satisfaction scoring
# ---------------------------------------------------------------------------


def test_target_satisfaction_all_met():
    """All targets met scores 1.0."""
    targets = {"band_gap": 1.0, "bulk_modulus": 100.0}
    achieved = {"band_gap": 1.5, "bulk_modulus": 150.0}
    assert compute_target_satisfaction_score(targets, achieved) == 1.0


def test_target_satisfaction_partial():
    """One of two targets met scores 0.5."""
    targets = {"band_gap": 1.0, "bulk_modulus": 200.0}
    achieved = {"band_gap": 1.5, "bulk_modulus": 100.0}
    assert compute_target_satisfaction_score(targets, achieved) == 0.5


def test_target_satisfaction_none_met():
    """No targets met scores 0.0."""
    targets = {"band_gap": 3.0}
    achieved = {"band_gap": 1.0}
    assert compute_target_satisfaction_score(targets, achieved) == 0.0


def test_target_satisfaction_missing_property():
    """Missing property in achieved counts as unmet."""
    targets = {"band_gap": 1.0, "shear_modulus": 50.0}
    achieved = {"band_gap": 2.0}
    assert compute_target_satisfaction_score(targets, achieved) == 0.5


def test_target_satisfaction_rejects_invalid():
    """Empty targets dict raises ValueError with descriptive message."""
    with pytest.raises(ValueError) as exc_info:
        compute_target_satisfaction_score({}, {"band_gap": 1.0})
    assert "targets must not be empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# RankedCandidate dataclass
# ---------------------------------------------------------------------------


def test_ranked_candidate_band_gap_none_by_default():
    """Band gap defaults to None; explicit value is stored exactly."""
    rc_default = _make_candidate()
    assert rc_default.band_gap_eV is None

    # Verify explicit value round-trips correctly
    rc_with_gap = _make_candidate(candidate_id="with_gap", band_gap_eV=1.12)
    assert rc_with_gap.band_gap_eV == 1.12


def test_ranked_candidate_evidence_level_heuristic():
    """Evidence level defaults to 'heuristic_estimated'."""
    rc = _make_candidate()
    assert rc.evidence_level == "heuristic_estimated"


def test_ranked_candidate_deferred_constant():
    """Class constant for DFT-pending fields has expected value."""
    assert RankedCandidate.DEFERRED_EVIDENCE_LEVEL == "requested"


def test_ranked_candidate_rank_defaults_zero():
    """Rank defaults to 0 (unranked)."""
    rc = _make_candidate()
    assert rc.rank == 0


# ---------------------------------------------------------------------------
# MaterialsPropertyRanker integration
# ---------------------------------------------------------------------------


def test_rank_candidates_sorts_descending():
    """Candidates are sorted by composite score in descending order."""
    candidates = [
        _make_input("low", energy_above_hull=0.08),
        _make_input("high", energy_above_hull=0.0),
        _make_input("mid", energy_above_hull=0.04),
    ]
    ranker = MaterialsPropertyRanker()
    result = ranker.rank_candidates(candidates)

    scores = [r.composite_score for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].candidate_id == "high"


def test_rank_candidates_assigns_rank_positions():
    """Rank positions are 1-based and sequential."""
    candidates = [
        _make_input("a", energy_above_hull=0.0),
        _make_input("b", energy_above_hull=0.05),
    ]
    ranker = MaterialsPropertyRanker()
    result = ranker.rank_candidates(candidates)

    ranks = [r.rank for r in result]
    assert ranks == [1, 2]


def test_custom_weights_change_ranking():
    """Different weight profiles can reverse candidate ordering."""
    # Candidate A: great stability (0.0 eV), poor symmetry (orthorhombic vs cubic target)
    # Candidate B: poor stability (0.08 eV), great symmetry (cubic = target)
    candidates = [
        _make_input("stable", energy_above_hull=0.0, space_group_number=62),
        _make_input("symmetric", energy_above_hull=0.08, space_group_number=225),
    ]

    stability_heavy = MaterialsPropertyRanker(
        weights={"stability": 0.9, "symmetry": 0.01}
    )
    symmetry_heavy = MaterialsPropertyRanker(
        weights={"stability": 0.01, "symmetry": 0.9}
    )

    result_stab = stability_heavy.rank_candidates(
        candidates, target_space_group=225,
    )
    result_sym = symmetry_heavy.rank_candidates(
        candidates, target_space_group=225,
    )

    assert result_stab[0].candidate_id == "stable"
    assert result_sym[0].candidate_id == "symmetric"


def test_empty_candidates_returns_empty():
    """Empty candidate list returns empty result."""
    ranker = MaterialsPropertyRanker()
    result = ranker.rank_candidates([])
    assert result == []
