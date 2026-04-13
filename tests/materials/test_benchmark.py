"""Tests for materials benchmark metrics."""

from __future__ import annotations

from discovery_workbench.materials.benchmark import (
    MaterialsBenchmarkReport,
    compute_materials_benchmark,
)


# ---------------------------------------------------------------------------
# Helper to build candidate dicts concisely
# ---------------------------------------------------------------------------

def _candidate(
    *,
    is_valid: bool = False,
    is_duplicate: bool = False,
    is_novel: bool = False,
    is_stable: bool = False,
    satisfies_target: bool = False,
) -> dict[str, object]:
    return {
        "is_valid": is_valid,
        "is_duplicate": is_duplicate,
        "is_novel": is_novel,
        "is_stable": is_stable,
        "satisfies_target": satisfies_target,
    }


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------

def test_empty_input_returns_zero_metrics() -> None:
    result = compute_materials_benchmark([])

    assert isinstance(result, MaterialsBenchmarkReport)
    assert result.validity_pct == 0.0
    assert result.uniqueness_pct == 0.0
    assert result.novelty_pct == 0.0
    assert result.stability_proxy_pct == 0.0
    assert result.target_satisfaction_pct == 0.0
    assert result.shortlist_usefulness == 0.0
    assert result.dft_conversion_rate is None


def test_single_candidate_all_true() -> None:
    c = _candidate(
        is_valid=True,
        is_duplicate=False,
        is_novel=True,
        is_stable=True,
        satisfies_target=True,
    )
    result = compute_materials_benchmark([c])

    assert result.validity_pct == 1.0
    assert result.uniqueness_pct == 1.0
    assert result.novelty_pct == 1.0
    assert result.stability_proxy_pct == 1.0
    assert result.target_satisfaction_pct == 1.0
    assert result.shortlist_usefulness == 1.0


def test_single_candidate_all_false() -> None:
    c = _candidate(
        is_valid=False,
        is_duplicate=True,
        is_novel=False,
        is_stable=False,
        satisfies_target=False,
    )
    result = compute_materials_benchmark([c])

    assert result.validity_pct == 0.0
    assert result.uniqueness_pct == 0.0
    assert result.novelty_pct == 0.0
    assert result.stability_proxy_pct == 0.0
    assert result.target_satisfaction_pct == 0.0
    assert result.shortlist_usefulness == 0.0


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def test_all_valid_candidates_returns_100_validity() -> None:
    candidates = [_candidate(is_valid=True) for _ in range(5)]
    result = compute_materials_benchmark(candidates)

    assert result.validity_pct == 1.0


def test_mixed_validity_flags() -> None:
    candidates = [
        _candidate(is_valid=True),
        _candidate(is_valid=False),
        _candidate(is_valid=True),
        _candidate(is_valid=False),
    ]
    result = compute_materials_benchmark(candidates)

    assert result.validity_pct == 0.5


def test_novelty_pct_counts_is_novel_flag() -> None:
    candidates = [
        _candidate(is_novel=True),
        _candidate(is_novel=True),
        _candidate(is_novel=False),
    ]
    result = compute_materials_benchmark(candidates)

    # 2/3 novel
    assert abs(result.novelty_pct - 2.0 / 3.0) < 1e-9


def test_uniqueness_pct_counts_not_duplicate() -> None:
    candidates = [
        _candidate(is_duplicate=False),
        _candidate(is_duplicate=True),
        _candidate(is_duplicate=False),
        _candidate(is_duplicate=False),
    ]
    result = compute_materials_benchmark(candidates)

    # 3/4 unique (not duplicate)
    assert result.uniqueness_pct == 0.75


def test_stability_proxy_pct() -> None:
    candidates = [
        _candidate(is_stable=True),
        _candidate(is_stable=True),
        _candidate(is_stable=False),
    ]
    result = compute_materials_benchmark(candidates)

    assert abs(result.stability_proxy_pct - 2.0 / 3.0) < 1e-9


def test_target_satisfaction_pct() -> None:
    candidates = [
        _candidate(satisfies_target=True),
        _candidate(satisfies_target=False),
    ]
    result = compute_materials_benchmark(candidates)

    assert result.target_satisfaction_pct == 0.5


# ---------------------------------------------------------------------------
# Shortlist usefulness — requires ALL five flags passing
# ---------------------------------------------------------------------------

def test_shortlist_usefulness_requires_all_passing() -> None:
    """A candidate missing any single flag should not count toward shortlist."""
    # All flags set except is_stable
    almost = _candidate(
        is_valid=True,
        is_duplicate=False,
        is_novel=True,
        is_stable=False,
        satisfies_target=True,
    )
    result = compute_materials_benchmark([almost])

    assert result.shortlist_usefulness == 0.0


def test_shortlist_usefulness_partial_pass() -> None:
    perfect = _candidate(
        is_valid=True,
        is_duplicate=False,
        is_novel=True,
        is_stable=True,
        satisfies_target=True,
    )
    # Fails novelty
    imperfect = _candidate(
        is_valid=True,
        is_duplicate=False,
        is_novel=False,
        is_stable=True,
        satisfies_target=True,
    )
    result = compute_materials_benchmark([perfect, imperfect])

    assert result.shortlist_usefulness == 0.5


# ---------------------------------------------------------------------------
# DFT conversion rate
# ---------------------------------------------------------------------------

def test_dft_conversion_always_none() -> None:
    candidates = [
        _candidate(is_valid=True, is_novel=True, is_stable=True),
        _candidate(is_valid=False),
    ]
    result = compute_materials_benchmark(candidates)

    assert result.dft_conversion_rate is None
    # Verify other metrics computed correctly alongside the None rate
    assert result.validity_pct == 0.5
    assert result.novelty_pct == 0.5


# ---------------------------------------------------------------------------
# Dataclass clamping
# ---------------------------------------------------------------------------

def test_clamping_negative_values() -> None:
    """Values below 0.0 are clamped to 0.0."""
    report = MaterialsBenchmarkReport(
        validity_pct=-0.5,
        uniqueness_pct=-1.0,
        novelty_pct=0.5,
        stability_proxy_pct=0.0,
        target_satisfaction_pct=0.0,
        shortlist_usefulness=0.0,
        dft_conversion_rate=None,
    )
    assert report.validity_pct == 0.0
    assert report.uniqueness_pct == 0.0
    assert report.novelty_pct == 0.5


def test_clamping_values_above_one() -> None:
    """Values above 1.0 are clamped to 1.0."""
    report = MaterialsBenchmarkReport(
        validity_pct=1.5,
        uniqueness_pct=2.0,
        novelty_pct=0.8,
        stability_proxy_pct=0.0,
        target_satisfaction_pct=0.0,
        shortlist_usefulness=0.0,
        dft_conversion_rate=None,
    )
    assert report.validity_pct == 1.0
    assert report.uniqueness_pct == 1.0
    assert report.novelty_pct == 0.8


def test_clamping_skips_none_dft_rate() -> None:
    """dft_conversion_rate=None is not clamped; float fields keep their values."""
    report = MaterialsBenchmarkReport(
        validity_pct=0.5,
        uniqueness_pct=0.5,
        novelty_pct=0.5,
        stability_proxy_pct=0.5,
        target_satisfaction_pct=0.5,
        shortlist_usefulness=0.5,
        dft_conversion_rate=None,
    )
    assert report.dft_conversion_rate is None
    # Float fields survive clamping unchanged when already in [0, 1]
    assert report.validity_pct == 0.5
    assert report.shortlist_usefulness == 0.5
