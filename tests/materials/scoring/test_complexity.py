"""Tests for structural complexity scoring."""

import pytest

from discovery_workbench.materials.scoring.complexity import complexity_score


# ---------------------------------------------------------------------------
# Typical structures — all dimensions within bounds
# ---------------------------------------------------------------------------


def test_typical_ternary_scores_one():
    """A ternary oxide with modest cell and normal density scores 1.0."""
    score = complexity_score(num_elements=3, atoms_per_cell=20, density=5.0)
    assert score == 1.0


def test_unary_at_lower_bound_scores_one():
    """Single-element material at lower bounds still scores 1.0."""
    score = complexity_score(num_elements=1, atoms_per_cell=1, density=0.5)
    assert score == 1.0


def test_upper_bounds_score_one():
    """Values exactly at upper bounds score 1.0."""
    score = complexity_score(num_elements=6, atoms_per_cell=200, density=25.0)
    assert score == 1.0


# ---------------------------------------------------------------------------
# Penalty ramps — values outside bounds
# ---------------------------------------------------------------------------


def test_num_elements_above_upper_penalised():
    """8 elements is 2 beyond upper bound of 6; linear decay toward 12."""
    # At 8: score = (12 - 8) / (12 - 6) = 4/6 ≈ 0.667
    score = complexity_score(num_elements=8, atoms_per_cell=20, density=5.0)
    expected = (1.0 + 1.0 + 4.0 / 6.0) / 3.0
    assert score == pytest.approx(expected)


def test_num_elements_at_double_upper_scores_zero_dimension():
    """At 12 elements (2x upper bound), element sub-score is 0."""
    score = complexity_score(num_elements=12, atoms_per_cell=20, density=5.0)
    # elem=0, cell=1, dens=1 → avg = 2/3
    assert score == pytest.approx(2.0 / 3.0)


def test_density_above_upper_penalised():
    """Density 40 g/cm³ is above upper bound 25; linear decay toward 50."""
    # At 40: dens_score = (50 - 40) / (50 - 25) = 10/25 = 0.4
    score = complexity_score(num_elements=3, atoms_per_cell=20, density=40.0)
    expected = (1.0 + 1.0 + 0.4) / 3.0
    assert score == pytest.approx(expected)


def test_density_below_lower_penalised():
    """Density 0.3 g/cm³ is below lower bound 0.5; linear decay toward 0.25."""
    # At 0.3: dens_score = (0.3 - 0.25) / (0.5 - 0.25) = 0.05/0.25 = 0.2
    score = complexity_score(num_elements=3, atoms_per_cell=20, density=0.3)
    expected = (1.0 + 1.0 + 0.2) / 3.0
    assert score == pytest.approx(expected)


def test_density_at_half_lower_scores_zero_dimension():
    """At density 0.25 (half of lower 0.5), density sub-score is 0."""
    score = complexity_score(num_elements=3, atoms_per_cell=20, density=0.25)
    # elem=1, cell=1, dens=0 → avg = 2/3
    assert score == pytest.approx(2.0 / 3.0)


def test_atoms_per_cell_above_upper_penalised():
    """300 atoms is above upper bound 200; linear decay toward 400."""
    # At 300: cell_score = (400 - 300) / (400 - 200) = 100/200 = 0.5
    score = complexity_score(num_elements=3, atoms_per_cell=300, density=5.0)
    expected = (1.0 + 0.5 + 1.0) / 3.0
    assert score == pytest.approx(expected)


def test_all_dimensions_penalised():
    """When all dimensions violate bounds, score reflects all penalties."""
    # elem=10: score = (12-10)/6 = 2/6 ≈ 0.333
    # cell=350: score = (400-350)/200 = 50/200 = 0.25
    # dens=45: score = (50-45)/25 = 5/25 = 0.2
    score = complexity_score(num_elements=10, atoms_per_cell=350, density=45.0)
    expected = (2.0 / 6.0 + 0.25 + 0.2) / 3.0
    assert score == pytest.approx(expected)


def test_extreme_values_clamp_to_zero_subscore():
    """Values far beyond 2x bounds still produce non-negative avg."""
    # elem=20: 20 >= 12 → 0; cell=500: 500 >= 400 → 0; dens=60: 60 >= 50 → 0
    score = complexity_score(num_elements=20, atoms_per_cell=500, density=60.0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Custom bounds
# ---------------------------------------------------------------------------


def test_custom_bounds_override_defaults():
    """Custom bounds change the scoring thresholds."""
    # Upper density bound = 10.0; at density 15: score = (20-15)/10 = 0.5
    custom = {"density": (0.5, 10.0)}
    score = complexity_score(
        num_elements=3, atoms_per_cell=20, density=15.0, bounds=custom,
    )
    expected = (1.0 + 1.0 + 0.5) / 3.0
    assert score == pytest.approx(expected)


def test_custom_bounds_partial_override():
    """Overriding one key preserves defaults for the others."""
    custom = {"num_elements": (1.0, 10.0)}
    # 8 elements is now within [1, 10] → score 1.0
    score = complexity_score(
        num_elements=8, atoms_per_cell=20, density=5.0, bounds=custom,
    )
    assert score == 1.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_num_elements():
    """Zero or negative num_elements raises ValueError."""
    with pytest.raises(ValueError, match="num_elements must be positive") as exc_info:
        complexity_score(num_elements=0, atoms_per_cell=20, density=5.0)
    assert "num_elements must be positive" in str(exc_info.value)

    with pytest.raises(ValueError, match="num_elements must be positive") as exc_info:
        complexity_score(num_elements=-1, atoms_per_cell=20, density=5.0)
    assert "num_elements must be positive" in str(exc_info.value)


def test_rejects_nonpositive_atoms_per_cell():
    """Zero or negative atoms_per_cell raises ValueError."""
    with pytest.raises(ValueError, match="atoms_per_cell must be positive") as exc_info:
        complexity_score(num_elements=3, atoms_per_cell=0, density=5.0)
    assert "atoms_per_cell must be positive" in str(exc_info.value)

    with pytest.raises(ValueError, match="atoms_per_cell must be positive") as exc_info:
        complexity_score(num_elements=3, atoms_per_cell=-5, density=5.0)
    assert "atoms_per_cell must be positive" in str(exc_info.value)


def test_rejects_nonpositive_density():
    """Zero or negative density raises ValueError."""
    with pytest.raises(ValueError, match="density must be positive") as exc_info:
        complexity_score(num_elements=3, atoms_per_cell=20, density=0.0)
    assert "density must be positive" in str(exc_info.value)

    with pytest.raises(ValueError, match="density must be positive") as exc_info:
        complexity_score(num_elements=3, atoms_per_cell=20, density=-1.0)
    assert "density must be positive" in str(exc_info.value)


def test_rejects_unknown_bounds_keys():
    """Unknown keys in bounds dict raise ValueError."""
    with pytest.raises(ValueError, match="Unknown bounds keys") as exc_info:
        complexity_score(
            num_elements=3, atoms_per_cell=20, density=5.0,
            bounds={"volume": (1.0, 100.0)},
        )
    assert "Unknown bounds keys" in str(exc_info.value)


def test_rejects_inverted_bounds():
    """Bounds where lower > upper raise ValueError."""
    with pytest.raises(ValueError, match="inverted") as exc_info:
        complexity_score(
            num_elements=3, atoms_per_cell=20, density=5.0,
            bounds={"density": (25.0, 0.5)},
        )
    assert "inverted" in str(exc_info.value)


def test_rejects_negative_bounds():
    """Negative bound values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative") as exc_info:
        complexity_score(
            num_elements=3, atoms_per_cell=20, density=5.0,
            bounds={"density": (-1.0, 25.0)},
        )
    assert "non-negative" in str(exc_info.value)
