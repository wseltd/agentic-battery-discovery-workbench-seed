"""Tests for molecular and materials benchmark metric aggregation.

Verifies compute_molecular_metrics (T044) and compute_materials_metrics
(T045) on small synthetic batches with hand-computed expected values.
"""

from amdw.molecules.mol_benchmark import compute_molecular_metrics, MolecularMetrics
from amdw.materials.mat_benchmark import compute_materials_metrics, MaterialsMetrics


# ---------------------------------------------------------------------------
# Synthetic batch data — hand-computed expected values documented inline
# ---------------------------------------------------------------------------

# 5 molecule result dicts:
#   3 valid (mols 0, 1, 2), 2 invalid (mols 3, 4)   → validity = 3/5 = 0.6
#   1 duplicate (mol 2), 4 unique                     → uniqueness = 4/5 = 0.8
#   2 novel (mols 0, 1), 3 not novel                  → novelty = 2/5 = 0.4
#   2 meet target (mols 0, 1), 3 do not               → target_satisfaction = 2/5 = 0.4
#   diversity_scores on mols 0, 1, 2: 0.8, 0.6, 0.4  → diversity = 1.8/3 = 0.6
MOLECULAR_BATCH = [
    {
        "is_valid": True,
        "is_duplicate": False,
        "is_novel": True,
        "meets_target": True,
        "diversity_score": 0.8,
    },
    {
        "is_valid": True,
        "is_duplicate": False,
        "is_novel": True,
        "meets_target": True,
        "diversity_score": 0.6,
    },
    {
        "is_valid": True,
        "is_duplicate": True,
        "is_novel": False,
        "meets_target": False,
        "diversity_score": 0.4,
    },
    {
        "is_valid": False,
        "is_duplicate": False,
        "is_novel": False,
        "meets_target": False,
    },
    {
        "is_valid": False,
        "is_duplicate": False,
        "is_novel": False,
        "meets_target": False,
    },
]

# 5 crystal result dicts:
#   4 valid (0, 1, 2, 3), 1 invalid (4)                → validity = 4/5 = 0.8
#   1 duplicate (mol 2), 4 unique                       → uniqueness = 4/5 = 0.8
#   2 novel (mols 0, 1), 3 not novel                    → novelty = 2/5 = 0.4
#   1 stable (mol 0), 4 not stable                      → stability_proxy = 1/5 = 0.2
#   2 meet target (mols 0, 1), 3 do not                 → target_satisfaction = 2/5 = 0.4
MATERIALS_BATCH = [
    {
        "is_valid": True,
        "is_duplicate": False,
        "is_novel": True,
        "meets_stability_threshold": True,
        "meets_target": True,
    },
    {
        "is_valid": True,
        "is_duplicate": False,
        "is_novel": True,
        "meets_stability_threshold": False,
        "meets_target": True,
    },
    {
        "is_valid": True,
        "is_duplicate": True,
        "is_novel": False,
        "meets_stability_threshold": False,
        "meets_target": False,
    },
    {
        "is_valid": True,
        "is_duplicate": False,
        "is_novel": False,
        "meets_stability_threshold": False,
        "meets_target": False,
    },
    {
        "is_valid": False,
        "is_duplicate": False,
        "is_novel": False,
        "meets_stability_threshold": False,
        "meets_target": False,
    },
]


def test_molecular_metrics_on_synthetic_batch() -> None:
    """Verify molecular metric aggregation against hand-computed ratios."""
    result = compute_molecular_metrics(MOLECULAR_BATCH)

    assert isinstance(result, MolecularMetrics)

    # Exact hand-computed values
    assert result.validity == 0.6, f"Expected validity 3/5=0.6, got {result.validity}"
    assert result.uniqueness == 0.8, f"Expected uniqueness 4/5=0.8, got {result.uniqueness}"
    assert result.novelty == 0.4, f"Expected novelty 2/5=0.4, got {result.novelty}"
    assert result.target_satisfaction == 0.4, (
        f"Expected target_satisfaction 2/5=0.4, got {result.target_satisfaction}"
    )
    # Diversity = mean(0.8, 0.6, 0.4) = 0.6
    assert abs(result.diversity - 0.6) < 1e-9, (
        f"Expected diversity mean(0.8,0.6,0.4)=0.6, got {result.diversity}"
    )

    # All fields are fractions in [0.0, 1.0]
    for field_name in ("validity", "uniqueness", "novelty", "diversity", "target_satisfaction"):
        value = getattr(result, field_name)
        assert 0.0 <= value <= 1.0, f"{field_name}={value} outside [0, 1]"


def test_materials_sun_metrics_on_synthetic_batch() -> None:
    """Verify materials S.U.N.-style metric aggregation against hand-computed ratios."""
    result = compute_materials_metrics(MATERIALS_BATCH)

    assert isinstance(result, MaterialsMetrics)

    # Exact hand-computed values
    assert result.validity == 0.8, f"Expected validity 4/5=0.8, got {result.validity}"
    assert result.uniqueness == 0.8, f"Expected uniqueness 4/5=0.8, got {result.uniqueness}"
    assert result.novelty == 0.4, f"Expected novelty 2/5=0.4, got {result.novelty}"
    assert result.stability_proxy == 0.2, (
        f"Expected stability_proxy 1/5=0.2, got {result.stability_proxy}"
    )
    assert result.target_satisfaction == 0.4, (
        f"Expected target_satisfaction 2/5=0.4, got {result.target_satisfaction}"
    )

    # All fields are fractions in [0.0, 1.0]
    for field_name in ("validity", "uniqueness", "novelty", "stability_proxy", "target_satisfaction"):
        value = getattr(result, field_name)
        assert 0.0 <= value <= 1.0, f"{field_name}={value} outside [0, 1]"


def test_empty_input_returns_zero_metrics() -> None:
    """Empty candidate list must return all-zero metrics without ZeroDivisionError."""
    mol_result = compute_molecular_metrics([])
    mat_result = compute_materials_metrics([])

    # Molecular: all fields must be exactly 0.0
    assert mol_result.validity == 0.0, f"mol validity={mol_result.validity}"
    assert mol_result.uniqueness == 0.0, f"mol uniqueness={mol_result.uniqueness}"
    assert mol_result.novelty == 0.0, f"mol novelty={mol_result.novelty}"
    assert mol_result.diversity == 0.0, f"mol diversity={mol_result.diversity}"
    assert mol_result.target_satisfaction == 0.0, (
        f"mol target_satisfaction={mol_result.target_satisfaction}"
    )

    # Materials: all fields must be exactly 0.0
    assert mat_result.validity == 0.0, f"mat validity={mat_result.validity}"
    assert mat_result.uniqueness == 0.0, f"mat uniqueness={mat_result.uniqueness}"
    assert mat_result.novelty == 0.0, f"mat novelty={mat_result.novelty}"
    assert mat_result.stability_proxy == 0.0, (
        f"mat stability_proxy={mat_result.stability_proxy}"
    )
    assert mat_result.target_satisfaction == 0.0, (
        f"mat target_satisfaction={mat_result.target_satisfaction}"
    )
