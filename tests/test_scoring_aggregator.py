"""Tests for the molecular scoring aggregator.

Focuses on weight validation edge cases, PAINS clamping behaviour,
NaN handling, and correct ranking — the areas where silent bugs
would produce misleading drug-candidate rankings.
"""

from __future__ import annotations

import math

import pytest

from agentic_discovery.molecules.scoring_aggregator import (
    MolecularScoringAggregator,
    ScoringWeights,
    score_molecules,
    validate_weights,
)
from discovery_workbench.evidence import EvidenceLevel


def _mol(
    smiles: str = "CCO",
    pains_pass: bool = True,
    property_score: float = 0.5,
    novelty_penalty: float = 0.5,
    diversity_reward: float = 0.5,
) -> dict[str, object]:
    """Build a minimal molecule dict for testing."""
    return {
        "smiles": smiles,
        "pains_pass": pains_pass,
        "property_score": property_score,
        "novelty_penalty": novelty_penalty,
        "diversity_reward": diversity_reward,
    }


class TestCompositeScoreWithinZeroOne:
    """Composite scores must always land in [0, 1]."""

    def test_composite_score_within_zero_one(self) -> None:
        results = score_molecules(
            [_mol(property_score=1.0, novelty_penalty=1.0, diversity_reward=1.0)],
            ScoringWeights(),
        )
        assert 0.0 <= results[0].composite_score <= 1.0

    def test_composite_score_at_lower_bound(self) -> None:
        results = score_molecules(
            [_mol(property_score=0.0, novelty_penalty=0.0, diversity_reward=0.0)],
            ScoringWeights(),
        )
        assert results[0].composite_score >= 0.0


class TestPainsClamping:
    """PAINS failure must hard-clamp the composite to zero."""

    def test_pains_fail_clamps_score_to_zero(self) -> None:
        results = score_molecules(
            [_mol(pains_pass=False, property_score=1.0, novelty_penalty=1.0, diversity_reward=1.0)],
            ScoringWeights(),
        )
        assert results[0].composite_score == 0.0

    def test_pains_pass_true_does_not_clamp(self) -> None:
        results = score_molecules(
            [_mol(pains_pass=True, property_score=0.8, novelty_penalty=0.6, diversity_reward=0.7)],
            ScoringWeights(),
        )
        assert results[0].composite_score > 0.0


class TestWeightValidation:
    """validate_weights must reject negative and all-zero configurations."""

    def test_all_zero_weights_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="All weights are zero"):
            validate_weights(ScoringWeights(0.0, 0.0, 0.0, 0.0))

    def test_negative_weight_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            validate_weights(ScoringWeights(property_score=-1.0))

    def test_valid_weights_pass(self) -> None:
        # Must not raise — a single positive weight is valid.
        # Prove it by scoring with these weights and checking a result exists.
        weights = ScoringWeights(1.0, 0.0, 0.0, 0.0)
        validate_weights(weights)
        results = score_molecules([_mol()], weights)
        assert len(results) == 1


class TestNanHandling:
    """NaN component scores must degrade gracefully, not poison composites."""

    def test_nan_component_treated_as_zero(self) -> None:
        results = score_molecules(
            [_mol(property_score=float("nan"), novelty_penalty=0.8, diversity_reward=0.8)],
            ScoringWeights(),
        )
        # NaN property_score → 0, so composite should still be finite and valid
        assert math.isfinite(results[0].composite_score)
        assert results[0].component_scores["property_score"] == 0.0


class TestSorting:
    """Results must be sorted descending by composite score."""

    def test_results_sorted_descending_by_composite(self) -> None:
        mols = [
            _mol(smiles="A", property_score=0.1, novelty_penalty=0.1, diversity_reward=0.1),
            _mol(smiles="B", property_score=0.9, novelty_penalty=0.9, diversity_reward=0.9),
            _mol(smiles="C", property_score=0.5, novelty_penalty=0.5, diversity_reward=0.5),
        ]
        results = score_molecules(mols, ScoringWeights())
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].smiles == "B"
        assert results[-1].smiles == "A"


class TestWeightBehaviour:
    """Weights must control relative component contribution."""

    def test_equal_weights_equal_contribution(self) -> None:
        """With equal weights and pains_pass=True, each component contributes equally."""
        mol = _mol(property_score=1.0, novelty_penalty=0.0, diversity_reward=0.0)
        results = score_molecules([mol], ScoringWeights(1.0, 1.0, 1.0, 1.0))
        # property=1.0, pains_pass_contribution=1.0, novelty=0.0, diversity=0.0
        # weighted sum = 1+1+0+0 = 2, total_weight = 4, composite = 0.5
        assert results[0].composite_score == pytest.approx(0.5)

    def test_zero_weight_excludes_component(self) -> None:
        """A zero-weighted component must not affect the composite."""
        mol = _mol(property_score=0.0, novelty_penalty=1.0, diversity_reward=1.0)
        # With property and pains weights zeroed, only novelty and diversity matter
        results = score_molecules(
            [mol],
            ScoringWeights(property_score=0.0, pains_penalty=0.0, novelty_penalty=1.0, diversity_reward=1.0),
        )
        assert results[0].composite_score == pytest.approx(1.0)

    def test_custom_weights_shift_ranking(self) -> None:
        """Changing weights must change which molecule ranks first."""
        mol_a = _mol(smiles="A", property_score=1.0, novelty_penalty=0.0, diversity_reward=0.0)
        mol_b = _mol(smiles="B", property_score=0.0, novelty_penalty=1.0, diversity_reward=0.0)

        # Property-heavy weights → A wins
        results_prop = score_molecules(
            [mol_a, mol_b],
            ScoringWeights(property_score=10.0, pains_penalty=0.0, novelty_penalty=1.0, diversity_reward=0.0),
        )
        assert results_prop[0].smiles == "A"

        # Novelty-heavy weights → B wins
        results_nov = score_molecules(
            [mol_a, mol_b],
            ScoringWeights(property_score=1.0, pains_penalty=0.0, novelty_penalty=10.0, diversity_reward=0.0),
        )
        assert results_nov[0].smiles == "B"


class TestEvidenceLevel:
    """ScoredMolecule must always carry HEURISTIC_ESTIMATED evidence."""

    def test_evidence_level_is_heuristic_estimated(self) -> None:
        results = score_molecules([_mol()], ScoringWeights())
        assert results[0].evidence_level == EvidenceLevel.HEURISTIC_ESTIMATED


class TestSingleMolecule:
    """Single-molecule scoring must work without ranking artefacts."""

    def test_single_molecule_scores_correctly(self) -> None:
        results = score_molecules(
            [_mol(property_score=0.6, novelty_penalty=0.4, diversity_reward=0.8)],
            ScoringWeights(1.0, 1.0, 1.0, 1.0),
        )
        assert len(results) == 1
        # (0.6 + 1.0 + 0.4 + 0.8) / 4.0 = 0.7
        assert results[0].composite_score == pytest.approx(0.7)
        assert results[0].smiles == "CCO"
        assert results[0].pains_pass is True


class TestAggregatorClass:
    """MolecularScoringAggregator wraps score_molecules with stored weights."""

    def test_aggregator_delegates_to_score_molecules(self) -> None:
        agg = MolecularScoringAggregator(ScoringWeights())
        results = agg.score([_mol()])
        assert len(results) == 1
        assert results[0].composite_score > 0.0

    def test_aggregator_rejects_invalid_weights(self) -> None:
        with pytest.raises(ValueError):
            MolecularScoringAggregator(ScoringWeights(0.0, 0.0, 0.0, 0.0))
