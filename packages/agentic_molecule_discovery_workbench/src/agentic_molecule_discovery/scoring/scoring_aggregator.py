"""Molecular scoring aggregator.

Combines per-molecule component scores (property fitness, PAINS,
novelty, diversity) into a single composite score using configurable
weights.  Composite scores are clamped to [0, 1].  Molecules that
fail PAINS filtering are clamped to zero — a hard gate, not a soft
penalty, because PAINS hits indicate likely assay interference that
no amount of good property scores can compensate for.

NaN component scores are treated as zero rather than propagating,
since a missing component (e.g. novelty not yet computed) should not
poison the entire composite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agentic_molecule_discovery.evidence import EvidenceLevel


@dataclass
class ScoringWeights:
    """Per-component weights for composite score calculation.

    Parameters
    ----------
    property_score:
        Weight for the property-fitness component.
    pains_penalty:
        Weight for the PAINS penalty component.
    novelty_penalty:
        Weight for the novelty penalty component.
    diversity_reward:
        Weight for the diversity reward component.
    """

    property_score: float = 1.0
    pains_penalty: float = 1.0
    novelty_penalty: float = 1.0
    diversity_reward: float = 1.0


@dataclass(frozen=True)
class ScoredMolecule:
    """A molecule with an aggregated composite score.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.
    composite_score:
        Weighted, clamped composite score in [0, 1].
    component_scores:
        Individual component scores before weighting.
    pains_pass:
        Whether the molecule passed PAINS filtering.
    evidence_level:
        Always ``HEURISTIC_ESTIMATED`` — composite scoring is a
        heuristic aggregation, not a physics-based calculation.
    """

    smiles: str
    composite_score: float
    component_scores: dict[str, float]
    pains_pass: bool
    evidence_level: EvidenceLevel = field(
        default=EvidenceLevel.HEURISTIC_ESTIMATED, init=False
    )


def validate_weights(weights: ScoringWeights) -> None:
    """Check that scoring weights are usable.

    Raises
    ------
    ValueError
        If any weight is negative or all weights are zero.
    """
    fields = {
        "property_score": weights.property_score,
        "pains_penalty": weights.pains_penalty,
        "novelty_penalty": weights.novelty_penalty,
        "diversity_reward": weights.diversity_reward,
    }
    for name, value in fields.items():
        if value < 0:
            raise ValueError(
                f"Weight '{name}' is negative ({value}); all weights must be >= 0"
            )
    if all(v == 0 for v in fields.values()):
        raise ValueError(
            "All weights are zero; at least one weight must be positive"
        )


def _safe_score(value: float) -> float:
    """Return *value* if finite, else 0.0."""
    if math.isfinite(value):
        return value
    return 0.0


def score_molecules(
    molecules: list[dict[str, object]],
    weights: ScoringWeights,
) -> list[ScoredMolecule]:
    """Score and rank a batch of molecules.

    Parameters
    ----------
    molecules:
        Each dict must contain at minimum ``"smiles"`` (str),
        ``"pains_pass"`` (bool), and zero or more component scores
        keyed as ``"property_score"``, ``"novelty_score"``,
        ``"diversity_reward"``.
    weights:
        Relative importance of each component.

    Returns
    -------
    list[ScoredMolecule]
        Scored molecules sorted descending by composite score.
    """
    validate_weights(weights)

    scored: list[ScoredMolecule] = []
    for mol in molecules:
        smiles = str(mol["smiles"])
        pains_pass = bool(mol["pains_pass"])

        components: dict[str, float] = {}
        raw_property = _safe_score(float(mol.get("property_score", 0.0)))  # type: ignore[arg-type]
        raw_novelty = _safe_score(float(mol.get("novelty_score", 0.0)))  # type: ignore[arg-type]
        raw_diversity = _safe_score(float(mol.get("diversity_reward", 0.0)))  # type: ignore[arg-type]

        components["property_score"] = raw_property
        components["pains_penalty"] = 1.0 if pains_pass else 0.0
        components["novelty_score"] = raw_novelty
        components["diversity_reward"] = raw_diversity

        # Weighted sum, normalised by total active weight.
        weighted_sum = (
            weights.property_score * raw_property
            + weights.pains_penalty * (1.0 if pains_pass else 0.0)
            + weights.novelty_penalty * raw_novelty
            + weights.diversity_reward * raw_diversity
        )
        total_weight = (
            weights.property_score
            + weights.pains_penalty
            + weights.novelty_penalty
            + weights.diversity_reward
        )
        composite = weighted_sum / total_weight

        # PAINS failure is a hard gate — clamp to zero.
        if not pains_pass:
            composite = 0.0

        # Clamp to [0, 1].
        composite = max(0.0, min(1.0, composite))

        scored.append(
            ScoredMolecule(
                smiles=smiles,
                composite_score=composite,
                component_scores=components,
                pains_pass=pains_pass,
            )
        )

    scored.sort(key=lambda m: m.composite_score, reverse=True)
    return scored


class MolecularScoringAggregator:
    """Stateful wrapper around :func:`score_molecules`.

    Holds a fixed set of weights so callers do not need to pass
    them on every call.
    """

    def __init__(self, weights: ScoringWeights) -> None:
        validate_weights(weights)
        self.weights = weights

    def score_molecules(
        self,
        molecules: list[dict[str, object]],
    ) -> list[ScoredMolecule]:
        """Score molecules using the stored weights."""
        return score_molecules(molecules, self.weights)
