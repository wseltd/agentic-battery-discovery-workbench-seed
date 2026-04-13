"""Tests for the confidence-scored domain router."""

from __future__ import annotations

import dataclasses

import pytest

from agentic_discovery_core.routing.scorer import (
    CONFIDENCE_AUTO_THRESHOLD,
    CONFIDENCE_CLARIFY_THRESHOLD,
    ScoredRoutingResult,
    route_with_confidence,
)


class TestThresholdConstants:
    def test_auto_threshold_value(self) -> None:
        assert CONFIDENCE_AUTO_THRESHOLD == 0.80

    def test_clarify_threshold_value(self) -> None:
        assert CONFIDENCE_CLARIFY_THRESHOLD == 0.55

    def test_auto_above_clarify(self) -> None:
        assert CONFIDENCE_AUTO_THRESHOLD > CONFIDENCE_CLARIFY_THRESHOLD


class TestConfidenceBoundaries:
    def test_confidence_above_auto_threshold(self) -> None:
        result = route_with_confidence("smiles clogp tpsa hbd hba qed")
        assert result.action == "auto"
        assert result.confidence >= CONFIDENCE_AUTO_THRESHOLD
        assert result.domain == "small_molecule"

    def test_confidence_at_exact_auto_boundary(self) -> None:
        result = route_with_confidence("smiles clogp tpsa hbd something")
        assert result.confidence == pytest.approx(0.80)
        assert result.action == "auto"

    def test_confidence_in_clarify_range(self) -> None:
        result = route_with_confidence("smiles clogp tpsa random words")
        assert CONFIDENCE_CLARIFY_THRESHOLD <= result.confidence < CONFIDENCE_AUTO_THRESHOLD
        assert result.action == "clarify"

    def test_confidence_at_exact_clarify_boundary(self) -> None:
        mol_kws = "smiles clogp tpsa hbd hba qed pains scaffold ligand docking admet"
        filler = "foo bar baz qux corge grault garply waldo fred"
        result = route_with_confidence(f"{mol_kws} {filler}")
        assert result.confidence == pytest.approx(0.55)
        assert result.action == "clarify"

    def test_confidence_below_clarify_threshold(self) -> None:
        result = route_with_confidence(
            "smiles foo bar baz qux quux corge grault garply waldo"
        )
        assert result.confidence < CONFIDENCE_CLARIFY_THRESHOLD
        assert result.action == "unsupported"

    def test_single_keyword_low_confidence(self) -> None:
        result = route_with_confidence(
            "please help me with crystal in this long request about nothing"
        )
        assert result.confidence < CONFIDENCE_CLARIFY_THRESHOLD
        assert len(result.matched_keywords) == 1


class TestEdgeCases:
    def test_empty_input_zero_confidence(self) -> None:
        result = route_with_confidence("")
        assert result.confidence == 0.0
        assert result.action == "unsupported"
        assert result.matched_keywords == frozenset()

    def test_scored_result_is_frozen(self) -> None:
        result = route_with_confidence("smiles")
        with pytest.raises(dataclasses.FrozenInstanceError) as exc_info:
            result.domain = "hacked"
        assert exc_info.value is not None
        assert result.domain != "hacked"

    def test_confidence_clamped_zero_to_one(self) -> None:
        result = route_with_confidence("smiles clogp tpsa hbd hba")
        assert 0.0 <= result.confidence <= 1.0

    def test_stage_is_scored(self) -> None:
        result = route_with_confidence("smiles")
        assert result.stage == "scored"

    def test_invalid_stage_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage must be 'scored'") as exc_info:
            ScoredRoutingResult(
                domain=None,
                confidence=0.5,
                action="clarify",
                matched_keywords=frozenset(),
                ambiguity_hits=frozenset(),
                stage="deterministic",
            )
        assert "deterministic" in str(exc_info.value)

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in") as exc_info:
            ScoredRoutingResult(
                domain=None,
                confidence=1.5,
                action="clarify",
                matched_keywords=frozenset(),
                ambiguity_hits=frozenset(),
                stage="scored",
            )
        assert "1.5" in str(exc_info.value)


class TestDomainRouting:
    def test_ambiguity_forces_clarify_even_high_confidence(self) -> None:
        result = route_with_confidence("catalyst smiles clogp tpsa hbd hba")
        assert result.action == "clarify"
        assert "catalyst" in result.ambiguity_hits

    def test_unsupported_domain_forces_unsupported_action(self) -> None:
        result = route_with_confidence("polymer protein biologics")
        assert result.action == "unsupported"
        assert result.domain is None

    def test_mixed_domain_forces_clarify(self) -> None:
        result = route_with_confidence("smiles crystal lattice clogp tpsa")
        assert result.action == "clarify"
        assert result.domain is None

    def test_pure_molecule_high_density_auto(self) -> None:
        result = route_with_confidence(
            "smiles clogp tpsa hbd hba qed pains scaffold ligand docking"
        )
        assert result.action == "auto"
        assert result.domain == "small_molecule"
        assert result.confidence >= CONFIDENCE_AUTO_THRESHOLD

    def test_pure_materials_high_density_auto(self) -> None:
        result = route_with_confidence(
            "crystal lattice cif poscar vasp phonon"
        )
        assert result.action == "auto"
        assert result.domain == "inorganic_materials"
        assert result.confidence >= CONFIDENCE_AUTO_THRESHOLD
