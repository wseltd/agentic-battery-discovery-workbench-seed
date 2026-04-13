"""Tests for the deterministic domain router."""

from __future__ import annotations

import pytest

from agentic_discovery_core.routing.router import DeterministicRoutingResult, route_deterministic


class TestDeterministicRoutingResultDataclass:
    def test_routing_result_is_frozen(self) -> None:
        result = DeterministicRoutingResult(
            domain="small_molecule",
            matched_keywords=frozenset({"clogp"}),
            ambiguity_hits=frozenset(),
            stage="deterministic",
        )
        with pytest.raises(AttributeError):
            result.domain = "inorganic_materials"
        with pytest.raises(AttributeError):
            result.matched_keywords = frozenset()
        with pytest.raises(AttributeError):
            result.stage = "probabilistic"
        assert result.domain == "small_molecule"
        assert result.matched_keywords == frozenset({"clogp"})
        assert result.stage == "deterministic"

    def test_routing_result_default_construction(self) -> None:
        result = DeterministicRoutingResult(
            domain=None,
            matched_keywords=frozenset(),
            ambiguity_hits=frozenset(),
            stage="deterministic",
        )
        assert result.domain is None
        assert isinstance(result.matched_keywords, frozenset)
        assert isinstance(result.ambiguity_hits, frozenset)
        assert result.stage == "deterministic"

    def test_routing_result_rejects_wrong_stage(self) -> None:
        with pytest.raises(ValueError, match="stage must be 'deterministic'") as exc_info:
            DeterministicRoutingResult(
                domain=None,
                matched_keywords=frozenset(),
                ambiguity_hits=frozenset(),
                stage="probabilistic",
            )
        assert "probabilistic" in str(exc_info.value)


class TestRouteDeterministic:
    def test_route_pure_molecule_input(self) -> None:
        result = route_deterministic(
            "Generate molecules with good cLogP and low TPSA"
        )
        assert result.domain == "small_molecule"
        assert "clogp" in result.matched_keywords
        assert "tpsa" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0
        assert result.stage == "deterministic"

    def test_route_pure_materials_input(self) -> None:
        result = route_deterministic(
            "Design a cubic crystal in Li-Fe-P-O"
        )
        assert result.domain == "inorganic_materials"
        assert "crystal" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0

    def test_route_mixed_domain_returns_none(self) -> None:
        result = route_deterministic(
            "I need a SMILES string with good band gap"
        )
        assert result.domain is None
        assert "smiles" in result.matched_keywords
        assert "band gap" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0

    def test_route_ambiguity_blocks_routing(self) -> None:
        result = route_deterministic(
            "Design a catalyst crystal structure"
        )
        assert result.domain is None
        assert "catalyst" in result.ambiguity_hits
        assert "crystal" in result.matched_keywords

    def test_route_empty_input_returns_none(self) -> None:
        result = route_deterministic("")
        assert result.domain is None
        assert len(result.matched_keywords) == 0
        assert len(result.ambiguity_hits) == 0

    def test_route_unsupported_domain(self) -> None:
        result = route_deterministic("Design a polymer membrane")
        assert result.domain == "unsupported"
        assert "polymer" in result.matched_keywords

    def test_route_no_keywords_returns_none(self) -> None:
        result = route_deterministic("Hello, can you help me?")
        assert result.domain is None
        assert len(result.matched_keywords) == 0
        assert len(result.ambiguity_hits) == 0

    def test_route_case_insensitive(self) -> None:
        result = route_deterministic("Check ADMET properties for LIPINSKI")
        assert result.domain == "small_molecule"
        assert "admet" in result.matched_keywords
        assert "lipinski" in result.matched_keywords

    def test_route_bigram_matching(self) -> None:
        result = route_deterministic(
            "Apply lead optimisation to improve hit generation"
        )
        assert result.domain == "small_molecule"
        assert "lead optimisation" in result.matched_keywords
        assert "hit generation" in result.matched_keywords

    def test_route_smiles_string_triggers_molecule(self) -> None:
        result = route_deterministic(
            "Optimise the SMILES representation CC(=O)Oc1ccccc1C(=O)O"
        )
        assert result.domain == "small_molecule"
        assert "smiles" in result.matched_keywords

    def test_route_cif_mention_triggers_materials(self) -> None:
        result = route_deterministic("Parse the CIF file for analysis")
        assert result.domain == "inorganic_materials"
        assert "cif" in result.matched_keywords

    def test_route_polymer_triggers_unsupported(self) -> None:
        result = route_deterministic("Synthesise a new polymer")
        assert result.domain == "unsupported"
        assert "polymer" in result.matched_keywords

    def test_route_ambiguity_only_returns_none(self) -> None:
        result = route_deterministic("Design a catalyst")
        assert result.domain is None
        assert "catalyst" in result.ambiguity_hits
        assert len(result.matched_keywords) == 0

    def test_route_multiple_ambiguity_keywords(self) -> None:
        result = route_deterministic(
            "Build a semiconductor nanoparticle catalyst"
        )
        assert result.domain is None
        assert "semiconductor" in result.ambiguity_hits
        assert "nanoparticle" in result.ambiguity_hits
        assert "catalyst" in result.ambiguity_hits

    def test_route_unsupported_plus_routable_is_routable(self) -> None:
        result = route_deterministic(
            "Compare this protein scaffold with SMILES"
        )
        assert result.domain == "small_molecule"
        assert "smiles" in result.matched_keywords
        assert "scaffold" in result.matched_keywords

    def test_route_stage_always_deterministic(self) -> None:
        inputs = [
            "Generate molecules with TPSA",
            "Design a crystal",
            "Design a catalyst",
            "",
            "Hello world",
        ]
        for text in inputs:
            result = route_deterministic(text)
            assert result.stage == "deterministic"
