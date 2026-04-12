"""Tests for the deterministic domain router.

Routing is risky logic — a wrong verdict sends the entire pipeline down
the wrong branch.  Tests are weighted toward edge cases and adversarial
inputs rather than confirmatory happy-path checks.
"""

from __future__ import annotations

import pytest

from discovery_workbench.routing.router import RoutingResult, route_deterministic


# ---------------------------------------------------------------------------
# RoutingResult dataclass tests
# ---------------------------------------------------------------------------


class TestRoutingResultDataclass:
    """Verify RoutingResult structural invariants."""

    def test_routing_result_is_frozen(self) -> None:
        """Mutation of any field must raise FrozenInstanceError."""
        result = RoutingResult(
            domain="small_molecule",
            matched_keywords=frozenset({"clogp"}),
            ambiguity_hits=frozenset(),
            stage="deterministic",
        )
        with pytest.raises(AttributeError):
            result.domain = "inorganic_materials"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.matched_keywords = frozenset()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.stage = "probabilistic"  # type: ignore[misc]

    def test_routing_result_default_construction(self) -> None:
        """Verify all four fields are present with expected types."""
        result = RoutingResult(
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
        """stage must be 'deterministic' — anything else is a bug."""
        with pytest.raises(ValueError, match="stage must be 'deterministic'"):
            RoutingResult(
                domain=None,
                matched_keywords=frozenset(),
                ambiguity_hits=frozenset(),
                stage="probabilistic",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# route_deterministic — core routing tests
# ---------------------------------------------------------------------------


class TestRouteDeterministic:
    """Verify keyword-based domain routing."""

    def test_route_pure_molecule_input(self) -> None:
        """Request with only small-molecule keywords routes correctly."""
        result = route_deterministic(
            "Generate molecules with good cLogP and low TPSA"
        )
        assert result.domain == "small_molecule"
        assert "clogp" in result.matched_keywords
        assert "tpsa" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0
        assert result.stage == "deterministic"

    def test_route_pure_materials_input(self) -> None:
        """Request with only inorganic-materials keywords routes correctly."""
        result = route_deterministic(
            "Design a cubic crystal in Li-Fe-P-O"
        )
        assert result.domain == "inorganic_materials"
        assert "crystal" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0

    def test_route_mixed_domain_returns_none(self) -> None:
        """Keywords from both domains → domain must be None (unresolvable)."""
        result = route_deterministic(
            "I need a SMILES string with good band gap"
        )
        assert result.domain is None
        assert "smiles" in result.matched_keywords
        assert "band gap" in result.matched_keywords
        assert len(result.ambiguity_hits) == 0

    def test_route_ambiguity_blocks_routing(self) -> None:
        """Ambiguity keywords block routing even with domain keywords."""
        result = route_deterministic(
            "Design a catalyst crystal structure"
        )
        assert result.domain is None
        assert "catalyst" in result.ambiguity_hits
        # Domain keywords are still recorded
        assert "crystal" in result.matched_keywords

    def test_route_empty_input_returns_none(self) -> None:
        """Empty string has no keywords to match."""
        result = route_deterministic("")
        assert result.domain is None
        assert len(result.matched_keywords) == 0
        assert len(result.ambiguity_hits) == 0

    def test_route_unsupported_domain(self) -> None:
        """Unsupported keywords route to 'unsupported', not None."""
        result = route_deterministic("Design a polymer membrane")
        assert result.domain == "unsupported"
        assert "polymer" in result.matched_keywords

    def test_route_no_keywords_returns_none(self) -> None:
        """Input with no registry keywords returns None domain."""
        result = route_deterministic("Hello, can you help me?")
        assert result.domain is None
        assert len(result.matched_keywords) == 0
        assert len(result.ambiguity_hits) == 0

    def test_route_case_insensitive(self) -> None:
        """Keywords must match regardless of input casing."""
        result = route_deterministic("Check ADMET properties for LIPINSKI")
        assert result.domain == "small_molecule"
        assert "admet" in result.matched_keywords
        assert "lipinski" in result.matched_keywords

    def test_route_bigram_matching(self) -> None:
        """Multi-word keywords (bigrams) must be detected."""
        result = route_deterministic(
            "Apply lead optimisation to improve hit generation"
        )
        assert result.domain == "small_molecule"
        assert "lead optimisation" in result.matched_keywords
        assert "hit generation" in result.matched_keywords

    def test_route_smiles_string_triggers_molecule(self) -> None:
        """Mention of 'SMILES' in input triggers small_molecule routing."""
        result = route_deterministic(
            "Optimise the SMILES representation CC(=O)Oc1ccccc1C(=O)O"
        )
        assert result.domain == "small_molecule"
        assert "smiles" in result.matched_keywords

    def test_route_cif_mention_triggers_materials(self) -> None:
        """Mention of 'CIF' triggers inorganic_materials routing."""
        result = route_deterministic("Parse the CIF file for analysis")
        assert result.domain == "inorganic_materials"
        assert "cif" in result.matched_keywords

    def test_route_polymer_triggers_unsupported(self) -> None:
        """Polymer is an unsupported domain — should route to 'unsupported'."""
        result = route_deterministic("Synthesise a new polymer")
        assert result.domain == "unsupported"
        assert "polymer" in result.matched_keywords

    def test_route_ambiguity_only_returns_none(self) -> None:
        """Ambiguity keyword alone, no domain keywords → None domain."""
        result = route_deterministic("Design a catalyst")
        assert result.domain is None
        assert "catalyst" in result.ambiguity_hits
        assert len(result.matched_keywords) == 0

    def test_route_multiple_ambiguity_keywords(self) -> None:
        """Multiple ambiguity keywords all recorded in ambiguity_hits."""
        result = route_deterministic(
            "Build a semiconductor nanoparticle catalyst"
        )
        assert result.domain is None
        assert "semiconductor" in result.ambiguity_hits
        assert "nanoparticle" in result.ambiguity_hits
        assert "catalyst" in result.ambiguity_hits

    def test_route_unsupported_plus_routable_is_routable(self) -> None:
        """Unsupported keywords don't block routing when a real domain is present.

        'unsupported' is not a routable domain — if small_molecule keywords
        also appear, the routable domain wins.
        """
        result = route_deterministic(
            "Compare this protein scaffold with SMILES"
        )
        # "protein" -> unsupported, "scaffold" + "smiles" -> small_molecule
        # Only one routable domain, so it routes there
        assert result.domain == "small_molecule"
        assert "smiles" in result.matched_keywords
        assert "scaffold" in result.matched_keywords

    def test_route_stage_always_deterministic(self) -> None:
        """Every result from route_deterministic has stage='deterministic'."""
        inputs = [
            "Generate molecules with TPSA",
            "Design a crystal",
            "Design a catalyst",
            "",
            "Hello world",
        ]
        for text in inputs:
            result = route_deterministic(text)
            assert result.stage == "deterministic", f"Failed for input: {text!r}"
