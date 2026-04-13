"""Integration tests for the confidence-scored domain router."""

from __future__ import annotations

from agentic_discovery_core.routing.scorer import route_with_confidence


def test_molecule_route_smiles_prompt() -> None:
    result = route_with_confidence("SMILES ADMET QED ligand")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "smiles" in result.matched_keywords


def test_molecule_route_logp_property_prompt() -> None:
    result = route_with_confidence("logP TPSA lipinski QED")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "logp" in result.matched_keywords


def test_molecule_route_scaffold_optimization_prompt() -> None:
    result = route_with_confidence("scaffold linker SAR lead optimisation")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "scaffold" in result.matched_keywords


def test_materials_route_crystal_prompt() -> None:
    result = route_with_confidence("crystal lattice VASP phonon")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "crystal" in result.matched_keywords


def test_materials_route_space_group_prompt() -> None:
    result = route_with_confidence("crystal space group lattice CIF")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "space group" in result.matched_keywords


def test_materials_route_element_system_prompt() -> None:
    result = route_with_confidence("crystal formation energy lattice CIF")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "formation energy" in result.matched_keywords


def test_ambiguous_battery_electrolyte() -> None:
    result = route_with_confidence("battery electrolyte")

    assert result.domain is None
    assert result.action == "clarify"
    assert result.confidence < 0.80
    assert "battery electrolyte" in result.ambiguity_hits


def test_unsupported_polymer_prompt() -> None:
    result = route_with_confidence("polymer with Tg")

    assert result.domain is None
    assert result.action == "unsupported"
    assert result.confidence < 0.80


def test_ambiguous_catalyst_prompt() -> None:
    result = route_with_confidence("catalyst for CO2 reduction")

    assert result.domain is None
    assert result.action == "clarify"
    assert result.confidence < 0.80
    assert "catalyst" in result.ambiguity_hits
