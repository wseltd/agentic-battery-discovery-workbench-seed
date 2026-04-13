"""Integration tests for the confidence-scored domain router (T007).

Nine tests exercising route_with_confidence against realistic keyword-dense
prompts.  Three molecule routing cases, three materials routing cases, and
three ambiguous/unsupported rejection cases.

Tests call the real router — no mocks.  Prompts are keyword-dense because
the confidence formula (keyword_hits / unigram_count) requires ~80% keyword
density to cross the auto-route threshold.  This is by design: the router
is conservative and only auto-routes when the signal is strong.
"""

from __future__ import annotations

from discovery_workbench.routing.scorer import route_with_confidence


# -- Molecule routing (3 tests) -------------------------------------------
# Each prompt must route to "small_molecule" with confidence >= 0.80
# and action "auto".


def test_molecule_route_smiles_prompt() -> None:
    """SMILES keyword in a keyword-rich drug-design query auto-routes."""
    result = route_with_confidence("SMILES ADMET QED ligand")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "smiles" in result.matched_keywords


def test_molecule_route_logp_property_prompt() -> None:
    """logP property keyword alongside other drug descriptors auto-routes."""
    result = route_with_confidence("logP TPSA lipinski QED")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "logp" in result.matched_keywords


def test_molecule_route_scaffold_optimization_prompt() -> None:
    """Scaffold keyword with SAR and lead-optimisation bigram auto-routes."""
    result = route_with_confidence("scaffold linker SAR lead optimisation")

    assert result.domain == "small_molecule"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "scaffold" in result.matched_keywords


# -- Materials routing (3 tests) ------------------------------------------
# Each prompt must route to "inorganic_materials" with confidence >= 0.80
# and action "auto".


def test_materials_route_crystal_prompt() -> None:
    """Crystal keyword with lattice/VASP/phonon auto-routes to materials."""
    result = route_with_confidence("crystal lattice VASP phonon")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "crystal" in result.matched_keywords


def test_materials_route_space_group_prompt() -> None:
    """Space-group bigram alongside crystal/lattice/CIF auto-routes."""
    result = route_with_confidence("crystal space group lattice CIF")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "space group" in result.matched_keywords


def test_materials_route_element_system_prompt() -> None:
    """Formation-energy bigram with crystal/lattice/CIF auto-routes."""
    result = route_with_confidence("crystal formation energy lattice CIF")

    assert result.domain == "inorganic_materials"
    assert result.confidence >= 0.80
    assert result.action == "auto"
    assert "formation energy" in result.matched_keywords


# -- Ambiguous / unsupported rejection (3 tests) --------------------------
# These prompts must NOT auto-route.  Ambiguity keywords force "clarify";
# unsupported-domain keywords force "unsupported".


def test_ambiguous_battery_electrolyte() -> None:
    """'battery electrolyte' is an ambiguity keyword — must not auto-route."""
    result = route_with_confidence("battery electrolyte")

    assert result.domain is None
    assert result.action == "clarify"
    assert result.confidence < 0.80
    assert "battery electrolyte" in result.ambiguity_hits


def test_unsupported_polymer_prompt() -> None:
    """'polymer' is an unsupported-domain keyword — routed to unsupported."""
    result = route_with_confidence("polymer with Tg")

    assert result.domain is None
    assert result.action == "unsupported"
    assert result.confidence < 0.80


def test_ambiguous_catalyst_prompt() -> None:
    """'catalyst' is an ambiguity keyword — must not auto-route."""
    result = route_with_confidence("catalyst for CO2 reduction")

    assert result.domain is None
    assert result.action == "clarify"
    assert result.confidence < 0.80
    assert "catalyst" in result.ambiguity_hits
