"""Known-value tests for molecular property calculators.

Reference values are pre-computed with RDKit and hardcoded.
Each assertion uses pytest.approx with explicit tolerances.
"""

from __future__ import annotations

import pytest
from rdkit.Chem import MolFromSmiles

from agentic_molecule_discovery.properties.molecular_weight import calc_molecular_weight
from agentic_molecule_discovery.properties.clogp import calc_clogp
from agentic_molecule_discovery.properties.tpsa import calc_tpsa
from agentic_molecule_discovery.scoring.property_scores import compute_qed, compute_sa_score
from agentic_molecule_discovery.novelty.pains_filter import run_pains_filter

# Canonical SMILES — stable across RDKit versions.
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# Pre-computed RDKit reference values (RDKit 2024.03+).
ASPIRIN_MW = 180.042259
ASPIRIN_LOGP = 1.3101
ASPIRIN_TPSA = 63.60
ASPIRIN_QED = 0.5501
ASPIRIN_SA = 1.5800

CAFFEINE_QED = 0.5385
CAFFEINE_SA = 2.2980

# Tolerances per the design spec.
MW_TOL = 0.01
LOGP_TOL = 0.1
TPSA_TOL = 0.1
QED_TOL = 0.01
SA_TOL = 0.1


def test_molecular_weight_known_value() -> None:
    """Aspirin exact molecular weight should match the RDKit-computed reference."""
    mol = MolFromSmiles(ASPIRIN_SMILES)
    mw = calc_molecular_weight(mol)
    assert mw == pytest.approx(ASPIRIN_MW, abs=MW_TOL), (
        f"Aspirin MW {mw} not within ±{MW_TOL} of {ASPIRIN_MW}"
    )


def test_logp_known_value() -> None:
    """Aspirin Crippen cLogP should match the RDKit-computed reference."""
    mol = MolFromSmiles(ASPIRIN_SMILES)
    logp = calc_clogp(mol)
    assert logp == pytest.approx(ASPIRIN_LOGP, abs=LOGP_TOL), (
        f"Aspirin cLogP {logp} not within ±{LOGP_TOL} of {ASPIRIN_LOGP}"
    )


def test_tpsa_known_value() -> None:
    """Aspirin TPSA should match the RDKit-computed reference."""
    mol = MolFromSmiles(ASPIRIN_SMILES)
    tpsa = calc_tpsa(mol)
    assert tpsa == pytest.approx(ASPIRIN_TPSA, abs=TPSA_TOL), (
        f"Aspirin TPSA {tpsa} not within ±{TPSA_TOL} of {ASPIRIN_TPSA}"
    )


def test_qed_known_value() -> None:
    """QED for aspirin and caffeine should match RDKit-computed references.

    Both molecules exercise QED's composite weighting across different
    descriptor profiles — aspirin has moderate lipophilicity and low MW,
    caffeine has negative logP and heterocyclic complexity.
    """
    aspirin = MolFromSmiles(ASPIRIN_SMILES)
    caffeine = MolFromSmiles(CAFFEINE_SMILES)

    aspirin_qed = compute_qed(aspirin)
    caffeine_qed = compute_qed(caffeine)

    assert aspirin_qed.value == pytest.approx(ASPIRIN_QED, abs=QED_TOL), (
        f"Aspirin QED {aspirin_qed.value} not within ±{QED_TOL} of {ASPIRIN_QED}"
    )
    assert caffeine_qed.value == pytest.approx(CAFFEINE_QED, abs=QED_TOL), (
        f"Caffeine QED {caffeine_qed.value} not within ±{QED_TOL} of {CAFFEINE_QED}"
    )
    # Both should be in the valid [0, 1] range
    assert 0.0 <= aspirin_qed.value <= 1.0
    assert 0.0 <= caffeine_qed.value <= 1.0


def test_sa_score_known_value() -> None:
    """SA scores for aspirin and caffeine should match RDKit-computed references.

    Aspirin is synthetically accessible (low score ~1.6).  Caffeine is
    slightly harder (score ~2.3) due to its fused heterocyclic core.
    Both exercise different fragment-contribution paths in the SA scorer.
    """
    aspirin = MolFromSmiles(ASPIRIN_SMILES)
    caffeine = MolFromSmiles(CAFFEINE_SMILES)

    aspirin_sa = compute_sa_score(aspirin)
    caffeine_sa = compute_sa_score(caffeine)

    assert aspirin_sa.value == pytest.approx(ASPIRIN_SA, abs=SA_TOL), (
        f"Aspirin SA {aspirin_sa.value} not within ±{SA_TOL} of {ASPIRIN_SA}"
    )
    assert caffeine_sa.value == pytest.approx(CAFFEINE_SA, abs=SA_TOL), (
        f"Caffeine SA {caffeine_sa.value} not within ±{SA_TOL} of {CAFFEINE_SA}"
    )
    # SA scores are in [1, 10]; both reference molecules should be easy (< 4)
    assert 1.0 <= aspirin_sa.value < 4.0
    assert 1.0 <= caffeine_sa.value < 4.0
    # Caffeine harder than aspirin due to fused purine ring system
    assert caffeine_sa.value > aspirin_sa.value


def test_pains_filter_positive_hit() -> None:
    """An ene-rhodanine derivative must trigger a PAINS alert.

    5-benzylidenerhodanine is a textbook PAINS-positive scaffold
    (ene_rhod_E family) flagged in all standard PAINS filter sets.
    """
    rhodanine = MolFromSmiles("O=C1NC(=S)/C(=C/c2ccccc2)S1")
    result = run_pains_filter(rhodanine)
    assert result.passed is False, "Ene-rhodanine should fail the PAINS filter"
    assert len(result.matched_filters) >= 1
    assert any("ene_rhod" in name for name in result.matched_filters), (
        f"Expected ene_rhod match, got: {result.matched_filters}"
    )
