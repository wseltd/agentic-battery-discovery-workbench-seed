"""Tests for QED and SA score property calculators."""

from __future__ import annotations

import rdkit
import pytest
from rdkit.Chem import MolFromSmiles

from workbench.molecules.property_scores import (
    PropertyScore,
    compute_qed,
    compute_sa_score,
)
from workbench.shared.evidence import EvidenceLevel


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture()
def benzene():
    return MolFromSmiles("c1ccccc1")


@pytest.fixture()
def aspirin():
    return MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture()
def ethanol():
    """Simple molecule — expected to have a low SA score (easy to make)."""
    return MolFromSmiles("CCO")


@pytest.fixture()
def taxol():
    """Complex natural product — expected to have a high SA score."""
    return MolFromSmiles(
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)"
        "C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)"
        "OC(=O)C)O)C)OC(=O)C"
    )


# --- QED tests --------------------------------------------------------------


class TestQedValueRange:
    def test_qed_returns_value_between_zero_and_one(self, benzene):
        result = compute_qed(benzene)
        assert 0.0 <= result.value <= 1.0


class TestQedEvidenceLevel:
    def test_qed_evidence_level_is_heuristic_estimated(self, benzene):
        result = compute_qed(benzene)
        assert result.evidence_level is EvidenceLevel.HEURISTIC_ESTIMATED


class TestQedToolVersion:
    def test_qed_tool_version_matches_rdkit(self, benzene):
        result = compute_qed(benzene)
        assert result.tool_version == rdkit.__version__


class TestQedNoneMol:
    def test_qed_none_mol_raises(self):
        with pytest.raises(TypeError, match="NoneType") as exc_info:
            compute_qed(None)
        assert "NoneType" in str(exc_info.value)


class TestQedKnownMolecule:
    def test_qed_known_molecule_aspirin(self, aspirin):
        """Aspirin has a well-characterised QED around 0.46."""
        result = compute_qed(aspirin)
        assert 0.3 <= result.value <= 0.7, (
            f"Aspirin QED {result.value} outside expected range"
        )


# --- SA score tests ----------------------------------------------------------


class TestSaScoreValueRange:
    def test_sa_score_returns_value_between_one_and_ten(self, benzene):
        result = compute_sa_score(benzene)
        assert 1.0 <= result.value <= 10.0


class TestSaScoreEasyMolecule:
    def test_sa_score_easy_molecule_scores_low(self, ethanol):
        """Ethanol is trivial to synthesise — score should be well below 3."""
        result = compute_sa_score(ethanol)
        assert result.value < 3.0, (
            f"Ethanol SA score {result.value} unexpectedly high"
        )


class TestSaScoreComplexMolecule:
    def test_sa_score_complex_molecule_scores_higher(self, ethanol, taxol):
        """Taxol (complex) must score higher than ethanol (trivial)."""
        easy = compute_sa_score(ethanol)
        hard = compute_sa_score(taxol)
        assert hard.value > easy.value


class TestSaScoreEvidenceLevel:
    def test_sa_score_evidence_level_is_heuristic_estimated(self, benzene):
        result = compute_sa_score(benzene)
        assert result.evidence_level is EvidenceLevel.HEURISTIC_ESTIMATED


class TestSaScoreToolVersion:
    def test_sa_score_tool_version_matches_rdkit(self, benzene):
        result = compute_sa_score(benzene)
        assert result.tool_version == rdkit.__version__


class TestSaScoreNoneMol:
    def test_sa_score_none_mol_raises(self):
        with pytest.raises(TypeError, match="NoneType") as exc_info:
            compute_sa_score(None)
        assert "NoneType" in str(exc_info.value)


# --- PropertyScore dataclass tests ------------------------------------------


class TestPropertyScoreNameField:
    def test_property_score_name_field_populated(self, benzene):
        qed_result = compute_qed(benzene)
        sa_result = compute_sa_score(benzene)
        assert isinstance(qed_result, PropertyScore)
        assert isinstance(sa_result, PropertyScore)
        assert qed_result.name == "QED"
        assert sa_result.name == "SA_Score"
