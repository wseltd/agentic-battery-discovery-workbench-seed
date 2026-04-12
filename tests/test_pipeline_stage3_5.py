"""Integration tests for the molecular validity pipeline (stages 3-5).

Verifies that the pipeline correctly wires formal_charge, stereocentre,
and salt_strip stages — invokes each in order, returns structured results,
and handles molecules that trigger different combinations of stage outcomes.
"""

from __future__ import annotations

from rdkit import Chem

from molecular_validity.pipeline import (
    PipelineResult,
    StageResult,
    get_registered_stages,
    run_pipeline,
)
from molecular_validity.salt_strip import SaltStripResult
from molecular_validity.stereocentre import StereocentreReport
from validation.models import ValidationResult


# --- helpers -----------------------------------------------------------------

def _mol(smiles: str) -> Chem.Mol:
    """Parse SMILES, failing the test on bad input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    return mol


# --- stage registration ------------------------------------------------------

class TestStageRegistration:
    """The pipeline must expose exactly stages 3, 4, 5 in that order."""

    def test_registered_stage_numbers(self):
        stages = get_registered_stages()
        numbers = [num for num, _ in stages]
        assert numbers == [3, 4, 5]

    def test_registered_stage_names(self):
        stages = get_registered_stages()
        names = [name for _, name in stages]
        assert names == ["formal_charge", "stereocentre", "salt_strip"]

    def test_no_duplicate_stage_numbers(self):
        stages = get_registered_stages()
        numbers = [num for num, _ in stages]
        assert len(numbers) == len(set(numbers))


# --- execution order ---------------------------------------------------------

class TestExecutionOrder:
    """Stages must execute in ascending stage-number order."""

    def test_result_order_matches_registration(self):
        result = run_pipeline(_mol("CCO"))
        stage_nums = [sr.stage for sr in result.stage_results]
        assert stage_nums == [3, 4, 5]

    def test_result_names_match_registration(self):
        result = run_pipeline(_mol("CCO"))
        names = [sr.name for sr in result.stage_results]
        assert names == ["formal_charge", "stereocentre", "salt_strip"]


# --- all stages invoked (clean molecule) -------------------------------------

class TestCleanMolecule:
    """A simple, clean molecule should pass through all stages without errors."""

    def test_all_three_stages_run(self):
        result = run_pipeline(_mol("CCO"))
        assert len(result.stage_results) == 3

    def test_formal_charge_passes(self):
        result = run_pipeline(_mol("CCO"))
        fc_result = result.stage_results[0].result
        assert isinstance(fc_result, ValidationResult)
        assert fc_result.is_valid is True
        assert fc_result.errors == []

    def test_stereocentre_returns_report(self):
        result = run_pipeline(_mol("CCO"))
        sc_result = result.stage_results[1].result
        assert isinstance(sc_result, StereocentreReport)
        # Ethanol has no chiral centres
        assert sc_result.has_stereocentres is False

    def test_salt_strip_returns_result(self):
        result = run_pipeline(_mol("CCO"))
        ss_result = result.stage_results[2].result
        assert isinstance(ss_result, SaltStripResult)
        # Single fragment — nothing to strip
        assert ss_result.fragments_removed == 0

    def test_pipeline_preserves_input_mol(self):
        mol = _mol("CCO")
        result = run_pipeline(mol)
        assert result.mol is mol


# --- formal charge failure does not block later stages -----------------------

class TestFormalChargeFailure:
    """When formal_charge rejects a molecule, stereocentre and salt_strip
    must still execute — the pipeline is collect-all, not fail-fast."""

    def test_all_stages_run_despite_charge_failure(self):
        # Mg2+ has charge +2, outside default [-1, +1]
        mol = _mol("[Mg+2]")
        result = run_pipeline(mol)
        assert len(result.stage_results) == 3

    def test_charge_stage_reports_failure(self):
        mol = _mol("[Mg+2]")
        result = run_pipeline(mol)
        fc = result.stage_results[0].result
        assert isinstance(fc, ValidationResult)
        assert fc.is_valid is False
        assert len(fc.errors) >= 1

    def test_subsequent_stages_still_produce_results(self):
        mol = _mol("[Mg+2]")
        result = run_pipeline(mol)
        sc = result.stage_results[1].result
        assert isinstance(sc, StereocentreReport)
        # Mg2+ is a single atom — no stereocentres
        assert sc.has_stereocentres is False
        ss = result.stage_results[2].result
        assert isinstance(ss, SaltStripResult)
        # Mg2+ is a single ionic fragment — salt_strip removes it entirely
        assert ss.original_fragment_count == 1
        assert ss.fragments_removed == 1


# --- molecule with stereocentres --------------------------------------------

class TestStereocentreDetection:
    """Pipeline correctly propagates stereocentre information."""

    def test_chiral_molecule_flagged(self):
        # L-alanine has one chiral centre
        mol = _mol("[C@@H](N)(C)C(=O)O")
        result = run_pipeline(mol)
        sc = result.stage_results[1].result
        assert isinstance(sc, StereocentreReport)
        assert sc.has_stereocentres is True
        assert sc.total >= 1


# --- salt-containing molecule ------------------------------------------------

class TestSaltStripping:
    """Pipeline correctly strips salt counterions."""

    def test_salt_form_stripped(self):
        # Sodium phenoxide: [Na+].[O-]c1ccccc1
        mol = _mol("[Na+].[O-]c1ccccc1")
        result = run_pipeline(mol)
        ss = result.stage_results[2].result
        assert isinstance(ss, SaltStripResult)
        assert ss.original_fragment_count == 2
        assert ss.fragments_removed == 1
        assert ss.mol is not None

    def test_single_fragment_not_stripped(self):
        mol = _mol("c1ccccc1")
        result = run_pipeline(mol)
        ss = result.stage_results[2].result
        assert ss.fragments_removed == 0
        assert ss.original_fragment_count == 1


# --- combined edge case: charged salt with stereocentre ----------------------

class TestCombinedEdgeCase:
    """A molecule triggering multiple stages simultaneously should produce
    correct results from every stage."""

    def test_charged_salt_with_stereocentre(self):
        # Sodium salt of a chiral amino acid fragment:
        # [Na+].N[C@@H](CC)C(=O)[O-]
        # - formal_charge: Na+ is +1, O- is -1 — both within default range
        # - stereocentre: one specified chiral centre
        # - salt_strip: two fragments, sodium removed
        mol = _mol("[Na+].N[C@@H](CC)C(=O)[O-]")
        result = run_pipeline(mol)

        fc = result.stage_results[0].result
        assert fc.is_valid is True

        sc = result.stage_results[1].result
        assert sc.has_stereocentres is True

        ss = result.stage_results[2].result
        assert ss.original_fragment_count == 2
        assert ss.fragments_removed == 1


# --- return type structure ---------------------------------------------------

class TestReturnTypes:
    """Pipeline return types must match the documented contract."""

    def test_pipeline_result_type(self):
        mol = _mol("C")
        result = run_pipeline(mol)
        assert isinstance(result, PipelineResult)
        assert result.mol is mol
        assert len(result.stage_results) == 3

    def test_stage_result_type(self):
        result = run_pipeline(_mol("C"))
        expected = [(3, "formal_charge"), (4, "stereocentre"), (5, "salt_strip")]
        for sr, (num, name) in zip(result.stage_results, expected):
            assert isinstance(sr, StageResult)
            assert sr.stage == num
            assert sr.name == name
