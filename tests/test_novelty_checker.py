"""Tests for ChEMBL novelty classification.

Heavy coverage on threshold boundaries and classification precedence —
the risk surface is in the boundary logic (off-by-one at 0.40 and 0.70)
and in exact-match taking precedence over Tanimoto score.
"""

from __future__ import annotations

import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from rdkit.DataStructs import ExplicitBitVect

from workbench.molecules.novelty_checker import (
    CLOSE_ANALOGUE_THRESHOLD,
    NOVEL_LIKE_THRESHOLD,
    ChEMBLNoveltyChecker,
    NoveltyClass,
    NoveltyResult,
    _MORGAN_NBITS,
    _MORGAN_RADIUS,
    _classify_tanimoto,
)
from workbench.shared.evidence import EvidenceLevel


# --- Helpers ----------------------------------------------------------------


def _mol_to_ref_tuple(smiles: str) -> tuple[str, str, ExplicitBitVect]:
    """Convert a SMILES string to a (smiles, inchikey, fingerprint) tuple."""
    mol = MolFromSmiles(smiles)
    inchikey = InchiToInchiKey(MolToInchi(mol))
    fp = GetMorganFingerprintAsBitVect(mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS)
    return (smiles, inchikey, fp)


def _build_checker(smiles_list: list[str], tmp_path, reference_db: str = "ChEMBL_36"):
    """Build a ChEMBLNoveltyChecker from SMILES via pickle round-trip."""
    ref_tuples = [_mol_to_ref_tuple(s) for s in smiles_list]
    pkl_path = tmp_path / "ref.pkl"
    ChEMBLNoveltyChecker.build_reference(ref_tuples, pkl_path)
    return ChEMBLNoveltyChecker(pkl_path, reference_db=reference_db)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture()
def aspirin():
    return MolFromSmiles("CC(=O)Oc1ccccc1C(O)=O")


@pytest.fixture()
def benzene():
    return MolFromSmiles("c1ccccc1")


@pytest.fixture()
def ethanol():
    return MolFromSmiles("CCO")


@pytest.fixture()
def lorazepam():
    return MolFromSmiles("OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O")


@pytest.fixture()
def checker_with_aspirin(aspirin, tmp_path):
    """Checker with aspirin as the single reference compound."""
    return _build_checker(["CC(=O)Oc1ccccc1C(O)=O"], tmp_path)


@pytest.fixture()
def diverse_checker(tmp_path):
    """Checker with structurally diverse references for max_tanimoto tests."""
    smiles = [
        "c1ccccc1",                  # benzene
        "CC(=O)Oc1ccccc1C(O)=O",    # aspirin
        "CCCCCCCCCC",                # decane
    ]
    return _build_checker(smiles, tmp_path)


# --- Tests ------------------------------------------------------------------


def test_exact_known_by_inchikey_match(checker_with_aspirin, aspirin):
    """A molecule identical to a reference compound is EXACT_KNOWN."""
    result = checker_with_aspirin.check(aspirin)
    assert result.classification is NoveltyClass.EXACT_KNOWN
    assert result.max_tanimoto == 1.0


def test_close_analogue_above_070_threshold(lorazepam, tmp_path):
    """A molecule structurally similar to a reference (Tanimoto >= 0.70) is CLOSE_ANALOGUE.

    Lorazepam and oxazepam share a benzodiazepine core — lorazepam has an
    extra chlorine on the phenyl ring, giving Tanimoto ~0.75 (above 0.70).
    """
    checker = _build_checker(
        ["OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O"], tmp_path
    )
    oxazepam = MolFromSmiles("OC1N=C(c2ccccc2)c2cc(Cl)ccc2NC1=O")
    result = checker.check(oxazepam)
    assert result.classification is NoveltyClass.CLOSE_ANALOGUE
    assert result.max_tanimoto >= CLOSE_ANALOGUE_THRESHOLD


def test_novel_like_below_040_threshold(tmp_path):
    """A molecule structurally distant from all references (Tanimoto < 0.40) is NOVEL_LIKE.

    Adamantane vs. a long-chain alkene — very different structural scaffolds.
    """
    checker = _build_checker(
        ["C=CCCCCCCCCCCCCCCCCC"], tmp_path  # long alkene
    )
    # Adamantane — rigid cage, very different from linear alkene
    adamantane = MolFromSmiles("C1C2CC3CC1CC(C2)C3")
    result = checker.check(adamantane)
    assert result.classification is NoveltyClass.NOVEL_LIKE
    assert result.max_tanimoto < NOVEL_LIKE_THRESHOLD


def test_gap_region_classifies_as_close_analogue():
    """A molecule in the gap region (0.40 <= Tanimoto < 0.70) is CLOSE_ANALOGUE.

    Uses _classify_tanimoto directly for deterministic boundary testing
    without molecule pair uncertainty.
    """
    # Mid-gap value
    assert _classify_tanimoto(0.55) is NoveltyClass.CLOSE_ANALOGUE
    # Just above lower bound
    assert _classify_tanimoto(0.40) is NoveltyClass.CLOSE_ANALOGUE
    # Just below upper bound
    assert _classify_tanimoto(0.699) is NoveltyClass.CLOSE_ANALOGUE


def test_exact_known_takes_precedence_over_tanimoto(checker_with_aspirin, aspirin):
    """InChIKey match must return EXACT_KNOWN regardless of Tanimoto score.

    Even though Tanimoto would also be 1.0, the classification should be
    EXACT_KNOWN (not CLOSE_ANALOGUE), proving the exact-match stage runs first.
    """
    result = checker_with_aspirin.check(aspirin)
    assert result.classification is NoveltyClass.EXACT_KNOWN
    # Not just a close analogue with Tanimoto == 1.0
    assert result.classification is not NoveltyClass.CLOSE_ANALOGUE


def test_max_tanimoto_is_highest_across_reference(diverse_checker, tmp_path):
    """max_tanimoto must be the highest similarity across all reference molecules.

    Toluene is most similar to benzene in the diverse reference set.
    """
    toluene = MolFromSmiles("Cc1ccccc1")
    result = diverse_checker.check(toluene)
    # Toluene should have highest similarity to benzene
    assert result.max_tanimoto > 0.0
    # Verify it's higher than what we'd get against just decane
    decane_only = _build_checker(["CCCCCCCCCC"], tmp_path)
    decane_result = decane_only.check(toluene)
    assert result.max_tanimoto >= decane_result.max_tanimoto


def test_closest_inchikey_populated(lorazepam, tmp_path):
    """closest_inchikey must be populated when a reference set exists."""
    checker = _build_checker(
        ["OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O"], tmp_path
    )
    oxazepam = MolFromSmiles("OC1N=C(c2ccccc2)c2cc(Cl)ccc2NC1=O")
    result = checker.check(oxazepam)
    assert result.closest_inchikey is not None
    # InChIKey format: 14 chars - 10 chars - 1 char
    assert len(result.closest_inchikey) == 27
    assert result.closest_inchikey.count("-") == 2


def test_reference_db_field_is_chembl_36(tmp_path):
    """Default reference_db must be 'ChEMBL_36'."""
    checker = _build_checker(["CCO"], tmp_path)
    result = checker.check(MolFromSmiles("CCCO"))
    assert result.reference_db == "ChEMBL_36"


def test_evidence_level_is_heuristic_estimated(checker_with_aspirin, aspirin):
    """Evidence level must always be HEURISTIC_ESTIMATED."""
    result = checker_with_aspirin.check(aspirin)
    assert result.evidence_level is EvidenceLevel.HEURISTIC_ESTIMATED


def test_empty_reference_set_classifies_novel_like(tmp_path):
    """An empty reference set means the molecule is NOVEL_LIKE with max_tanimoto=0.0."""
    checker = _build_checker([], tmp_path)
    result = checker.check(MolFromSmiles("CCO"))
    assert result.classification is NoveltyClass.NOVEL_LIKE
    assert result.max_tanimoto == 0.0
    assert result.closest_inchikey is None


def test_boundary_at_exactly_070():
    """Tanimoto of exactly 0.70 must classify as CLOSE_ANALOGUE (inclusive boundary)."""
    assert _classify_tanimoto(0.70) is NoveltyClass.CLOSE_ANALOGUE
    # Just below — still CLOSE_ANALOGUE (gap region)
    assert _classify_tanimoto(0.6999) is NoveltyClass.CLOSE_ANALOGUE


def test_boundary_at_exactly_040():
    """Tanimoto of exactly 0.40 must classify as CLOSE_ANALOGUE (inclusive boundary).

    Just below 0.40 must be NOVEL_LIKE.
    """
    assert _classify_tanimoto(0.40) is NoveltyClass.CLOSE_ANALOGUE
    assert _classify_tanimoto(0.3999) is NoveltyClass.NOVEL_LIKE


def test_fingerprint_params_match_t019():
    """Morgan fingerprint parameters must match T019 DuplicateDetector: radius=2, nBits=2048."""
    assert _MORGAN_RADIUS == 2
    assert _MORGAN_NBITS == 2048


def test_build_reference_creates_valid_checker(tmp_path):
    """build_reference classmethod saves a pickle that produces a working checker."""
    ref_tuples = [
        _mol_to_ref_tuple("CCO"),
        _mol_to_ref_tuple("c1ccccc1"),
    ]
    pkl_path = tmp_path / "test_ref.pkl"
    ChEMBLNoveltyChecker.build_reference(ref_tuples, pkl_path)
    assert pkl_path.exists()
    checker = ChEMBLNoveltyChecker(pkl_path, reference_db="TestDB")
    assert isinstance(checker, ChEMBLNoveltyChecker)
    result = checker.check(MolFromSmiles("CCCO"))
    assert isinstance(result, NoveltyResult)
    assert result.reference_db == "TestDB"


def test_none_mol_raises(tmp_path):
    """Passing None instead of Mol must raise TypeError with helpful message."""
    checker = _build_checker(["CCO"], tmp_path)
    with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
        checker.check(None)
    assert "NoneType" in str(exc_info.value)
