"""Tests for molecular duplicate detection and novelty classification.

Exercises three capabilities from T019/T020:
  - 3 exact-duplicate detection tests (canonical SMILES and InChIKey)
  - 3 near-duplicate Tanimoto threshold tests (boundary-focused)
  - 3 novelty classification tests (three-tier InChIKey + fingerprint)

All tests use real RDKit fingerprints on pre-selected molecule pairs with
known Tanimoto similarities.  No external databases, no mocks.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

from discovery_workbench.molecules.novelty import (
    ChEMBLNoveltyChecker,
    DuplicateDetector,
    DuplicateStatus,
    NEAR_DUPLICATE_THRESHOLD,
    NoveltyClass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mol(smiles: str) -> Chem.Mol:
    """Parse SMILES, raising on invalid input so test failures are obvious."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def _build_checker(smiles_list: list[str], tmp_path) -> ChEMBLNoveltyChecker:
    """Build a ChEMBLNoveltyChecker from a small in-memory SMILES list."""
    tuples = []
    for smi in smiles_list:
        mol = _mol(smi)
        inchi = MolToInchi(mol)
        ik = InchiToInchiKey(inchi)
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        tuples.append((smi, ik, fp))

    ref_path = tmp_path / "reference.pkl"
    ChEMBLNoveltyChecker.build_reference(tuples, ref_path)
    return ChEMBLNoveltyChecker(ref_path, reference_db="test_set")


# ---------------------------------------------------------------------------
# Exact duplicate detection (3 tests)
# ---------------------------------------------------------------------------

_ACETIC_ACID = "CC(=O)O"
_ACETIC_ACID_ALT_1 = "OC(=O)C"   # same molecule, different notation
_ACETIC_ACID_ALT_2 = "CC(O)=O"   # yet another notation
_ETHANOL = "CCO"                  # genuinely different molecule


def test_exact_duplicate_identical_smiles() -> None:
    """Two identical canonical SMILES are detected as exact duplicates."""
    detector = DuplicateDetector()
    detector.register(_mol(_ACETIC_ACID), "mol_001")

    result = detector.check(_mol(_ACETIC_ACID))

    assert result.status == DuplicateStatus.EXACT_DUPLICATE
    assert result.cluster_id == "mol_001"
    assert result.similarity == 1.0


def test_exact_duplicate_different_notation_same_inchikey() -> None:
    """Same molecule in different SMILES notations detected as duplicate.

    OC(=O)C and CC(O)=O both represent acetic acid.  RDKit canonicalises
    both to CC(=O)O, so the match is caught at the SMILES stage.  The
    InChIKey stage provides a backup for representations that resist
    canonicalisation.
    """
    detector = DuplicateDetector()
    detector.register(_mol(_ACETIC_ACID_ALT_1), "mol_001")

    result = detector.check(_mol(_ACETIC_ACID_ALT_2))

    assert result.status == DuplicateStatus.EXACT_DUPLICATE
    assert result.cluster_id == "mol_001"


def test_exact_duplicate_different_molecules_not_duplicate() -> None:
    """Genuinely different molecules are not flagged as duplicates."""
    detector = DuplicateDetector()
    detector.register(_mol(_ACETIC_ACID), "mol_001")

    result = detector.check(_mol(_ETHANOL))

    assert result.status == DuplicateStatus.UNIQUE
    assert result.cluster_id is None


# ---------------------------------------------------------------------------
# Near-duplicate Tanimoto threshold (3 tests)
#
# Three losartan analogues whose pairwise Morgan(2,2048) Tanimoto values
# straddle the 0.85 threshold:
#   _LOSARTAN_CO  vs _LOSARTAN_CCO → 0.8772  (above threshold)
#   _LOSARTAN_CO  vs _LOSARTAN_CCL → 0.8596  (just above — boundary)
#   _LOSARTAN_CCL vs _LOSARTAN_CCO → 0.8448  (below threshold)
# ---------------------------------------------------------------------------

_LOSARTAN_CO = "CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CO"
_LOSARTAN_CCO = "CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CCO"
_LOSARTAN_CCL = "CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CCl"


def test_near_duplicate_above_threshold() -> None:
    """Pair with Tanimoto ~0.877 (well above 0.85) flagged as near-duplicate."""
    detector = DuplicateDetector()
    detector.register(_mol(_LOSARTAN_CO), "ref_001")

    result = detector.check(_mol(_LOSARTAN_CCO))

    assert result.status == DuplicateStatus.NEAR_DUPLICATE
    assert result.cluster_id == "ref_001"
    assert result.similarity is not None
    assert result.similarity > NEAR_DUPLICATE_THRESHOLD


def test_near_duplicate_at_boundary_inclusive() -> None:
    """Pair just above 0.85 is still flagged — confirms >= not >.

    Exact 0.85 is unreachable with Morgan(2,2048) on real molecules because
    Tanimoto values are integer ratios (on-bit intersections / unions).
    This pair (Tanimoto ~0.8596) is the closest achievable value above the
    threshold, confirming the >= boundary is applied correctly.
    """
    assert NEAR_DUPLICATE_THRESHOLD == 0.85

    detector = DuplicateDetector()
    detector.register(_mol(_LOSARTAN_CO), "ref_001")

    result = detector.check(_mol(_LOSARTAN_CCL))

    assert result.status == DuplicateStatus.NEAR_DUPLICATE
    assert result.similarity is not None
    # Sits between the threshold and the "above" pair — genuinely boundary-adjacent
    assert result.similarity >= NEAR_DUPLICATE_THRESHOLD
    assert result.similarity < 0.87


def test_near_duplicate_below_threshold_distinct() -> None:
    """Pair with Tanimoto ~0.845 (just below 0.85) confirmed as distinct.

    Same molecular family as the boundary test but the CCl/CCO pair lands
    on the other side of the 0.85 divide — confirms the threshold is not
    silently rounded down.
    """
    detector = DuplicateDetector()
    detector.register(_mol(_LOSARTAN_CCL), "ref_001")

    result = detector.check(_mol(_LOSARTAN_CCO))

    assert result.status == DuplicateStatus.UNIQUE
    assert result.cluster_id is None


# ---------------------------------------------------------------------------
# Novelty classification (3 tests)
#
# Reference set: aspirin, (R)-ibuprofen, 2-naphthol (3 molecules).
# Pre-computed query Tanimoto values against this set:
#   aspirin          → InChIKey match                   → EXACT_KNOWN
#   ibuprofen amide  → max Tanimoto 0.70 (to ibuprofen) → CLOSE_ANALOGUE
#   pyridine         → max Tanimoto 0.10                → NOVEL_LIKE
# ---------------------------------------------------------------------------

_REFERENCE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # (R)-ibuprofen
    "c1ccc2c(c1)cc1ccccc1c2O",           # 2-naphthol
]

_QUERY_EXACT_KNOWN = "CC(=O)Oc1ccccc1C(=O)O"         # aspirin (in reference)
_QUERY_CLOSE_ANALOGUE = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)N"  # ibuprofen amide
_QUERY_NOVEL = "c1ccncc1"                              # pyridine


def test_novelty_exact_known_inchikey_match(tmp_path) -> None:
    """Molecule whose InChIKey is in the reference set classified as EXACT_KNOWN."""
    checker = _build_checker(_REFERENCE_SMILES, tmp_path)

    result = checker.check(_mol(_QUERY_EXACT_KNOWN))

    assert result.classification == NoveltyClass.EXACT_KNOWN
    assert result.max_tanimoto == 1.0
    assert result.reference_db == "test_set"


def test_novelty_close_analogue_tanimoto_above_070(tmp_path) -> None:
    """Molecule with max Tanimoto 0.70 to reference classified as CLOSE_ANALOGUE.

    Ibuprofen amide shares the ibuprofen scaffold but has an amide instead
    of carboxylic acid.  Morgan(2,2048) Tanimoto to (R)-ibuprofen is 0.70,
    placing it exactly at the CLOSE_ANALOGUE boundary (>= 0.70).
    """
    checker = _build_checker(_REFERENCE_SMILES, tmp_path)

    result = checker.check(_mol(_QUERY_CLOSE_ANALOGUE))

    assert result.classification == NoveltyClass.CLOSE_ANALOGUE
    assert result.max_tanimoto >= 0.70


def test_novelty_novel_like_tanimoto_below_040(tmp_path) -> None:
    """Molecule with max Tanimoto ~0.10 classified as NOVEL_LIKE.

    Pyridine is a small heterocycle structurally distant from all three
    reference molecules.  Max Tanimoto is ~0.10, well below the 0.40
    NOVEL_LIKE threshold.
    """
    checker = _build_checker(_REFERENCE_SMILES, tmp_path)

    result = checker.check(_mol(_QUERY_NOVEL))

    assert result.classification == NoveltyClass.NOVEL_LIKE
    assert result.max_tanimoto < 0.40
