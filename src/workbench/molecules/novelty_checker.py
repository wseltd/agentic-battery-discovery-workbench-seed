"""ChEMBL-based novelty classification for small molecules.

Classifies molecules against a reference set of known compounds using
InChIKey exact matching and Tanimoto similarity on Morgan fingerprints.

Three classification tiers:
- EXACT_KNOWN: InChIKey matches a reference compound (Tanimoto irrelevant).
- CLOSE_ANALOGUE: max Tanimoto >= 0.70 (structural neighbour of a known compound).
- NOVEL_LIKE: max Tanimoto < 0.40 (structurally distant from all references).

The gap region (0.40 <= Tanimoto < 0.70) falls into CLOSE_ANALOGUE — a
conservative choice that avoids over-claiming novelty for moderately similar
structures. An alternative would be a fourth tier, but YAGNI applies here.

Morgan fingerprints use radius=2, nBits=2048 — same parameterisation as the
DuplicateDetector (T019) so fingerprint comparisons are consistent across
the pipeline.

Evidence level is always HEURISTIC_ESTIMATED because Tanimoto similarity on
2D fingerprints is a structural heuristic, not a computed or measured property.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from rdkit.DataStructs import BulkTanimotoSimilarity

from workbench.shared.evidence import EvidenceLevel


# --- Constants ---------------------------------------------------------------

# Boundary at exactly 0.70 → CLOSE_ANALOGUE (inclusive).
CLOSE_ANALOGUE_THRESHOLD: float = 0.70

# Boundary at exactly 0.40 → CLOSE_ANALOGUE (inclusive); below → NOVEL_LIKE.
NOVEL_LIKE_THRESHOLD: float = 0.40

# Morgan fingerprint parameters — must match T019 DuplicateDetector for
# consistency across the pipeline.
_MORGAN_RADIUS: int = 2
_MORGAN_NBITS: int = 2048


# --- Data structures ---------------------------------------------------------


class NoveltyClass(Enum):
    """Three-tier novelty classification for a query molecule."""

    EXACT_KNOWN = "exact_known"
    CLOSE_ANALOGUE = "close_analogue"
    NOVEL_LIKE = "novel_like"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    """Result of a novelty check against a reference database.

    Attributes
    ----------
    classification:
        Novelty tier for this molecule.
    max_tanimoto:
        Highest Tanimoto similarity to any reference molecule.
        0.0 when the reference set is empty.
    closest_inchikey:
        InChIKey of the most similar reference molecule, or None
        when the reference set is empty.
    reference_db:
        Name of the reference database used for comparison.
    evidence_level:
        Always HEURISTIC_ESTIMATED — Tanimoto on 2D fingerprints
        is a structural heuristic.
    """

    classification: NoveltyClass
    max_tanimoto: float
    closest_inchikey: str | None
    reference_db: str
    evidence_level: EvidenceLevel


# --- Checker -----------------------------------------------------------------


class ChEMBLNoveltyChecker:
    """Novelty checker against a reference set of known compounds.

    Two-stage classification: exact InChIKey match first (dict lookup),
    then Tanimoto fingerprint similarity. Exact match takes precedence
    because it is cheaper and deterministic — same rationale as
    DuplicateDetector.

    Parameters
    ----------
    reference_db:
        Name tag for the reference database (default ``'ChEMBL_36'``).
    """

    def __init__(self, reference_db: str = "ChEMBL_36") -> None:
        self._reference_db = reference_db
        # InChIKey → InChIKey (identity map for fast lookup)
        self._inchikey_index: dict[str, str] = {}
        # list of (inchikey, fingerprint) for similarity scanning
        self._fingerprints: list[tuple[str, object]] = []

    @classmethod
    def build_reference(
        cls,
        mols: list[Mol],
        reference_db: str = "ChEMBL_36",
    ) -> ChEMBLNoveltyChecker:
        """Build a checker from a list of reference molecules.

        Parameters
        ----------
        mols:
            RDKit Mol objects representing known compounds.
        reference_db:
            Name tag for the reference database.

        Returns
        -------
        ChEMBLNoveltyChecker
            Ready-to-use checker with the reference set indexed.

        Raises
        ------
        TypeError
            If any element in *mols* is not an RDKit Mol.
        """
        checker = cls(reference_db=reference_db)
        for mol in mols:
            if not isinstance(mol, Mol):
                raise TypeError(
                    f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
                )
            inchi_str = MolToInchi(mol)
            inchikey = InchiToInchiKey(inchi_str) if inchi_str is not None else None

            fp = GetMorganFingerprintAsBitVect(
                mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS
            )

            if inchikey is not None:
                self = checker
                self._inchikey_index.setdefault(inchikey, inchikey)
                self._fingerprints.append((inchikey, fp))
        return checker

    def check(self, mol: Mol) -> NoveltyResult:
        """Classify a molecule's novelty against the reference set.

        Parameters
        ----------
        mol:
            RDKit Mol object to classify.

        Returns
        -------
        NoveltyResult
            Classification with similarity details and evidence level.

        Raises
        ------
        TypeError
            If *mol* is not an RDKit Mol.
        """
        if not isinstance(mol, Mol):
            raise TypeError(
                f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
            )

        # Stage 1: exact InChIKey match
        inchi_str = MolToInchi(mol)
        query_inchikey = (
            InchiToInchiKey(inchi_str) if inchi_str is not None else None
        )

        if query_inchikey is not None and query_inchikey in self._inchikey_index:
            return NoveltyResult(
                classification=NoveltyClass.EXACT_KNOWN,
                max_tanimoto=1.0,
                closest_inchikey=query_inchikey,
                reference_db=self._reference_db,
                evidence_level=EvidenceLevel.HEURISTIC_ESTIMATED,
            )

        # Stage 2: fingerprint similarity
        if not self._fingerprints:
            return NoveltyResult(
                classification=NoveltyClass.NOVEL_LIKE,
                max_tanimoto=0.0,
                closest_inchikey=None,
                reference_db=self._reference_db,
                evidence_level=EvidenceLevel.HEURISTIC_ESTIMATED,
            )

        fp = GetMorganFingerprintAsBitVect(
            mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS
        )

        ref_fps = [rfp for _, rfp in self._fingerprints]
        similarities = BulkTanimotoSimilarity(fp, ref_fps)

        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        best_sim = similarities[best_idx]
        best_inchikey = self._fingerprints[best_idx][0]

        classification = _classify_tanimoto(best_sim)

        return NoveltyResult(
            classification=classification,
            max_tanimoto=best_sim,
            closest_inchikey=best_inchikey,
            reference_db=self._reference_db,
            evidence_level=EvidenceLevel.HEURISTIC_ESTIMATED,
        )


def _classify_tanimoto(tanimoto: float) -> NoveltyClass:
    """Map a Tanimoto similarity score to a NoveltyClass.

    >= 0.70 → CLOSE_ANALOGUE
    >= 0.40 → CLOSE_ANALOGUE  (gap region — conservative, avoids over-claiming novelty)
    <  0.40 → NOVEL_LIKE
    """
    if tanimoto >= CLOSE_ANALOGUE_THRESHOLD:
        return NoveltyClass.CLOSE_ANALOGUE
    if tanimoto >= NOVEL_LIKE_THRESHOLD:
        return NoveltyClass.CLOSE_ANALOGUE
    return NoveltyClass.NOVEL_LIKE
