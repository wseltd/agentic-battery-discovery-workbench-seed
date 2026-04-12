"""Molecular duplicate detection via canonical SMILES, InChIKey, and fingerprint similarity.

Tracks registered molecules and classifies new candidates as unique, exact
duplicates (by canonical SMILES or InChIKey), or near-duplicates (by
Tanimoto similarity of Morgan fingerprints). Near-duplicate threshold is a
named constant rather than a magic number — chose 0.85 based on common
cheminformatics practice for structural analogue detection.

Morgan fingerprints use radius=2, nBits=2048 — the standard parameterisation
in most virtual screening workflows. Radius 3 captures more context but
inflates the bitvector sparsity for small molecules; radius 2 is a better
default for a general-purpose detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rdkit.Chem import Mol, MolToSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from rdkit.DataStructs import BulkTanimotoSimilarity


# --- Constants ---------------------------------------------------------------

NEAR_DUPLICATE_THRESHOLD: float = 0.85

# Morgan fingerprint parameters — centralised here so tests and future
# callers reference the same values.
_MORGAN_RADIUS: int = 2
_MORGAN_NBITS: int = 2048


# --- Data structures ---------------------------------------------------------


class DuplicateStatus(Enum):
    """Classification of a molecule relative to previously registered molecules."""

    UNIQUE = "unique"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@dataclass(frozen=True, slots=True)
class DuplicateResult:
    """Result of a duplicate check.

    Attributes
    ----------
    status:
        Whether the molecule is unique, an exact duplicate, or a near duplicate.
    cluster_id:
        Identifier of the first-seen molecule in the duplicate cluster,
        or None if the molecule is unique.
    similarity:
        Tanimoto similarity to the nearest registered molecule, or None
        if the molecule is an exact duplicate or unique with no near match.
    """

    status: DuplicateStatus
    cluster_id: str | None
    similarity: float | None


# --- Detector ----------------------------------------------------------------


class DuplicateDetector:
    """Registry-based molecular duplicate detector.

    Molecules are registered with an identifier. Subsequent calls to
    ``check`` classify molecules against the registry using a two-stage
    test: exact match first (canonical SMILES or InChIKey), then
    fingerprint similarity.

    Exact checking precedes near-duplicate checking because it is cheaper
    (dict lookup vs. O(n) fingerprint comparison) and produces a
    deterministic answer.
    """

    def __init__(self) -> None:
        # canonical SMILES → molecule id
        self._smiles_index: dict[str, str] = {}
        # InChIKey → molecule id
        self._inchikey_index: dict[str, str] = {}
        # list of (mol_id, fingerprint) for near-duplicate scanning
        self._fingerprints: list[tuple[str, object]] = []

    def register(self, mol: Mol, mol_id: str) -> None:
        """Register a molecule in the detector's index.

        Parameters
        ----------
        mol:
            RDKit Mol object.
        mol_id:
            Unique identifier for this molecule (e.g. ``'mol_001'``).

        Raises
        ------
        TypeError
            If *mol* is not an RDKit Mol.
        """
        if not isinstance(mol, Mol):
            raise TypeError(
                f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
            )

        canon = MolToSmiles(mol)
        # Two-step InChIKey: MolToInchi then InchiToInchiKey
        inchi_str = MolToInchi(mol)
        inchikey = InchiToInchiKey(inchi_str) if inchi_str is not None else None
        fp = GetMorganFingerprintAsBitVect(
            mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS
        )

        self._smiles_index.setdefault(canon, mol_id)
        if inchikey is not None:
            self._inchikey_index.setdefault(inchikey, mol_id)
        self._fingerprints.append((mol_id, fp))

    def check(self, mol: Mol) -> DuplicateResult:
        """Check whether a molecule duplicates any registered molecule.

        Parameters
        ----------
        mol:
            RDKit Mol object to check.

        Returns
        -------
        DuplicateResult
            Classification with optional cluster id and similarity score.

        Raises
        ------
        TypeError
            If *mol* is not an RDKit Mol.
        """
        if not isinstance(mol, Mol):
            raise TypeError(
                f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
            )

        # Stage 1: exact match by canonical SMILES or InChIKey
        canon = MolToSmiles(mol)
        if canon in self._smiles_index:
            return DuplicateResult(
                status=DuplicateStatus.EXACT_DUPLICATE,
                cluster_id=self._smiles_index[canon],
                similarity=1.0,
            )

        inchi_str = MolToInchi(mol)
        inchikey = InchiToInchiKey(inchi_str) if inchi_str is not None else None
        if inchikey is not None and inchikey in self._inchikey_index:
            return DuplicateResult(
                status=DuplicateStatus.EXACT_DUPLICATE,
                cluster_id=self._inchikey_index[inchikey],
                similarity=1.0,
            )

        # Stage 2: near-duplicate by fingerprint similarity
        if not self._fingerprints:
            return DuplicateResult(
                status=DuplicateStatus.UNIQUE,
                cluster_id=None,
                similarity=None,
            )

        fp = GetMorganFingerprintAsBitVect(
            mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS
        )

        # Bulk similarity is O(n) but avoids Python-level loop overhead
        registered_fps = [rfp for _, rfp in self._fingerprints]
        similarities = BulkTanimotoSimilarity(fp, registered_fps)

        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        best_sim = similarities[best_idx]
        best_id = self._fingerprints[best_idx][0]

        if best_sim >= NEAR_DUPLICATE_THRESHOLD:
            return DuplicateResult(
                status=DuplicateStatus.NEAR_DUPLICATE,
                cluster_id=best_id,
                similarity=best_sim,
            )

        return DuplicateResult(
            status=DuplicateStatus.UNIQUE,
            cluster_id=None,
            similarity=None,
        )
