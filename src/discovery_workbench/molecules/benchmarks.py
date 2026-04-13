"""Molecular benchmark metrics for evaluating generated candidate libraries.

Computes validity, uniqueness, novelty, target satisfaction, internal
diversity, and shortlist quality metrics following the standard
generative-chemistry benchmark framework (MOSES/GuacaMol conventions).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import statistics
import types
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, QED, DataStructs, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from rdkit.ML.Cluster import Butina

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Morgan fingerprint parameters for diversity and clustering.
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Butina clustering distance threshold (Tanimoto).  0.4 is a common
# default in cheminformatics — tight enough to separate scaffolds,
# loose enough to avoid singleton-dominated clusters.
BUTINA_DISTANCE_CUTOFF = 0.4

# ---------------------------------------------------------------------------
# Expensive module-level resources — built once, reused across calls
# ---------------------------------------------------------------------------

_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS,
)

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(_pains_params)


def _load_sa_scorer() -> types.ModuleType:
    """Load the Ertl/Schuffenhauer SA score calculator from RDKit Contrib."""
    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
    spec = importlib.util.spec_from_file_location("sascorer", sa_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load SA score module from {sa_path}. "
            "Ensure rdkit is installed with Contrib files."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sa_module = _load_sa_scorer()

# Property calculators for target satisfaction.
# Maps property name to a callable (Mol -> float).  Typed as Any
# because rdkit descriptor functions lack consistent type stubs.
_PROPERTY_CALCULATORS: dict[str, Any] = {
    "qed": QED.qed,
    "mw": Descriptors.MolWt,  # type: ignore[attr-defined]
    "logp": Descriptors.MolLogP,  # type: ignore[attr-defined]
    "tpsa": Descriptors.TPSA,  # type: ignore[attr-defined]
    "hbd": Descriptors.NumHDonors,  # type: ignore[attr-defined]
    "hba": Descriptors.NumHAcceptors,  # type: ignore[attr-defined]
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MolecularBenchmarkResult:
    """Complete benchmark metrics for a generated molecular library.

    Args:
        total_generated: Number of SMILES strings in the input.
        valid_count: Number that parsed as valid molecules.
        validity_pct: valid_count / total_generated * 100.
        unique_count: Number of unique canonical SMILES among valid.
        uniqueness_pct: unique_count / valid_count * 100.
        novel_count: Number of unique SMILES not in the known set.
        novelty_pct: novel_count / unique_count * 100.
        target_satisfaction_fraction: Fraction of novel molecules meeting
            all property targets (0.0–1.0).
        diversity_mean: Mean pairwise Tanimoto diversity among novel molecules.
        diversity_std: Population std of pairwise Tanimoto diversity.
        shortlist_pains_pass_pct: Percentage of novel molecules passing PAINS.
        shortlist_median_qed: Median QED score of novel molecules.
        shortlist_median_sa: Median SA score of novel molecules (1–10).
        shortlist_cluster_count: Number of Butina clusters among novel molecules.
    """

    total_generated: int
    valid_count: int
    validity_pct: float
    unique_count: int
    uniqueness_pct: float
    novel_count: int
    novelty_pct: float
    target_satisfaction_fraction: float
    diversity_mean: float
    diversity_std: float
    shortlist_pains_pass_pct: float
    shortlist_median_qed: float
    shortlist_median_sa: float
    shortlist_cluster_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """Compute Morgan fingerprint using the module-level generator."""
    return _morgan_generator.GetFingerprint(mol)


def _sa_score(mol: Chem.Mol) -> float:
    """Compute Ertl/Schuffenhauer synthetic accessibility score (1–10)."""
    return _sa_module.calculateScore(mol)


def _mol_to_inchikey(mol: Chem.Mol) -> str | None:
    """Convert an RDKit Mol to its InChIKey, or None on failure."""
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    return InchiToInchiKey(inchi)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_validity(
    smiles: list[str],
) -> tuple[int, int, float, list[str]]:
    """Determine which SMILES parse as valid molecules.

    Uses RDKit MolFromSmiles parse success as the validity criterion.
    Valid SMILES are canonicalised before return.

    Args:
        smiles: Raw SMILES strings to validate.

    Returns:
        Tuple of (total_generated, valid_count, validity_pct,
        valid_canonical_smiles).  validity_pct is 0–100.
    """
    logger.info("compute_validity called with %d SMILES", len(smiles))
    total = len(smiles)
    valid_canonical: list[str] = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_canonical.append(Chem.MolToSmiles(mol, canonical=True))

    valid_count = len(valid_canonical)
    pct = (valid_count / total * 100.0) if total > 0 else 0.0

    logger.info(
        "Validity: %d/%d (%.1f%%)", valid_count, total, pct,
    )
    return total, valid_count, pct, valid_canonical


def compute_uniqueness(
    valid_canonical_smiles: list[str],
) -> tuple[int, float, list[str]]:
    """Deduplicate canonical SMILES and compute uniqueness rate.

    Args:
        valid_canonical_smiles: Canonical SMILES from compute_validity.

    Returns:
        Tuple of (unique_count, uniqueness_pct, unique_smiles).
        uniqueness_pct is unique_count / len(input) * 100, or 0.0
        if input is empty.
    """
    logger.info(
        "compute_uniqueness called with %d SMILES",
        len(valid_canonical_smiles),
    )
    seen: set[str] = set()
    unique: list[str] = []
    for smi in valid_canonical_smiles:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)

    total = len(valid_canonical_smiles)
    pct = (len(unique) / total * 100.0) if total > 0 else 0.0

    return len(unique), pct, unique


def compute_novelty(
    unique_smiles: list[str],
    reference_inchikeys: set[str],
) -> tuple[int, float, list[str]]:
    """Identify which unique SMILES are absent from a known reference set.

    Converts each SMILES to an InChIKey via RDKit and checks membership
    in reference_inchikeys.  InChIKey comparison is preferred over raw
    SMILES because it is canonicalisation-independent and matches the
    standard cheminformatics de-duplication approach.

    Args:
        unique_smiles: Deduplicated canonical SMILES.
        reference_inchikeys: Set of known InChIKeys to compare against.

    Returns:
        Tuple of (novel_count, novelty_pct, novel_smiles).
        novelty_pct is novel_count / len(unique_smiles) * 100,
        or 0.0 if input is empty.  Molecules that fail InChIKey
        conversion are treated as novel (benefit of the doubt).
    """
    logger.info(
        "compute_novelty called with %d unique, %d known",
        len(unique_smiles), len(reference_inchikeys),
    )
    novel: list[str] = []
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Unparseable — should not happen after validity, but be safe
            novel.append(smi)
            continue
        ik = _mol_to_inchikey(mol)
        if ik is None or ik not in reference_inchikeys:
            novel.append(smi)

    total = len(unique_smiles)
    pct = (len(novel) / total * 100.0) if total > 0 else 0.0

    return len(novel), pct, novel


def compute_target_satisfaction(
    smiles: list[str],
    constraints: dict[str, tuple[float | None, float | None]],
) -> float:
    """Compute fraction of molecules satisfying all property constraint windows.

    Each constraint maps a property name to a ``(min, max)`` window.
    A molecule satisfies a constraint when ``min <= value <= max``.
    Either bound can be ``None`` for an open-ended window (e.g.
    ``(200.0, None)`` means MW >= 200 with no upper limit).

    Supported property names: qed, mw, logp, tpsa, hbd, hba.

    Args:
        smiles: Canonical SMILES to evaluate.
        constraints: Property name to ``(min, max)`` window.  Empty
            dict means no constraints (returns 1.0 unless smiles is
            empty).

    Returns:
        Fraction in [0.0, 1.0].

    Raises:
        ValueError: If a constraint property name is not supported.
    """
    logger.info(
        "compute_target_satisfaction: %d molecules, %d constraints",
        len(smiles), len(constraints),
    )
    if not constraints:
        return 1.0
    if not smiles:
        return 0.0

    for prop in constraints:
        if prop not in _PROPERTY_CALCULATORS:
            raise ValueError(
                f"Unknown property {prop!r}, "
                f"supported: {sorted(_PROPERTY_CALCULATORS)}"
            )

    satisfied = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        all_met = True
        for prop, (lo, hi) in constraints.items():
            value = _PROPERTY_CALCULATORS[prop](mol)
            if lo is not None and value < lo:
                all_met = False
                break
            if hi is not None and value > hi:
                all_met = False
                break
        if all_met:
            satisfied += 1

    return satisfied / len(smiles)


def compute_internal_diversity(
    smiles: list[str],
) -> tuple[float, float]:
    """Compute mean and std of pairwise Tanimoto diversity.

    Diversity for a pair is 1 − Tanimoto(fp_i, fp_j).  With fewer
    than two molecules there are no pairs, so (0.0, 0.0) is returned.

    Args:
        smiles: Canonical SMILES to compare.

    Returns:
        Tuple of (diversity_mean, diversity_std).  Std is population
        standard deviation.
    """
    logger.info("compute_internal_diversity: %d molecules", len(smiles))
    if len(smiles) <= 1:
        return 0.0, 0.0

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [_fingerprint(m) for m in mols if m is not None]

    if len(fps) <= 1:
        return 0.0, 0.0

    diversities: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        diversities.extend(1.0 - s for s in sims)

    if not diversities:
        return 0.0, 0.0

    mean_d = statistics.mean(diversities)
    std_d = statistics.pstdev(diversities)
    return mean_d, std_d


def compute_shortlist_quality(
    smiles: list[str],
) -> tuple[float, float, float, int]:
    """Compute shortlist quality metrics: PAINS pass rate, median QED, median SA, cluster count.

    Args:
        smiles: Canonical SMILES of shortlisted (novel) molecules.

    Returns:
        Tuple of (pains_pass_pct, median_qed, median_sa, cluster_count).
        pains_pass_pct is 0–100.  SA score is 1–10 (lower = easier).
        If smiles is empty, returns (0.0, 0.0, 0.0, 0).
    """
    logger.info("compute_shortlist_quality: %d molecules", len(smiles))
    if not smiles:
        return 0.0, 0.0, 0.0, 0

    mols: list[Chem.Mol] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)

    if not mols:
        return 0.0, 0.0, 0.0, 0

    # PAINS pass rate
    pains_pass = sum(1 for m in mols if not PAINS_CATALOG.HasMatch(m))
    pains_pct = pains_pass / len(mols) * 100.0

    # QED scores
    qed_scores = [QED.qed(m) for m in mols]
    median_qed = statistics.median(qed_scores)

    # SA scores
    sa_scores = [_sa_score(m) for m in mols]
    median_sa = statistics.median(sa_scores)

    # Butina clustering
    fps = [_fingerprint(m) for m in mols]
    if len(fps) <= 1:
        cluster_count = len(fps)
    else:
        # Compute condensed distance matrix (lower triangle)
        dists: list[float] = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)
        clusters = Butina.ClusterData(
            dists, len(fps), BUTINA_DISTANCE_CUTOFF, isDistData=True,
        )
        cluster_count = len(clusters)

    return pains_pct, median_qed, median_sa, cluster_count


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_molecular_benchmarks(
    smiles: list[str],
    reference_inchikeys: set[str],
    constraints: dict[str, tuple[float | None, float | None]],
) -> MolecularBenchmarkResult:
    """Run the full molecular benchmark pipeline.

    Sequentially computes validity, uniqueness, novelty, target
    satisfaction, internal diversity, and shortlist quality.

    Args:
        smiles: Raw generated SMILES strings.
        reference_inchikeys: Set of known InChIKeys (for novelty).
        constraints: Property name to ``(min, max)`` window (for
            target satisfaction).  Empty dict means no constraints.

    Returns:
        MolecularBenchmarkResult with all metrics populated.
    """
    logger.info("compute_molecular_benchmarks: %d input SMILES", len(smiles))

    total, valid_count, validity_pct, valid_canonical = compute_validity(
        smiles,
    )
    unique_count, uniqueness_pct, unique_smi = compute_uniqueness(
        valid_canonical,
    )
    novel_count, novelty_pct, novel_smi = compute_novelty(
        unique_smi, reference_inchikeys,
    )
    target_sat = compute_target_satisfaction(novel_smi, constraints)
    div_mean, div_std = compute_internal_diversity(novel_smi)
    pains_pct, med_qed, med_sa, clusters = compute_shortlist_quality(
        novel_smi,
    )

    return MolecularBenchmarkResult(
        total_generated=total,
        valid_count=valid_count,
        validity_pct=validity_pct,
        unique_count=unique_count,
        uniqueness_pct=uniqueness_pct,
        novel_count=novel_count,
        novelty_pct=novelty_pct,
        target_satisfaction_fraction=target_sat,
        diversity_mean=div_mean,
        diversity_std=div_std,
        shortlist_pains_pass_pct=pains_pct,
        shortlist_median_qed=med_qed,
        shortlist_median_sa=med_sa,
        shortlist_cluster_count=clusters,
    )
