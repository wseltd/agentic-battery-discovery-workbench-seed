"""Frozen dataclasses for molecule-generation report annexes.

These stats containers summarise validity, uniqueness, novelty,
constraint satisfaction, and export paths for a molecule-generation run.
All are frozen to prevent accidental mutation after construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ConstraintResult",
    "ExportPaths",
    "HEURISTIC_WARNING_TEMPLATES",
    "MoleculeReportAnnex",
    "NoveltyStats",
    "UniquenessStats",
    "ValidityStats",
    "build_molecule_annex",
    "format_heuristic_warning",
]

# Q31-approved heuristic warning templates.
HEURISTIC_WARNING_TEMPLATES: dict[str, str] = {
    "crippen_logp": (
        "Estimated by RDKit Crippen; heuristic prediction, may be inaccurate."
    ),
    "tpsa_approx": (
        "TPSA computed via fragment-based approximation; "
        "real value may differ for strained rings."
    ),
    "similarity_cutoff": (
        "Novelty uses Tanimoto threshold {threshold}; "
        "near-boundary molecules may be misclassified."
    ),
    "xtb_semiempirical": (
        "xTB energies are semi-empirical (GFN{level}); "
        "treat as ranking heuristic, not absolute values."
    ),
    "salt_strip_heuristic": (
        "Salt stripping uses a curated fragment list; "
        "unusual salts may not be removed."
    ),
    "stereo_perception": (
        "Stereocentre detection is RDKit-based; "
        "complex macrocyclic stereo may be missed."
    ),
}


def format_heuristic_warning(template_key: str, **kwargs: object) -> str:
    """Select an approved heuristic-warning template and format it.

    Parameters
    ----------
    template_key:
        Key into ``HEURISTIC_WARNING_TEMPLATES``.
    **kwargs:
        Values substituted into the template via ``str.format_map``.

    Returns
    -------
    str
        The formatted warning string.

    Raises
    ------
    ValueError
        If *template_key* is not a recognised template.
    """
    template = HEURISTIC_WARNING_TEMPLATES.get(template_key)
    if template is None:
        raise ValueError(
            f"Unknown heuristic-warning template key {template_key!r}. "
            f"Valid keys: {sorted(HEURISTIC_WARNING_TEMPLATES)}"
        )
    return template.format_map(kwargs)


@dataclass(frozen=True)
class ValidityStats:
    """Counts from the validity-checking pipeline stage."""

    total_generated: int
    syntax_valid: int
    valence_valid: int
    charge_valid: int
    stereo_flagged: int
    salt_stripped: int
    final_valid: int


@dataclass(frozen=True)
class UniquenessStats:
    """Counts from the deduplication pipeline stage."""

    total_valid: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    unique_count: int


@dataclass(frozen=True)
class NoveltyStats:
    """Counts from the novelty-checking pipeline stage."""

    reference_db: str
    reference_version: str
    exact_known_count: int
    close_analogue_count: int
    novel_like_count: int
    similarity_threshold: float


@dataclass(frozen=True)
class ConstraintResult:
    """Result of checking one design constraint against the candidate set."""

    constraint_name: str
    window_min: float | None
    window_max: float | None
    satisfied_count: int
    violated_count: int
    satisfaction_rate: float


@dataclass(frozen=True)
class ExportPaths:
    """Filesystem paths where generated molecule artefacts were written."""

    smiles_file: str
    inchikey_file: str
    sdf_dir: str
    xyz_dir: str
    xtb_handoff_dir: str


@dataclass(frozen=True)
class MoleculeReportAnnex:
    """Top-level annex aggregating all molecule-generation report sections."""

    generator_config: dict[str, Any]
    validity_stats: ValidityStats
    uniqueness_stats: UniquenessStats
    novelty_stats: NoveltyStats
    constraint_breakdown: list[ConstraintResult]
    export_paths: ExportPaths
    heuristic_warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the entire annex to a plain dict (JSON-safe)."""
        return asdict(self)


def build_molecule_annex(
    *,
    generator_config: dict[str, Any],
    validity_stats: ValidityStats,
    uniqueness_stats: UniquenessStats,
    novelty_stats: NoveltyStats,
    constraint_breakdown: list[ConstraintResult],
    export_paths: ExportPaths,
    heuristic_warnings: list[str] | None = None,
) -> MoleculeReportAnnex:
    """Construct a MoleculeReportAnnex from its component sections.

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    """
    if not isinstance(generator_config, dict):
        raise TypeError(
            f"generator_config must be a dict, got {type(generator_config).__name__}"
        )
    if not isinstance(validity_stats, ValidityStats):
        raise TypeError(
            f"validity_stats must be a ValidityStats, got {type(validity_stats).__name__}"
        )
    if not isinstance(uniqueness_stats, UniquenessStats):
        raise TypeError(
            f"uniqueness_stats must be an UniquenessStats, got {type(uniqueness_stats).__name__}"
        )
    if not isinstance(novelty_stats, NoveltyStats):
        raise TypeError(
            f"novelty_stats must be a NoveltyStats, got {type(novelty_stats).__name__}"
        )
    if not isinstance(export_paths, ExportPaths):
        raise TypeError(
            f"export_paths must be an ExportPaths, got {type(export_paths).__name__}"
        )
    for i, cr in enumerate(constraint_breakdown):
        if not isinstance(cr, ConstraintResult):
            raise TypeError(
                f"constraint_breakdown[{i}] must be a ConstraintResult, "
                f"got {type(cr).__name__}"
            )
    warnings = heuristic_warnings if heuristic_warnings is not None else []
    for i, w in enumerate(warnings):
        if not isinstance(w, str):
            raise TypeError(
                f"heuristic_warnings[{i}] must be a str, got {type(w).__name__}"
            )
    return MoleculeReportAnnex(
        generator_config=generator_config,
        validity_stats=validity_stats,
        uniqueness_stats=uniqueness_stats,
        novelty_stats=novelty_stats,
        constraint_breakdown=constraint_breakdown,
        export_paths=export_paths,
        heuristic_warnings=warnings,
    )
