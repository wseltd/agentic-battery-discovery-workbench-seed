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
    "MoleculeReportAnnex",
    "NoveltyStats",
    "UniquenessStats",
    "ValidityStats",
    "build_molecule_annex",
]


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

    Parameters
    ----------
    generator_config:
        Opaque dict capturing the generator settings used for this run.
    validity_stats:
        Aggregated counts from the validity pipeline stage.
    uniqueness_stats:
        Aggregated counts from the deduplication stage.
    novelty_stats:
        Aggregated counts from the novelty-check stage.
    constraint_breakdown:
        Per-constraint satisfaction results.
    export_paths:
        Filesystem locations of exported artefacts.
    heuristic_warnings:
        Free-text warnings about heuristic or approximate computations.

    Returns
    -------
    MoleculeReportAnnex
    """
    return MoleculeReportAnnex(
        generator_config=generator_config,
        validity_stats=validity_stats,
        uniqueness_stats=uniqueness_stats,
        novelty_stats=novelty_stats,
        constraint_breakdown=constraint_breakdown,
        export_paths=export_paths,
        heuristic_warnings=heuristic_warnings if heuristic_warnings is not None else [],
    )
