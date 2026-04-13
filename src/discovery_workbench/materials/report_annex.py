"""Materials report annex dataclasses and builder.

Assembles a structured annex section from generator configuration,
validation statistics, novelty results, and DFT handoff paths.  The
annex is a read-only summary intended for downstream report rendering
— it does not own or modify any of the data it references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from discovery_workbench.materials.ranker import RankedCandidate

logger = logging.getLogger(__name__)

__all__ = ["MaterialsAnnexInput", "MaterialsAnnex", "build_materials_annex"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MaterialsAnnexInput:
    """Input bundle for building a materials report annex.

    Args:
        generator_config: MatterGen generator configuration dict.
        scope_config: Scope constraints applied during generation.
        relaxer_version: Version string for MatterSim relaxer.
        ranked_candidates: Scored and ranked candidate list.
        validity_count: Number of candidates passing validity checks.
        uniqueness_count: Number of structurally unique candidates.
        novelty_count: Number of novel candidates vs reference databases.
        total_generated: Total candidates produced by the generator.
        matcher_tolerances: Tolerance parameters used for novelty matching.
        reference_db_ids: Identifiers of reference databases queried.
        dft_handoff_paths: Paths to DFT handoff bundle directories.
    """

    generator_config: dict
    scope_config: dict
    relaxer_version: str
    ranked_candidates: list[RankedCandidate]
    validity_count: int
    uniqueness_count: int
    novelty_count: int
    total_generated: int
    matcher_tolerances: dict
    reference_db_ids: list[str]
    dft_handoff_paths: list[Path]


@dataclass(slots=True)
class MaterialsAnnex:
    """Structured annex section for a materials discovery report.

    Args:
        generator_section: Generator provenance (checkpoint, conditioning, etc.).
        scope_section: Scope constraints (max_atoms, excluded_elements, etc.).
        relaxer_section: Relaxer metadata (version, convergence settings).
        validation_stats: Counts for validity, uniqueness, novelty, total.
        novelty_details: Reference databases, tolerances, novelty fraction.
        dft_handoff_summary: Per-candidate file path summaries.
        warnings: Diagnostic messages for the report consumer.
    """

    generator_section: dict
    scope_section: dict
    relaxer_section: dict
    validation_stats: dict
    novelty_details: dict
    dft_handoff_summary: list[dict]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_materials_annex(annex_input: MaterialsAnnexInput) -> MaterialsAnnex:
    """Build a MaterialsAnnex from validated pipeline outputs.

    Transforms raw configuration dicts and statistics into the
    structured sections expected by report rendering.

    Args:
        annex_input: Populated input bundle.

    Returns:
        MaterialsAnnex ready for report rendering.
    """
    logger.info(
        "Building materials annex for %d candidates",
        len(annex_input.ranked_candidates),
    )

    warnings: list[str] = []

    # --- Generator section ---
    generator_section = dict(annex_input.generator_config)

    # --- Scope section ---
    scope_section = dict(annex_input.scope_config)

    # --- Relaxer section ---
    relaxer_section = {"version": annex_input.relaxer_version}

    # --- Validation stats ---
    total = annex_input.total_generated
    validation_stats = {
        "validity_count": annex_input.validity_count,
        "uniqueness_count": annex_input.uniqueness_count,
        "novelty_count": annex_input.novelty_count,
        "total_generated": total,
    }

    # --- Novelty details ---
    novelty_fraction = (
        annex_input.novelty_count / total if total > 0 else 0.0
    )
    novelty_details = {
        "reference_dbs": list(annex_input.reference_db_ids),
        "matcher_tolerances": dict(annex_input.matcher_tolerances),
        "novelty_fraction": novelty_fraction,
    }

    # --- DFT handoff summary ---
    dft_handoff_summary: list[dict] = []
    for candidate in annex_input.ranked_candidates:
        matching_paths = [
            p for p in annex_input.dft_handoff_paths
            if candidate.candidate_id in str(p)
        ]
        entry: dict = {"candidate_id": candidate.candidate_id}
        if matching_paths:
            entry["paths"] = [str(p) for p in matching_paths]
        dft_handoff_summary.append(entry)

    if total == 0:
        warnings.append(
            "Zero structures generated — verify generator configuration"
        )

    if total > 0 and annex_input.validity_count == 0:
        warnings.append(
            f"No candidates passed validity checks out of "
            f"{total} generated"
        )

    if annex_input.novelty_count == 0 and annex_input.uniqueness_count > 0:
        warnings.append(
            "All unique candidates matched existing database entries — "
            "zero novel structures found"
        )

    if annex_input.ranked_candidates and not annex_input.dft_handoff_paths:
        warnings.append(
            "No DFT handoff paths provided for "
            f"{len(annex_input.ranked_candidates)} ranked candidates"
        )

    return MaterialsAnnex(
        generator_section=generator_section,
        scope_section=scope_section,
        relaxer_section=relaxer_section,
        validation_stats=validation_stats,
        novelty_details=novelty_details,
        dft_handoff_summary=dft_handoff_summary,
        warnings=warnings,
    )
