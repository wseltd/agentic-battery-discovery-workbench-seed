"""Materials report annex dataclasses and builder.

Assembles a structured annex section from generator configuration,
validation statistics, novelty results, and DFT handoff paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from agentic_materials_discovery.ranking.ranker import RankedCandidate

logger = logging.getLogger(__name__)

__all__ = ["MaterialsAnnexInput", "MaterialsAnnex", "build_materials_annex"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MaterialsAnnexInput:
    """Input bundle for building a materials report annex."""

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
    """Structured annex section for a materials discovery report."""

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

    generator_section = dict(annex_input.generator_config)
    scope_section = dict(annex_input.scope_config)
    relaxer_section = {"version": annex_input.relaxer_version}

    total = annex_input.total_generated
    validation_stats = {
        "validity_count": annex_input.validity_count,
        "uniqueness_count": annex_input.uniqueness_count,
        "novelty_count": annex_input.novelty_count,
        "total_generated": total,
    }

    novelty_fraction = (
        annex_input.novelty_count / total if total > 0 else 0.0
    )
    novelty_details = {
        "reference_dbs": list(annex_input.reference_db_ids),
        "matcher_tolerances": dict(annex_input.matcher_tolerances),
        "novelty_fraction": novelty_fraction,
    }

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
            "Zero structures generated -- verify generator configuration"
        )

    if total > 0 and annex_input.validity_count == 0:
        warnings.append(
            f"No candidates passed validity checks out of "
            f"{total} generated"
        )

    if annex_input.novelty_count == 0 and annex_input.uniqueness_count > 0:
        warnings.append(
            "All unique candidates matched existing database entries -- "
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
