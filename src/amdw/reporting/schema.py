"""Structured report schema for the agentic discovery workbench.

Dataclass definitions for DiscoveryReport (shared top-level fields),
MoleculeAnnex (small-molecule branch annex), and ReportMaterialsAnnex
(inorganic-materials branch annex).  These are pure data containers
with no serialisation or file I/O — downstream renderers consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from amdw.shared.evidence import EvidenceLevel

__all__ = ["DiscoveryReport", "MoleculeAnnex", "ReportMaterialsAnnex"]

# Allowed branch identifiers.
VALID_BRANCHES = frozenset({"molecule", "material"})


@dataclass(slots=True)
class MoleculeAnnex:
    """Annex section for a small-molecule discovery report.

    Args:
        generator: Name/version of the molecular generator used.
        validity_counts: Dict with keys 'valid', 'invalid', 'total'.
        uniqueness_count: Number of structurally unique molecules.
        novelty_counts: Dict with keys 'exact_known', 'close_analogue', 'novel_like'.
        constraint_satisfaction: Per-constraint pass/fail summary.
        export_bundle_paths: Mapping of bundle name to filesystem path.
        warnings: Diagnostic messages for the report consumer.
    """

    generator: str
    validity_counts: dict[str, int]
    uniqueness_count: int
    novelty_counts: dict[str, int]
    constraint_satisfaction: dict[str, object]
    export_bundle_paths: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReportMaterialsAnnex:
    """Annex section for an inorganic-materials discovery report.

    Args:
        generator: Name/version of the materials generator used.
        scope_filters: Scope constraints applied during generation.
        relaxation_backend: Identifier for the relaxation engine.
        validation_summary: Aggregate validation statistics.
        dft_handoff_paths: Mapping of candidate to DFT input path.
        warnings: Diagnostic messages for the report consumer.
    """

    generator: str
    scope_filters: dict[str, object]
    relaxation_backend: str
    validation_summary: dict[str, object]
    dft_handoff_paths: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiscoveryReport:
    """Top-level structured report for a single discovery run.

    Args:
        run_id: Unique identifier for the run.
        timestamp: ISO-8601 timestamp string.
        branch: Discovery branch — 'molecule' or 'material'.
        tool_versions: Mapping of tool names to version strings.
        user_brief: The original user request.
        parsed_constraints: Constraints extracted from the brief.
        budget: Budget parameters for the run.
        evidence_levels_used: Evidence tiers observed during the run.
        shortlist_summary: Ranked candidate summaries.
    """

    run_id: str
    timestamp: str
    branch: str
    tool_versions: dict[str, str]
    user_brief: str
    parsed_constraints: dict[str, object]
    budget: dict[str, int]
    evidence_levels_used: list[EvidenceLevel]
    shortlist_summary: list[dict[str, object]]
