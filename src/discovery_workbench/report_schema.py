"""Run-report schema for the discovery workbench.

Dataclass definitions for BudgetSettings, ShortlistEntry, and Report.
These capture the structured output of a single discovery run — budget
parameters, ranked candidate shortlist, and run metadata.

No serialisation, no file I/O — pure data containers with boundary
validation in __post_init__.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from discovery_workbench.evidence import EvidenceLevel

# Canonical set of valid discovery branches.
VALID_BRANCHES = frozenset({"small_molecule", "inorganic_materials"})

# Valid evidence-level labels, derived from the EvidenceLevel enum so
# we never duplicate the vocabulary.
_EVIDENCE_LABELS: frozenset[str] = frozenset(
    member.value[0] for member in EvidenceLevel
)


@dataclass
class BudgetSettings:
    """Computational budget parameters for a discovery run.

    Parameters
    ----------
    max_cycles:
        Maximum number of generate-evaluate cycles.
    max_batches:
        Maximum number of candidate batches to process.
    shortlist_size:
        Number of top candidates to retain in the final shortlist.
    """

    max_cycles: int
    max_batches: int
    shortlist_size: int


@dataclass
class ShortlistEntry:
    """A single candidate in the ranked shortlist.

    Parameters
    ----------
    candidate_id:
        Unique identifier for the candidate (must be non-empty).
    scores:
        Property scores keyed by property name.
    evidence_level:
        Evidence credibility label (must match an EvidenceLevel member label).
    rank:
        1-based rank position in the shortlist (must be >= 1).
    """

    candidate_id: str
    scores: dict[str, float]
    evidence_level: str
    rank: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(
                f"rank must be an integer >= 1, got {self.rank!r}"
            )
        if self.evidence_level not in _EVIDENCE_LABELS:
            raise ValueError(
                f"evidence_level must be one of {sorted(_EVIDENCE_LABELS)}, "
                f"got {self.evidence_level!r}"
            )


@dataclass
class Report:
    """Structured output of a single discovery run.

    Parameters
    ----------
    run_id:
        Unique identifier for the run (must be non-empty).
    timestamp:
        ISO 8601 timestamp string for when the run started.
    branch:
        Discovery branch — one of 'small_molecule' or 'inorganic_materials'.
    tool_versions:
        Mapping of tool/library names to version strings.
    user_brief:
        The original user request that initiated the run.
    parsed_constraints:
        Constraints extracted from the user brief, keyed by property name.
    budget:
        Computational budget settings for the run.
    stop_reason:
        Why the run stopped (e.g. 'budget_exhausted', 'converged'); None
        if not yet finished or if the reason is unknown.
    evidence_legend:
        Mapping of evidence-level labels to human-readable descriptions.
    shortlist:
        Ranked list of top candidates.
    warnings:
        Diagnostic messages emitted during the run.
    annexes:
        Arbitrary supplementary data keyed by annex name.
    """

    run_id: str
    timestamp: str
    branch: str
    tool_versions: dict[str, str]
    user_brief: str
    parsed_constraints: dict[str, Any]
    budget: BudgetSettings
    stop_reason: str | None
    evidence_legend: dict[str, str]
    shortlist: list[ShortlistEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    annexes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if self.branch not in VALID_BRANCHES:
            raise ValueError(
                f"branch must be one of {sorted(VALID_BRANCHES)}, "
                f"got {self.branch!r}"
            )
        # Validate that the timestamp is parseable as ISO 8601.
        try:
            datetime.datetime.fromisoformat(
                self.timestamp.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"timestamp must be a valid ISO 8601 string, "
                f"got {self.timestamp!r}"
            ) from exc
