"""Request and routing schemas for the discovery workbench.

Dataclass definitions for MoleculeRequest, MaterialsRequest, and RoutingResult.
These are the shared vocabulary that every downstream ticket references.
Mutable request classes (enriched by pipeline steps); frozen routing result.

No Pydantic — stdlib dataclasses only, keeping the dependency surface minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# Allowed task types for molecule requests — canonical source, import elsewhere.
MOLECULE_TASK_TYPES = frozenset({"de_novo", "scaffold_constrained", "optimise_existing"})

# Allowed routing domains — canonical source.
ROUTING_DOMAINS = frozenset(
    {"small_molecule_design", "inorganic_materials_design", "unsupported"}
)


@dataclass
class MoleculeRequest:
    """Small-molecule design request.

    Parameters
    ----------
    task_type:
        One of 'de_novo', 'scaffold_constrained', 'optimise_existing'.
    objective:
        Free-text description of the design goal.
    constraints:
        Raw constraint strings keyed by property name (e.g. {'MW': '300-450'}).
        Parsing into numeric ranges happens in T004, not here.
    output_count:
        Number of candidate molecules to generate.
    reference_set:
        Novelty-check database identifier.
    starting_scaffold:
        SMILES scaffold for constrained generation; None for de novo.
    starting_molecules:
        Seed SMILES for optimisation tasks; None otherwise.
    property_priorities:
        Optional weight map for multi-objective ranking.
    """

    task_type: str
    objective: str
    constraints: dict[str, str]
    output_count: int
    reference_set: str = "ChEMBL_36"
    starting_scaffold: str | None = None
    starting_molecules: list[str] | None = None
    property_priorities: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ValueError("task_type is required")
        if self.task_type not in MOLECULE_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(MOLECULE_TASK_TYPES)}, "
                f"got {self.task_type!r}"
            )
        if not self.objective:
            raise ValueError("objective is required")
        if not isinstance(self.constraints, dict):
            raise TypeError(
                f"constraints must be a dict, got {type(self.constraints).__name__}"
            )
        if not isinstance(self.output_count, int) or self.output_count < 1:
            raise ValueError(f"output_count must be a positive integer, got {self.output_count!r}")


@dataclass
class MaterialsRequest:
    """Inorganic materials design request.

    Parameters
    ----------
    chemistry_scope:
        Element system description (e.g. 'Li-Fe-O', 'ternary oxides').
    structure_size_limit:
        Maximum atoms per unit cell.
    symmetry_request:
        Desired symmetry constraint or 'any'.
    stability_target:
        Energy-above-hull threshold in eV/atom.
    output_count:
        Number of candidate structures to generate.
    band_gap_eV:
        Optional band-gap constraint as raw string (e.g. '1.5-3.0').
    bulk_modulus_GPa:
        Optional bulk modulus constraint as raw string.
    magnetic_density:
        Optional magnetic density constraint as raw string.
    exclude_elements:
        Elements to exclude from generation.
    allow_P1:
        Whether to allow P1 (no symmetry) structures.
    """

    chemistry_scope: str
    output_count: int
    structure_size_limit: int = 20
    symmetry_request: str = "any"
    stability_target: float = 0.1
    band_gap_eV: str | None = None
    bulk_modulus_GPa: str | None = None
    magnetic_density: str | None = None
    exclude_elements: list[str] | None = None
    allow_P1: bool = False

    def __post_init__(self) -> None:
        if not self.chemistry_scope:
            raise ValueError("chemistry_scope is required")
        if not isinstance(self.output_count, int) or self.output_count < 1:
            raise ValueError(
                f"output_count must be a positive integer, got {self.output_count!r}"
            )
        if not isinstance(self.structure_size_limit, int) or self.structure_size_limit < 1:
            raise ValueError(
                f"structure_size_limit must be a positive integer, "
                f"got {self.structure_size_limit!r}"
            )
        if not isinstance(self.stability_target, (int, float)) or self.stability_target < 0:
            raise ValueError(
                f"stability_target must be a non-negative number, "
                f"got {self.stability_target!r}"
            )


@dataclass(frozen=True)
class RoutingResult:
    """Immutable result of the domain router.

    Frozen because routing decisions must not be mutated after the router
    commits — downstream steps read but never alter the routing verdict.

    Parameters
    ----------
    domain:
        One of 'small_molecule_design', 'inorganic_materials_design', 'unsupported'.
    confidence:
        Router confidence score in [0, 1].
    clarification_question:
        Follow-up question when the router cannot decide; None otherwise.
    parsed_request:
        The domain-specific request if routing succeeded; None for 'unsupported'.
    """

    domain: str
    confidence: float
    clarification_question: str | None
    parsed_request: Union[MoleculeRequest, MaterialsRequest, None]

    def __post_init__(self) -> None:
        if self.domain not in ROUTING_DOMAINS:
            raise ValueError(
                f"domain must be one of {sorted(ROUTING_DOMAINS)}, "
                f"got {self.domain!r}"
            )
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a number, got {type(self.confidence).__name__}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
