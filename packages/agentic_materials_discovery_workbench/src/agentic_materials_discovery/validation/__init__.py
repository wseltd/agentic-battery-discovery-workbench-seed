"""Crystal structure validation: lattice sanity, elements, distances, coordination, dimensionality."""

from agentic_materials_discovery.validation.core import (
    validate_structure,
    StructureValidationResult,
)
from agentic_materials_discovery.validation.checks import (
    MaterialsValidationResult,
    validate_allowed_elements,
    validate_atom_count,
    validate_coordination_sanity,
    validate_interatomic_distances,
    validate_lattice_sanity,
)

__all__ = [
    "MaterialsValidationResult",
    "StructureValidationResult",
    "validate_allowed_elements",
    "validate_atom_count",
    "validate_coordination_sanity",
    "validate_interatomic_distances",
    "validate_lattice_sanity",
    "validate_structure",
]
