"""Typed constraint data models for molecular property and substructure filtering.

Provides PropertyConstraint for numeric property bounds and SubstructureConstraint
for SMARTS pattern matching, both with eager validation on construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

# Single source of truth for allowed molecular property names.
# Other modules should import this rather than defining their own copy.
ALLOWED_PROPERTIES: frozenset[str] = frozenset({
    "MW",
    "logP",
    "TPSA",
    "HBD",
    "HBA",
    "rotatable_bonds",
    "ring_count",
    "aromatic_rings",
})


class PropertyConstraint:
    """A numeric bound on a molecular property.

    Args:
        property_name: Must be one of ALLOWED_PROPERTIES.
        min_val: Lower bound (inclusive), or None for unbounded.
        max_val: Upper bound (inclusive), or None for unbounded.

    Raises:
        ValueError: If property_name is not in ALLOWED_PROPERTIES,
            or both bounds are None, or min_val > max_val.
    """

    __slots__ = ("property_name", "min_val", "max_val")

    def __init__(
        self,
        property_name: str,
        min_val: float | None,
        max_val: float | None,
    ) -> None:
        if property_name not in ALLOWED_PROPERTIES:
            sorted_names = sorted(ALLOWED_PROPERTIES)
            raise ValueError(
                f"Unknown property {property_name!r}. "
                f"Allowed: {sorted_names}"
            )
        if min_val is None and max_val is None:
            raise ValueError(
                f"PropertyConstraint for {property_name!r} must have "
                f"at least one of min_val or max_val."
            )
        if (
            min_val is not None
            and max_val is not None
            and min_val > max_val
        ):
            raise ValueError(
                f"min_val ({min_val}) > max_val ({max_val}) "
                f"for property {property_name!r}."
            )
        self.property_name = property_name
        self.min_val = min_val
        self.max_val = max_val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropertyConstraint):
            return NotImplemented
        return (
            self.property_name == other.property_name
            and self.min_val == other.min_val
            and self.max_val == other.max_val
        )

    def __hash__(self) -> int:
        return hash((self.property_name, self.min_val, self.max_val))

    def __repr__(self) -> str:
        return (
            f"PropertyConstraint(property_name={self.property_name!r}, "
            f"min_val={self.min_val!r}, max_val={self.max_val!r})"
        )


@dataclass(frozen=True)
class SubstructureConstraint:
    """A SMARTS-based substructure presence/absence constraint.

    Args:
        smarts: A SMARTS pattern string. Validated via RDKit on creation.
        must_match: If True the molecule must contain the substructure;
            if False it must not.

    Raises:
        ValueError: If the SMARTS string is empty or unparseable by RDKit.
    """

    smarts: str
    must_match: bool = True

    def __post_init__(self) -> None:
        if not self.smarts:
            raise ValueError("SMARTS string must not be empty.")
        pattern = Chem.MolFromSmarts(self.smarts)
        if pattern is None:
            raise ValueError(
                f"Invalid SMARTS pattern: {self.smarts!r}. "
                f"RDKit could not parse it."
            )


# Type alias — a constraint list can mix property and substructure constraints.
ConstraintList = list[PropertyConstraint | SubstructureConstraint]
