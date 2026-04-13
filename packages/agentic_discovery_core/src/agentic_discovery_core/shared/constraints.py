"""Shared constraint parser for user-facing discovery request constraints.

Parses a raw dict into structured ParsedConstraints with typed NumericRange
objects, validated SMARTS lists, and element allow/exclude lists.  Raises
ConstraintParseError with descriptive messages on malformed input.

Note: SMARTS validation requires rdkit at runtime.  If rdkit is not
available, SMARTS patterns are accepted without validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Numeric property names accepted in user-facing constraint dicts.
NUMERIC_PROPERTIES: dict[str, str] = {
    "MW": "MW",
    "cLogP": "cLogP",
    "logP": "cLogP",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "rotatable_bonds": "rotatable_bonds",
    "ring_count": "ring_count",
    "aromatic_rings": "aromatic_rings",
}

# Keys that receive special (non-numeric) parsing.
_SPECIAL_KEYS = frozenset({
    "smarts_required",
    "smarts_forbidden",
    "elements_allowed",
    "elements_excluded",
})


class ConstraintParseError(ValueError):
    """Raised when a constraint dict contains malformed or invalid data."""

    def __repr__(self) -> str:
        return f"ConstraintParseError({str(self)!r})"


@dataclass(frozen=True, slots=True)
class NumericRange:
    """A parsed numeric range with optional open-ended bounds.

    Args:
        min_val: Lower bound (inclusive), or None for unbounded below.
        max_val: Upper bound (inclusive), or None for unbounded above.
    """

    min_val: float | None = None
    max_val: float | None = None


@dataclass(slots=True)
class ParsedConstraints:
    """Structured result from parsing a user-facing constraint dict.

    Args:
        numeric: Mapping from canonical property name to NumericRange.
        smarts_required: SMARTS patterns the molecule must contain.
        smarts_forbidden: SMARTS patterns the molecule must not contain.
        elements_allowed: Permitted element symbols, or None (no restriction).
        elements_excluded: Forbidden element symbols, or None.
    """

    numeric: dict[str, NumericRange] = field(default_factory=dict)
    smarts_required: list[str] = field(default_factory=list)
    smarts_forbidden: list[str] = field(default_factory=list)
    elements_allowed: list[str] | None = None
    elements_excluded: list[str] | None = None


def parse_constraints(raw: dict) -> ParsedConstraints:
    """Parse a user-facing constraint dict into structured ParsedConstraints.

    Args:
        raw: Constraint dict from the user request.

    Returns:
        A ParsedConstraints populated with every constraint extracted
        from *raw*.

    Raises:
        ConstraintParseError: If any value is malformed.
    """
    logger.info("Parsing constraints with %d keys", len(raw))

    if not isinstance(raw, dict):
        raise ConstraintParseError(
            f"Constraints must be a dict, got {type(raw).__name__}."
        )

    result = ParsedConstraints()

    for key, value in raw.items():
        if key in _SPECIAL_KEYS:
            _parse_special(key, value, result)
        elif key in NUMERIC_PROPERTIES:
            canonical = NUMERIC_PROPERTIES[key]
            result.numeric[canonical] = _parse_numeric_range(key, value)
        else:
            raise ConstraintParseError(
                f"Unknown constraint key: {key!r}. "
                f"Valid numeric properties: {sorted(NUMERIC_PROPERTIES)}. "
                f"Valid special keys: {sorted(_SPECIAL_KEYS)}."
            )

    return result


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_numeric_range(key: str, spec: object) -> NumericRange:
    """Parse a ``{min, max}`` dict into a NumericRange."""
    if not isinstance(spec, dict):
        raise ConstraintParseError(
            f"Numeric constraint for '{key}' must be a dict with 'min' "
            f"and/or 'max' keys, got {type(spec).__name__}."
        )

    min_val = spec.get("min")
    max_val = spec.get("max")

    if min_val is None and max_val is None:
        raise ConstraintParseError(
            f"Numeric constraint for '{key}' must have at least "
            f"one of 'min' or 'max'."
        )

    if min_val is not None and not isinstance(min_val, (int, float)):
        raise ConstraintParseError(
            f"Numeric constraint '{key}' has non-numeric 'min' value: "
            f"{min_val!r} (type {type(min_val).__name__}). "
            f"Expected int or float."
        )
    if max_val is not None and not isinstance(max_val, (int, float)):
        raise ConstraintParseError(
            f"Numeric constraint '{key}' has non-numeric 'max' value: "
            f"{max_val!r} (type {type(max_val).__name__}). "
            f"Expected int or float."
        )

    if (
        min_val is not None
        and max_val is not None
        and min_val > max_val
    ):
        raise ConstraintParseError(
            f"Numeric constraint '{key}' has min ({min_val}) > max "
            f"({max_val}). The minimum must not exceed the maximum."
        )

    return NumericRange(
        min_val=float(min_val) if min_val is not None else None,
        max_val=float(max_val) if max_val is not None else None,
    )


def _validate_smarts(pattern: str) -> None:
    """Validate a SMARTS string via RDKit, raising ConstraintParseError on failure."""
    if not isinstance(pattern, str):
        raise ConstraintParseError(
            f"SMARTS pattern must be a string, got {type(pattern).__name__}."
        )
    stripped = pattern.strip()
    if not stripped:
        raise ConstraintParseError("SMARTS pattern must not be empty.")
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmarts(stripped)
        if mol is None:
            raise ConstraintParseError(
                f"Invalid SMARTS pattern: {pattern!r}. "
                f"Could not be parsed -- check for unbalanced brackets "
                f"or invalid atom symbols."
            )
    except ImportError:
        # rdkit not available -- accept the pattern without validation
        pass


def _parse_special(
    key: str, value: object, result: ParsedConstraints,
) -> None:
    """Parse a non-numeric constraint key into *result*."""
    if key in ("smarts_required", "smarts_forbidden"):
        if not isinstance(value, list):
            raise ConstraintParseError(
                f"'{key}' must be a list of SMARTS strings, "
                f"got {type(value).__name__}."
            )
        target = (
            result.smarts_required
            if key == "smarts_required"
            else result.smarts_forbidden
        )
        for pattern in value:
            _validate_smarts(pattern)
            target.append(pattern.strip())

    elif key in ("elements_allowed", "elements_excluded"):
        if not isinstance(value, list):
            raise ConstraintParseError(
                f"'{key}' must be a list of element symbols, "
                f"got {type(value).__name__}."
            )
        if key == "elements_allowed":
            result.elements_allowed = list(value)
        else:
            result.elements_excluded = list(value)
