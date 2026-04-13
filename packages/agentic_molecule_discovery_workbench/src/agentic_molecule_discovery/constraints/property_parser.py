"""Parse property window tokens like 'MW<500' into PropertyConstraint objects.

Supports single-operator forms (MW<500, logP>=1.5, HBD=2) and range forms
(100<=TPSA<=140, 200<=MW<=500).  Operators: <, <=, >, >=, =.  Property names
are case-sensitive and must match ALLOWED_PROPERTIES exactly.

Chose regex over a hand-rolled state machine because every token fits on one
line and the grammar is small enough that two patterns cover all valid forms.
"""

from __future__ import annotations

import re

from agentic_molecule_discovery.constraints.models import ALLOWED_PROPERTIES, PropertyConstraint

# Range form: <number> <= <name> <= <number>  (only <= makes sense for ranges)
_RANGE_RE = re.compile(
    r"^(?P<lo>-?\d+(?:\.\d+)?)"
    r"\s*<=\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*<=\s*"
    r"(?P<hi>-?\d+(?:\.\d+)?)$"
)

# Single-operator form: <name><op><number>  or  <number><op><name>
_SINGLE_NAME_LEFT_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<op>[<>]=?|=)\s*"
    r"(?P<val>-?\d+(?:\.\d+)?)$"
)


def parse_property_constraint(token: str) -> PropertyConstraint:
    """Parse a property window token into a PropertyConstraint.

    Args:
        token: A string such as 'MW<500', 'logP>=1.5', '100<=TPSA<=140',
            or 'HBD=2'.  Whitespace around operators is tolerated.

    Returns:
        A validated PropertyConstraint with the parsed bounds.

    Raises:
        ValueError: If the token is malformed, uses an unknown property name,
            or cannot be parsed into a valid constraint.
    """
    stripped = token.strip()
    if not stripped:
        raise ValueError("Property constraint token is empty.")

    # Try range form first: '100<=TPSA<=140'
    m = _RANGE_RE.match(stripped)
    if m:
        name = m.group("name")
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        _validate_property_name(name, token)
        # PropertyConstraint validates lo <= hi internally.
        return PropertyConstraint(property_name=name, min_val=lo, max_val=hi)

    # Try single-operator form: 'MW<500', 'logP>=1.5', 'HBD=2'
    m = _SINGLE_NAME_LEFT_RE.match(stripped)
    if m:
        name = m.group("name")
        op = m.group("op")
        val = float(m.group("val"))
        _validate_property_name(name, token)
        return _apply_operator(name, op, val)

    raise ValueError(
        f"Malformed property constraint: {token!r}. "
        f"Expected forms: 'MW<500', 'logP>=1.5', '100<=TPSA<=140', 'HBD=2'."
    )


def _validate_property_name(name: str, original_token: str) -> None:
    """Raise ValueError if *name* is not in ALLOWED_PROPERTIES."""
    if name not in ALLOWED_PROPERTIES:
        sorted_names = sorted(ALLOWED_PROPERTIES)
        raise ValueError(
            f"Unknown property {name!r} in token {original_token!r}. "
            f"Allowed: {sorted_names}"
        )


def _apply_operator(
    name: str, op: str, val: float
) -> PropertyConstraint:
    """Convert a name-op-value triple into a PropertyConstraint."""
    if op == "<" or op == "<=":
        return PropertyConstraint(property_name=name, min_val=None, max_val=val)
    if op == ">" or op == ">=":
        return PropertyConstraint(property_name=name, min_val=val, max_val=None)
    if op == "=":
        return PropertyConstraint(property_name=name, min_val=val, max_val=val)

    # Unreachable given the regex, but explicit for safety.
    raise ValueError(f"Unsupported operator {op!r} in constraint for {name!r}.")
