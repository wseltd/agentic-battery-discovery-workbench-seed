"""Parse 'has:[SMARTS]' / '!has:[SMARTS]' tokens into SubstructureConstraint.

Accepts a single token string and returns a validated SubstructureConstraint.
The token format is: 'has:<SMARTS>' for must-match or '!has:<SMARTS>' for
must-not-match.  Whitespace around the token and the SMARTS portion is
stripped.  Raises ValueError for missing prefix, empty SMARTS, or invalid
SMARTS patterns.
"""

from __future__ import annotations

from agentic_molecule_discovery.constraints.models import SubstructureConstraint

# Prefixes checked in order — negated first to avoid '!has:' matching 'has:'.
_NEGATED_PREFIX = "!has:"
_POSITIVE_PREFIX = "has:"


def parse_smarts_constraint(token: str) -> SubstructureConstraint:
    """Parse a token into a SubstructureConstraint.

    Args:
        token: A string like 'has:[#6]' or '!has:c1ccccc1'.
            Leading/trailing whitespace is stripped from both the token
            and the extracted SMARTS.

    Returns:
        A SubstructureConstraint with the parsed SMARTS and must_match flag.

    Raises:
        ValueError: If the token does not start with 'has:' or '!has:',
            the SMARTS portion is empty/whitespace-only, or RDKit cannot
            parse the SMARTS.
    """
    stripped = token.strip()

    if stripped.startswith(_NEGATED_PREFIX):
        must_match = False
        smarts_raw = stripped[len(_NEGATED_PREFIX):]
    elif stripped.startswith(_POSITIVE_PREFIX):
        must_match = True
        smarts_raw = stripped[len(_POSITIVE_PREFIX):]
    else:
        raise ValueError(
            f"Token must start with 'has:' or '!has:', got: {token!r}"
        )

    smarts = smarts_raw.strip()

    if not smarts:
        raise ValueError(
            f"SMARTS pattern is empty after prefix in token: {token!r}"
        )

    # SubstructureConstraint validates the SMARTS via RDKit on construction.
    return SubstructureConstraint(smarts=smarts, must_match=must_match)
