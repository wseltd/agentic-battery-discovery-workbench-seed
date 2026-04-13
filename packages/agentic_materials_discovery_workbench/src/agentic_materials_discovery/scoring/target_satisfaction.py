"""Target satisfaction scoring for candidate materials.

Scores what fraction of user-specified constraints a candidate satisfies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def target_satisfaction_score(
    candidate: dict,
    constraints: dict,
) -> float:
    """Score what fraction of constraints a candidate satisfies.

    For each constraint entry:
    - str value: the candidate's field must match exactly.
    - int/float value: the candidate's field must be <= the constraint value.
    Missing fields in the candidate count as unsatisfied.

    Args:
        candidate: Candidate property dict (field name to value).
        constraints: Field name to expected value.  Empty dict returns 1.0.

    Returns:
        Fraction of constraints satisfied, in [0.0, 1.0].

    Raises:
        TypeError: If a constraint value is not str, int, or float.
    """
    if not constraints:
        return 1.0

    logger.info("Evaluating %d constraints", len(constraints))

    met = 0
    for field_name, expected in constraints.items():
        if isinstance(expected, str):
            if candidate.get(field_name) == expected:
                met += 1
        elif isinstance(expected, (int, float)):
            value = candidate.get(field_name)
            if value is not None and value <= expected:
                met += 1
        else:
            raise TypeError(
                f"Constraint for '{field_name}' has unsupported type "
                f"{type(expected).__name__}; expected str, int, or float"
            )

    score = met / len(constraints)
    logger.info("Target satisfaction score=%s (%d/%d met)", score, met, len(constraints))
    return score
