"""Pre-ranking candidate validation.

Filters candidates whose score dicts contain NaN values or are missing
required objective keys.  Returns valid candidates and a list of
human-readable warnings -- never raises exceptions.
"""

from __future__ import annotations

from typing import Any

# ScoredCandidate is the same dict shape used by ranker.py:
# {"candidate_id": str, "scores": dict[str, float]}
ScoredCandidate = dict[str, Any]


def validate_candidates(
    candidates: list[ScoredCandidate],
    objective_names: list[str],
) -> tuple[list[ScoredCandidate], list[str]]:
    """Filter candidates that have NaN scores or missing objective keys.

    Parameters
    ----------
    candidates:
        Each dict must have ``candidate_id`` (str) and ``scores``
        (dict mapping objective name to float).
    objective_names:
        The objective keys every candidate's ``scores`` dict must contain.

    Returns
    -------
    tuple[list[ScoredCandidate], list[str]]
        ``(valid_candidates, warnings)`` where *warnings* lists each
        rejected candidate_id and the reason.  No exceptions are raised.
    """
    valid: list[ScoredCandidate] = []
    warnings: list[str] = []

    for cand in candidates:
        cid = cand["candidate_id"]
        scores = cand["scores"]

        # Check for missing objective keys first.
        missing = [name for name in objective_names if name not in scores]
        if missing:
            warnings.append(
                f"Candidate {cid!r} rejected: missing objectives {missing}"
            )
            continue

        # Check for NaN values -- val != val is the standard IEEE 754 NaN test.
        nan_keys = [
            key for key in objective_names if scores[key] != scores[key]  # noqa: PLR0124
        ]
        if nan_keys:
            warnings.append(
                f"Candidate {cid!r} rejected: NaN scores on {nan_keys}"
            )
            continue

        valid.append(cand)

    return valid, warnings
