"""Confidence computation for keyword-density scoring.

Pure arithmetic: hits / total, clamped to [0.0, 1.0], with safe
zero-denominator handling.  No side effects, no imports beyond builtins.

Chose simple ratio over weighted scoring because the caller (scorer.py)
already controls which tokens count as hits vs. total.  Adding weights
here would duplicate domain knowledge.
"""

from __future__ import annotations


def compute_confidence(domain_keyword_hits: int, total_tokens: int) -> float:
    """Compute keyword-density confidence score.

    Parameters
    ----------
    domain_keyword_hits:
        Number of tokens that matched a domain keyword.
    total_tokens:
        Total number of tokens in the input.  When zero, the result
        is 0.0 (no signal).

    Returns
    -------
    float
        ``domain_keyword_hits / total_tokens`` clamped to [0.0, 1.0].
    """
    if total_tokens == 0:
        return 0.0
    raw = domain_keyword_hits / total_tokens
    return max(0.0, min(raw, 1.0))
