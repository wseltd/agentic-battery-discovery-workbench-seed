"""Structural complexity scoring for candidate materials.

Scores how realistic a candidate's structural parameters are based on
element count, atoms per unit cell, and density.  Each dimension is
scored independently against configurable bounds, then averaged.

Chose linear penalty ramps (1.0 at the bound, 0.0 at 2x the bound)
rather than step functions because near-boundary candidates still
carry useful information and hard cutoffs would discard them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Allowed keys for custom bounds, matching the three scoring dimensions.
_VALID_BOUNDS_KEYS = frozenset({"num_elements", "atoms_per_cell", "density"})

# Default physical bounds for each dimension.
# num_elements: unary (1) through senaries (6) are the practical limit
# for synthesisable materials.
# atoms_per_cell: single atom up to ~200 covers most known structures.
# density: 0.5 g/cm³ (aerogel-like) to 25.0 g/cm³ (osmium-like).
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "num_elements": (1.0, 6.0),
    "atoms_per_cell": (1.0, 200.0),
    "density": (0.5, 25.0),
}


def _score_dimension(value: float, lower: float, upper: float) -> float:
    """Score a single dimension against its bounds.

    Within [lower, upper] the score is 1.0.  Below lower it decays
    linearly to 0.0 at lower/2.  Above upper it decays linearly to
    0.0 at 2*upper.

    Args:
        value: The observed value.
        lower: Lower bound (inclusive, must be positive).
        upper: Upper bound (inclusive).

    Returns:
        Dimension score in [0.0, 1.0].
    """
    if lower <= value <= upper:
        return 1.0

    if value < lower:
        # Linear decay: 1.0 at lower, 0.0 at lower/2
        half_lower = lower / 2.0
        if value <= half_lower:
            return 0.0
        return (value - half_lower) / (lower - half_lower)

    # value > upper: linear decay 1.0 at upper, 0.0 at 2*upper
    double_upper = 2.0 * upper
    if value >= double_upper:
        return 0.0
    return (double_upper - value) / (double_upper - upper)


def complexity_score(
    num_elements: int,
    atoms_per_cell: int,
    density: float,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Score structural complexity from element count, cell size, and density.

    Each dimension is scored independently: 1.0 if within bounds,
    linearly penalised toward 0.0 as the value moves to 2x outside the
    bound (half the lower bound or double the upper bound).  The final
    score is the average of the three dimension scores, clamped to [0, 1].

    Args:
        num_elements: Number of distinct chemical elements.
        atoms_per_cell: Number of atoms in the unit cell.
        density: Material density in g/cm^3.
        bounds: Override default bounds per dimension.  Keys must be from
            {"num_elements", "atoms_per_cell", "density"}.  Each value is
            a (lower, upper) tuple.

    Returns:
        Complexity score in [0.0, 1.0].

    Raises:
        ValueError: If num_elements, atoms_per_cell, or density is not positive.
        ValueError: If bounds contains unknown keys, inverted, or negative ranges.
    """
    if num_elements <= 0:
        raise ValueError(
            f"num_elements must be positive, got {num_elements}"
        )
    if atoms_per_cell <= 0:
        raise ValueError(
            f"atoms_per_cell must be positive, got {atoms_per_cell}"
        )
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")

    resolved = dict(DEFAULT_BOUNDS)
    if bounds:
        unknown = set(bounds) - _VALID_BOUNDS_KEYS
        if unknown:
            raise ValueError(
                f"Unknown bounds keys {sorted(unknown)}, "
                f"valid keys: {sorted(_VALID_BOUNDS_KEYS)}"
            )
        for key, (lo, hi) in bounds.items():
            if lo < 0 or hi < 0:
                raise ValueError(
                    f"Bounds for '{key}' must be non-negative, "
                    f"got ({lo}, {hi})"
                )
            if lo > hi:
                raise ValueError(
                    f"Bounds for '{key}' are inverted: lower={lo} > upper={hi}"
                )
        resolved.update(bounds)

    logger.info(
        "Scoring complexity num_elements=%d atoms_per_cell=%d density=%s",
        num_elements, atoms_per_cell, density,
    )

    elem_lo, elem_hi = resolved["num_elements"]
    cell_lo, cell_hi = resolved["atoms_per_cell"]
    dens_lo, dens_hi = resolved["density"]

    s_elem = _score_dimension(float(num_elements), elem_lo, elem_hi)
    s_cell = _score_dimension(float(atoms_per_cell), cell_lo, cell_hi)
    s_dens = _score_dimension(density, dens_lo, dens_hi)

    avg = (s_elem + s_cell + s_dens) / 3.0

    return min(1.0, max(0.0, avg))
