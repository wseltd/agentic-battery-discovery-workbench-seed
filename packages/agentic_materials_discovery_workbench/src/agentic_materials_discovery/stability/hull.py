"""Energy-above-hull proxy classification for ML-relaxed structures.

Classifies candidate materials into three stability bands based on
energy above the convex hull.

Three-band classification:
  - stable:     energy_above_hull <= 0.0 eV/atom
  - metastable: 0.0 < energy_above_hull <= 0.1 eV/atom
  - unstable:   energy_above_hull > 0.1 eV/atom
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Boundary between stable and metastable.
STABLE_THRESHOLD_EV: float = 0.0

# Boundary between metastable and unstable.
METASTABLE_THRESHOLD_EV: float = 0.1

# Float comparison tolerance for threshold boundaries.
_FLOAT_TOL: float = 1e-9

# Caveat attached when no competing phases are available.
_MISSING_PHASES_CAVEAT = (
    "no_competing_phases: hull distance is undefined without reference phases"
)


@dataclass(frozen=True, slots=True)
class HullResult:
    """Energy-above-hull classification result.

    Args:
        energy_above_hull_ev: Hull distance in eV/atom.
        classification: One of 'stable', 'metastable', 'unstable'.
        evidence_level: Provenance tag for the energy source.
        caveat: Optional caveat string when result quality is degraded.
    """

    energy_above_hull_ev: float
    classification: str
    evidence_level: str
    caveat: str | None


def _classify(energy_above_hull_ev: float) -> str:
    """Map hull distance to a three-band stability label."""
    if energy_above_hull_ev <= STABLE_THRESHOLD_EV + _FLOAT_TOL:
        return "stable"
    if energy_above_hull_ev <= METASTABLE_THRESHOLD_EV + _FLOAT_TOL:
        return "metastable"
    return "unstable"


def estimate_energy_above_hull(
    energy_per_atom: float,
    competing_phase_energies: dict[str, float],
) -> HullResult:
    """Estimate energy above hull and classify stability.

    Args:
        energy_per_atom: ML-predicted energy per atom in eV.
        competing_phase_energies: Mapping of phase identifiers to
            energy-per-atom values in eV.  May be empty.

    Returns:
        HullResult with classification and evidence metadata.
    """
    logger.info(
        "Estimating energy above hull energy_per_atom=%s n_phases=%d",
        energy_per_atom,
        len(competing_phase_energies),
    )

    if not competing_phase_energies:
        return HullResult(
            energy_above_hull_ev=float("inf"),
            classification="unstable",
            evidence_level="ml_predicted",
            caveat=_MISSING_PHASES_CAVEAT,
        )

    lowest_competing = min(competing_phase_energies.values())
    e_above_hull = energy_per_atom - lowest_competing

    return HullResult(
        energy_above_hull_ev=e_above_hull,
        classification=_classify(e_above_hull),
        evidence_level="ml_predicted",
        caveat=None,
    )
