"""Energy-above-hull proxy classification for ML-relaxed structures.

Classifies candidate materials into three stability bands based on
energy above the convex hull, computed as the difference between the
candidate's energy per atom and the lowest competing phase energy.

Three-band classification (research-pack Q26):
  - stable:     energy_above_hull <= 0.0 eV/atom
  - metastable: 0.0 < energy_above_hull <= 0.1 eV/atom
  - unstable:   energy_above_hull > 0.1 eV/atom

This is a pure classification function — no pymatgen, no external API
calls, no relaxation.  It operates on pre-computed energy values only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Boundary between stable and metastable (inclusive upper bound for stable).
STABLE_THRESHOLD_EV: float = 0.0

# Boundary between metastable and unstable (inclusive upper bound for metastable).
METASTABLE_THRESHOLD_EV: float = 0.1

# Float comparison tolerance for threshold boundaries.  ML energies have
# noise orders of magnitude larger than this, so 1e-9 eV/atom is safe.
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
    """Map hull distance to a three-band stability label.

    Args:
        energy_above_hull_ev: Energy above hull in eV/atom.

    Returns:
        Classification string: 'stable', 'metastable', or 'unstable'.
    """
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

    Computes hull distance as the difference between the candidate's
    energy per atom and the lowest-energy competing phase.  When no
    competing phases are provided the hull distance is undefined —
    the result is classified as 'unstable' with a caveat.

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
