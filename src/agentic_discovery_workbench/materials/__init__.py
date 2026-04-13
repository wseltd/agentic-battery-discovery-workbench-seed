"""Materials discovery modules for crystal structure analysis."""

from agentic_discovery_workbench.materials.competing_phases import (
    clear_phase_cache,
    fetch_competing_phases,
)
from agentic_discovery_workbench.materials.energy_above_hull import (
    STABILITY_THRESHOLD,
    EnergyAboveHullCalculator,
    HullEnergyResult,
)
from agentic_discovery_workbench.materials.energy_correction import (
    apply_mp2020_correction,
)

__all__ = [
    "EnergyAboveHullCalculator",
    "HullEnergyResult",
    "STABILITY_THRESHOLD",
    "apply_mp2020_correction",
    "clear_phase_cache",
    "fetch_competing_phases",
]
