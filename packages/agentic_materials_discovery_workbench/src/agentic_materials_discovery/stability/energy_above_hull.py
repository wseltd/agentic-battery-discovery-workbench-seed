"""Convex-hull stability analysis for ML-relaxed crystal structures.

Computes energy above the thermodynamic convex hull for a candidate
structure by comparing against reference phases fetched from an external
database (e.g. Materials Project).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

from agentic_materials_discovery.stability.competing_phases import fetch_competing_phases
from agentic_materials_discovery.stability.energy_correction import (
    apply_mp2020_correction,
    build_candidate_entry,
)

logger = logging.getLogger(__name__)


# --- Constants ---------------------------------------------------------------

STABILITY_THRESHOLD: float = 0.1

_INCOMPLETE_COVERAGE_CAVEAT = "incomplete_reference_coverage"


# --- Data structures ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HullEnergyResult:
    """Energy-above-hull analysis result for one crystal structure.

    Args:
        structure_id: Identifier for the analysed structure.
        energy_per_atom_ev: ML-relaxed energy per atom in eV.
        energy_above_hull_ev_per_atom: Distance above convex hull in eV/atom.
        is_stable_proxy: True if energy_above_hull < stability_threshold.
        reference_phase_count: Number of reference phases in the hull.
        chemical_system: Dash-separated elemental system (e.g. 'Fe-Li-O-P').
        stability_threshold_ev: Threshold used for the stable/unstable call.
        evidence_level: Always 'ml_relaxed' for ML-potential results.
        caveats: Caveat strings; always includes 'incomplete_reference_coverage'.
    """

    structure_id: str
    energy_per_atom_ev: float
    energy_above_hull_ev_per_atom: float
    is_stable_proxy: bool
    reference_phase_count: int
    chemical_system: str
    stability_threshold_ev: float = STABILITY_THRESHOLD
    evidence_level: str = "ml_relaxed"
    caveats: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure incomplete_reference_coverage caveat is always present."""
        if _INCOMPLETE_COVERAGE_CAVEAT not in self.caveats:
            self.caveats.append(_INCOMPLETE_COVERAGE_CAVEAT)


# --- Client protocol ---------------------------------------------------------


class PhaseEntryProvider(Protocol):
    """Protocol for reference phase data providers."""

    def get_entries(self, chemical_system: str) -> list[PDEntry]:
        """Fetch reference phase entries for a chemical system."""
        ...  # pragma: no cover

    def has_corrections(self) -> bool:
        """Whether energy corrections (e.g. GGA/GGA+U) are applied."""
        ...  # pragma: no cover

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{type(self).__name__}()"


# --- Calculator --------------------------------------------------------------


class EnergyAboveHullCalculator:
    """Compute energy above convex hull for ML-relaxed structures.

    Args:
        client: Reference phase data provider for compute().
        stability_threshold_ev: eV/atom threshold for stability classification.
        mp_api_key: Materials Project API key for calculate().

    Raises:
        ValueError: If stability_threshold_ev is negative.
    """

    def __init__(
        self,
        client: PhaseEntryProvider | None = None,
        stability_threshold_ev: float = STABILITY_THRESHOLD,
        mp_api_key: str | None = None,
    ) -> None:
        if stability_threshold_ev < 0:
            raise ValueError(
                f"stability_threshold_ev must be non-negative, "
                f"got {stability_threshold_ev}"
            )
        self._client = client
        self._threshold = stability_threshold_ev
        self._mp_api_key = mp_api_key
        self._system_cache: dict[str, list[PDEntry]] = {}

    def compute(
        self,
        structure_id: str,
        composition: str | Composition,
        energy_per_atom_ev: float,
    ) -> HullEnergyResult:
        """Compute energy above hull for a candidate structure.

        Args:
            structure_id: Identifier for the structure.
            composition: Chemical composition (formula string or Composition).
            energy_per_atom_ev: Energy per atom in eV from ML relaxation.

        Returns:
            HullEnergyResult with hull distance and stability classification.

        Raises:
            ValueError: If composition cannot be parsed.
            ConnectionError: If the phase entry client fails.
        """
        if self._client is None:
            raise ValueError(
                "compute() requires a client; pass client= to __init__ "
                "or use calculate() with mp_api_key instead"
            )

        logger.info(
            "Computing energy above hull structure_id=%s composition=%s",
            structure_id,
            composition,
        )

        comp = (
            Composition(composition)
            if isinstance(composition, str)
            else composition
        )
        chemical_system = comp.chemical_system

        entries = self._get_reference_entries(chemical_system)

        total_energy = energy_per_atom_ev * comp.num_atoms
        candidate = PDEntry(comp, total_energy)

        all_entries = entries + [candidate]
        phase_diagram = PhaseDiagram(all_entries)
        raw_e_above_hull = phase_diagram.get_e_above_hull(candidate)
        if raw_e_above_hull is None:
            raise ValueError(
                f"PhaseDiagram returned None for structure_id={structure_id}"
            )
        e_above_hull = float(raw_e_above_hull)
        is_stable = bool(e_above_hull < self._threshold)

        caveats: list[str] = []
        if not self._client.has_corrections():
            caveats.append("missing_energy_corrections")

        return HullEnergyResult(
            structure_id=structure_id,
            energy_per_atom_ev=energy_per_atom_ev,
            energy_above_hull_ev_per_atom=e_above_hull,
            is_stable_proxy=is_stable,
            stability_threshold_ev=self._threshold,
            reference_phase_count=len(entries),
            chemical_system=chemical_system,
            caveats=caveats,
        )

    def calculate(
        self,
        structure: Structure,
        energy_per_atom: float,
        structure_id: str,
    ) -> HullEnergyResult:
        """Compute energy above hull by orchestrating correction, phase fetch, and hull construction.

        Args:
            structure: ML-relaxed crystal structure.
            energy_per_atom: ML-predicted energy per atom in eV.
            structure_id: Identifier for the structure.

        Returns:
            HullEnergyResult with hull distance and stability classification.
        """
        logger.info(
            "Calculating energy above hull structure_id=%s formula=%s",
            structure_id,
            structure.composition.reduced_formula,
        )

        corrected_energy, caveats = apply_mp2020_correction(
            structure, energy_per_atom
        )

        chemical_system = structure.composition.chemical_system
        competing_entries = fetch_competing_phases(
            chemical_system, mp_api_key=self._mp_api_key
        )

        candidate = build_candidate_entry(structure, corrected_energy)

        all_entries = list(competing_entries) + [candidate]
        phase_diagram = PhaseDiagram(all_entries)

        raw_e_above_hull = phase_diagram.get_e_above_hull(candidate)
        if raw_e_above_hull is None:
            raise ValueError(
                f"PhaseDiagram returned None for structure_id={structure_id}"
            )
        e_above_hull = float(raw_e_above_hull)
        is_stable = bool(e_above_hull <= self._threshold)

        return HullEnergyResult(
            structure_id=structure_id,
            energy_per_atom_ev=energy_per_atom,
            energy_above_hull_ev_per_atom=e_above_hull,
            is_stable_proxy=is_stable,
            stability_threshold_ev=self._threshold,
            reference_phase_count=len(competing_entries),
            chemical_system=chemical_system,
            caveats=caveats,
        )

    def _get_reference_entries(
        self, chemical_system: str
    ) -> list[PDEntry]:
        """Fetch reference entries, using cache to avoid redundant queries."""
        if chemical_system in self._system_cache:
            return self._system_cache[chemical_system]

        if self._client is None:
            raise ValueError("No client configured")

        entries = self._client.get_entries(chemical_system)
        self._system_cache[chemical_system] = entries
        return entries


# --- Convenience function ----------------------------------------------------


def compute_energy_above_hull(
    structure_id: str,
    composition: str | Composition,
    energy_per_atom_ev: float,
    client: PhaseEntryProvider,
    stability_threshold_ev: float = STABILITY_THRESHOLD,
) -> HullEnergyResult:
    """Compute energy above hull for a single structure (convenience wrapper).

    Args:
        structure_id: Identifier for the structure.
        composition: Chemical composition string or Composition object.
        energy_per_atom_ev: ML-relaxed energy per atom in eV.
        client: Reference phase data provider.
        stability_threshold_ev: eV/atom threshold for stability classification.

    Returns:
        HullEnergyResult with hull distance and stability classification.
    """
    calc = EnergyAboveHullCalculator(
        client=client, stability_threshold_ev=stability_threshold_ev
    )
    return calc.compute(structure_id, composition, energy_per_atom_ev)
