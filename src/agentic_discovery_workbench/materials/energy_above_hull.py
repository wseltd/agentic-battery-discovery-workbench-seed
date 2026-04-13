"""Convex-hull stability analysis for ML-relaxed crystal structures.

Computes energy above the thermodynamic convex hull for a candidate
structure by comparing against reference phases fetched from an external
database (e.g. Materials Project).  The hull distance is a proxy for
thermodynamic stability — structures below a configurable threshold are
classified as potentially stable.

All results carry evidence_level='ml_relaxed' and an
'incomplete_reference_coverage' caveat because ML-potential energies do
not reproduce DFT-quality phase boundaries.  These annotations are always
present to prevent downstream overconfidence in stability predictions.

Chose to include the candidate entry in the PhaseDiagram rather than
computing against a reference-only diagram.  This matches the standard
pymatgen workflow and avoids separate hull-only construction.  Trade-off:
a very-low-energy candidate shifts the hull — acceptable for an
ML-screening proxy, not for publication-quality thermodynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

logger = logging.getLogger(__name__)


# --- Constants ---------------------------------------------------------------

# Structures with energy above hull below this value (eV/atom) are classified
# as thermodynamically stable proxies.  0.1 eV/atom is the standard heuristic
# in computational materials screening (e.g. Materials Project dashboards).
STABILITY_THRESHOLD: float = 0.1

# Always appended to results — ML relaxation cannot guarantee complete
# reference coverage, so hull distances are inherently approximate.
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
        """Ensure incomplete_reference_coverage caveat is always present.

        Mutates the caveats list in-place — allowed under frozen=True
        because frozen only prevents attribute reassignment, not mutation
        of mutable attribute values.
        """
        if _INCOMPLETE_COVERAGE_CAVEAT not in self.caveats:
            self.caveats.append(_INCOMPLETE_COVERAGE_CAVEAT)


# --- Client protocol ---------------------------------------------------------


class PhaseEntryProvider(Protocol):
    """Protocol for reference phase data providers.

    Implementors fetch reference phase entries for a chemical system from
    an external database (e.g. Materials Project) and report whether
    energy corrections are applied.
    """

    def get_entries(self, chemical_system: str) -> list[PDEntry]:
        """Fetch reference phase entries for a chemical system.

        Args:
            chemical_system: Dash-separated elemental system (e.g. 'Fe-Li-O').

        Returns:
            List of PDEntry objects representing known phases.
        """
        ...  # pragma: no cover

    def has_corrections(self) -> bool:
        """Whether energy corrections (e.g. GGA/GGA+U) are applied.

        Returns:
            True if corrections are applied to the returned entries.
        """
        ...  # pragma: no cover

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{type(self).__name__}()"


# --- Calculator --------------------------------------------------------------


class EnergyAboveHullCalculator:
    """Compute energy above convex hull for ML-relaxed structures.

    Wraps pymatgen PhaseDiagram to compute hull distances for candidate
    structures against reference phases from an external database.  Reference
    entries are cached by chemical system to avoid redundant API calls when
    processing multiple candidates in the same system.

    The client must implement two methods:
        get_entries(chemical_system: str) -> list[PDEntry]
        has_corrections() -> bool

    Args:
        client: Reference phase data provider.
        stability_threshold_ev: eV/atom threshold for stability classification.

    Raises:
        ValueError: If stability_threshold_ev is negative.
    """

    def __init__(
        self,
        client: PhaseEntryProvider,
        stability_threshold_ev: float = STABILITY_THRESHOLD,
    ) -> None:
        if stability_threshold_ev < 0:
            raise ValueError(
                f"stability_threshold_ev must be non-negative, "
                f"got {stability_threshold_ev}"
            )
        self._client = client
        self._threshold = stability_threshold_ev
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
            ConnectionError: If the phase entry client fails (not silenced).
        """
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
        # Cast to native Python float — pymatgen returns np.float64 which
        # propagates numpy scalar types through downstream comparisons.
        raw_e_above_hull = phase_diagram.get_e_above_hull(candidate)
        if raw_e_above_hull is None:
            # Should not happen for an entry in the diagram, but pymatgen
            # type stubs declare float | None.
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

    def _get_reference_entries(
        self, chemical_system: str
    ) -> list[PDEntry]:
        """Fetch reference entries, using cache to avoid redundant queries.

        Args:
            chemical_system: Dash-separated elemental system.

        Returns:
            List of reference PDEntry objects.
        """
        if chemical_system in self._system_cache:
            return self._system_cache[chemical_system]

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

    For batch processing, construct EnergyAboveHullCalculator once and call
    compute() repeatedly to benefit from chemical system caching.

    Args:
        structure_id: Identifier for the structure.
        composition: Chemical composition string or Composition object.
        energy_per_atom_ev: ML-relaxed energy per atom in eV.
        client: Reference phase data provider with get_entries() and
            has_corrections() methods.
        stability_threshold_ev: eV/atom threshold for stability classification.

    Returns:
        HullEnergyResult with hull distance and stability classification.
    """
    calc = EnergyAboveHullCalculator(
        client=client, stability_threshold_ev=stability_threshold_ev
    )
    return calc.compute(structure_id, composition, energy_per_atom_ev)
