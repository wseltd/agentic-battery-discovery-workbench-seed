"""Tests for energy-above-hull calculation.

Uses real pymatgen PhaseDiagram with simple reference entries rather than
mocking the phase diagram — this verifies actual hull computation, not
just wiring.  Only the external client (API calls) is faked.
"""

from __future__ import annotations

import pytest
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition

from agentic_discovery_workbench.materials.energy_above_hull import (
    STABILITY_THRESHOLD,
    EnergyAboveHullCalculator,
    HullEnergyResult,
    compute_energy_above_hull,
)


# --- Test helpers ------------------------------------------------------------


class _MockClient:
    """Fake phase entry provider that tracks query count."""

    def __init__(
        self,
        entries_by_system: dict[str, list[PDEntry]] | None = None,
        corrections: bool = True,
    ) -> None:
        self._entries = entries_by_system or {}
        self._corrections = corrections
        self.get_entries_call_count = 0

    def get_entries(self, chemical_system: str) -> list[PDEntry]:
        """Return pre-loaded entries and increment counter."""
        self.get_entries_call_count += 1
        return self._entries.get(chemical_system, [])

    def has_corrections(self) -> bool:
        """Whether energy corrections are applied."""
        return self._corrections


def _fe_reference_entries() -> list[PDEntry]:
    """Single Fe reference at -4.0 eV/atom (1 atom)."""
    return [PDEntry(Composition("Fe"), -4.0)]


def _binary_li_fe_entries() -> list[PDEntry]:
    """Li-Fe binary with a stable intermetallic on the hull.

    Li: -1.5 eV/atom, Fe: -4.0 eV/atom, LiFe: -3.0 eV/atom (2 atoms).
    Linear interpolation at 50%% Fe = -2.75 eV/atom, so LiFe at -3.0
    is 0.25 eV below the tie line — on the hull.
    """
    return [
        PDEntry(Composition("Li"), -1.5),
        PDEntry(Composition("Fe"), -4.0),
        PDEntry(Composition("LiFe"), -6.0),
    ]


# --- Stability classification ------------------------------------------------


def test_stable_structure_below_threshold() -> None:
    """Structure 0.05 eV/atom above hull (< 0.1 threshold) is stable."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("fe-stable", "Fe", -3.95)
    assert result.is_stable_proxy is True
    assert result.energy_above_hull_ev_per_atom < STABILITY_THRESHOLD
    assert result.energy_above_hull_ev_per_atom == pytest.approx(0.05, abs=1e-6)


def test_unstable_structure_above_threshold() -> None:
    """Structure 0.5 eV/atom above hull (> 0.1 threshold) is unstable."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("fe-unstable", "Fe", -3.5)
    assert result.is_stable_proxy is False
    assert result.energy_above_hull_ev_per_atom > STABILITY_THRESHOLD
    assert result.energy_above_hull_ev_per_atom == pytest.approx(0.5, abs=1e-6)


def test_on_hull_structure_returns_zero() -> None:
    """Structure at exactly the reference energy has e_above_hull = 0."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("fe-hull", "Fe", -4.0)
    assert result.energy_above_hull_ev_per_atom == pytest.approx(0.0, abs=1e-10)
    assert result.is_stable_proxy is True


# --- Threshold and constants -------------------------------------------------


def test_default_threshold_is_0_1_ev() -> None:
    """Module constant and calculator default both equal 0.1 eV/atom."""
    assert STABILITY_THRESHOLD == 0.1
    client = _MockClient()
    calc = EnergyAboveHullCalculator(client=client)
    assert calc._threshold == 0.1


def test_negative_threshold_rejected() -> None:
    """Negative stability threshold raises ValueError at construction."""
    client = _MockClient()
    with pytest.raises(ValueError) as exc_info:
        EnergyAboveHullCalculator(client=client, stability_threshold_ev=-0.1)
    assert "non-negative" in str(exc_info.value)


# --- Caveat and evidence handling --------------------------------------------


def test_incomplete_coverage_caveat_always_present() -> None:
    """Caveat is auto-appended when not provided and not duplicated when present."""
    result_without = HullEnergyResult(
        structure_id="t1",
        energy_per_atom_ev=-3.0,
        energy_above_hull_ev_per_atom=0.05,
        is_stable_proxy=True,
        reference_phase_count=5,
        chemical_system="Fe-Li",
    )
    assert "incomplete_reference_coverage" in result_without.caveats

    result_with = HullEnergyResult(
        structure_id="t2",
        energy_per_atom_ev=-3.0,
        energy_above_hull_ev_per_atom=0.05,
        is_stable_proxy=True,
        reference_phase_count=5,
        chemical_system="Fe-Li",
        caveats=["incomplete_reference_coverage"],
    )
    assert result_with.caveats.count("incomplete_reference_coverage") == 1


def test_evidence_level_always_ml_relaxed() -> None:
    """Default evidence_level is 'ml_relaxed' and calculator preserves it."""
    result = HullEnergyResult(
        structure_id="t3",
        energy_per_atom_ev=-3.0,
        energy_above_hull_ev_per_atom=0.05,
        is_stable_proxy=True,
        reference_phase_count=3,
        chemical_system="Fe",
    )
    assert result.evidence_level == "ml_relaxed"

    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    computed = calc.compute("fe-ev", "Fe", -3.9)
    assert computed.evidence_level == "ml_relaxed"


def test_missing_correction_adds_caveat() -> None:
    """Client without corrections triggers 'missing_energy_corrections' caveat."""
    client = _MockClient(
        entries_by_system={"Fe": _fe_reference_entries()},
        corrections=False,
    )
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("fe-nocorr", "Fe", -3.9)
    assert "missing_energy_corrections" in result.caveats
    assert "incomplete_reference_coverage" in result.caveats


# --- Caching -----------------------------------------------------------------


def test_chemical_system_cache_avoids_redundant_queries() -> None:
    """Two computes on the same system query the client only once."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    calc.compute("fe-1", "Fe", -3.95)
    calc.compute("fe-2", "Fe", -3.80)
    assert client.get_entries_call_count == 1


# --- Field completeness and typing -------------------------------------------


def test_result_fields_complete() -> None:
    """All HullEnergyResult fields are accessible and correctly typed."""
    result = HullEnergyResult(
        structure_id="full-check",
        energy_per_atom_ev=-3.5,
        energy_above_hull_ev_per_atom=0.08,
        is_stable_proxy=True,
        reference_phase_count=12,
        chemical_system="Fe-Li-O-P",
        stability_threshold_ev=0.1,
        evidence_level="ml_relaxed",
        caveats=["incomplete_reference_coverage"],
    )
    assert result.structure_id == "full-check"
    assert result.energy_per_atom_ev == -3.5
    assert result.energy_above_hull_ev_per_atom == 0.08
    assert result.is_stable_proxy is True
    assert result.stability_threshold_ev == 0.1
    assert result.reference_phase_count == 12
    assert result.chemical_system == "Fe-Li-O-P"
    assert result.evidence_level == "ml_relaxed"
    assert "incomplete_reference_coverage" in result.caveats


# --- Element systems ---------------------------------------------------------


def test_single_element_system() -> None:
    """Single-element system produces correct chemical_system and hull distance."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("fe-single", "Fe", -3.8)
    assert result.chemical_system == "Fe"
    assert result.reference_phase_count == 1
    assert result.energy_above_hull_ev_per_atom == pytest.approx(0.2, abs=1e-6)


def test_reference_phase_count_populated() -> None:
    """reference_phase_count matches the number of reference entries provided."""
    entries = _binary_li_fe_entries()
    client = _MockClient(entries_by_system={"Fe-Li": entries})
    calc = EnergyAboveHullCalculator(client=client)
    result = calc.compute("life-1", "LiFe", -2.5)
    assert result.reference_phase_count == len(entries)


# --- Error propagation -------------------------------------------------------


def test_convenience_wrapper_delegates_correctly() -> None:
    """compute_energy_above_hull produces same result as the calculator."""
    client = _MockClient(entries_by_system={"Fe": _fe_reference_entries()})
    result = compute_energy_above_hull("fe-wrap", "Fe", -3.9, client)
    assert result.structure_id == "fe-wrap"
    assert result.energy_above_hull_ev_per_atom == pytest.approx(0.1, abs=1e-6)
    assert "incomplete_reference_coverage" in result.caveats


def test_api_failure_raises_not_silenced() -> None:
    """Client errors propagate to the caller — not silently caught."""

    class _FailingClient:
        def get_entries(self, chemical_system: str) -> list:
            raise ConnectionError("MP API unavailable")

        def has_corrections(self) -> bool:
            return True

    calc = EnergyAboveHullCalculator(client=_FailingClient())
    with pytest.raises(ConnectionError) as exc_info:
        calc.compute("fail-1", "Fe", -4.0)
    assert "MP API unavailable" in str(exc_info.value)
