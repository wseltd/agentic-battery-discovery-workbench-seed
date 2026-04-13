"""Tests for energy correction module.

Tests mock the module-level compatibility instance to avoid requiring
pymatgen's full correction data files.  Only the boundary between
our code and pymatgen is mocked — ComputedEntry construction and
energy arithmetic use real pymatgen objects.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry

from agentic_discovery_workbench.materials.energy_correction import (
    UNCORRECTED_CAVEAT,
    apply_mp2020_correction,
    build_candidate_entry,
)


def _simple_structure(species: str = "Fe") -> Structure:
    """Create a minimal cubic structure for testing."""
    lattice = Lattice.cubic(3.0)
    return Structure(lattice, [species], [[0.0, 0.0, 0.0]])


# --- apply_mp2020_correction -------------------------------------------------


def test_correction_success_returns_adjusted_energy() -> None:
    """Successful correction returns the adjusted per-atom energy."""
    structure = _simple_structure("Fe")
    ml_energy = -4.0

    corrected_entry = ComputedEntry(structure.composition, -4.5)

    with patch(
        "agentic_discovery_workbench.materials.energy_correction._COMPAT"
    ) as mock_compat:
        mock_compat.process_entries.return_value = [corrected_entry]
        energy, caveats = apply_mp2020_correction(structure, ml_energy)

    assert energy == pytest.approx(corrected_entry.energy_per_atom)
    assert caveats == []


def test_correction_empty_result_returns_original_with_caveat() -> None:
    """Empty process_entries result returns original energy with caveat."""
    structure = _simple_structure("Fe")
    ml_energy = -4.0

    with patch(
        "agentic_discovery_workbench.materials.energy_correction._COMPAT"
    ) as mock_compat:
        mock_compat.process_entries.return_value = []
        energy, caveats = apply_mp2020_correction(structure, ml_energy)

    assert energy == ml_energy
    assert UNCORRECTED_CAVEAT in caveats


def test_correction_exception_returns_original_with_caveat() -> None:
    """Exception during correction returns original energy with caveat."""
    structure = _simple_structure("Fe")
    ml_energy = -4.0

    with patch(
        "agentic_discovery_workbench.materials.energy_correction._COMPAT"
    ) as mock_compat:
        mock_compat.process_entries.side_effect = RuntimeError("no data")
        energy, caveats = apply_mp2020_correction(structure, ml_energy)

    assert energy == ml_energy
    assert UNCORRECTED_CAVEAT in caveats


def test_correction_multi_atom_computes_total_correctly() -> None:
    """Total energy accounts for all sites, not just per-atom value."""
    lattice = Lattice.cubic(3.0)
    structure = Structure(
        lattice, ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    ml_energy_per_atom = -3.0
    # Corrected entry: total -7.0 for 2 atoms = -3.5 per atom
    corrected_entry = ComputedEntry(structure.composition, -7.0)

    with patch(
        "agentic_discovery_workbench.materials.energy_correction._COMPAT"
    ) as mock_compat:
        mock_compat.process_entries.return_value = [corrected_entry]
        energy, caveats = apply_mp2020_correction(structure, ml_energy_per_atom)

    assert energy == pytest.approx(-3.5)
    assert caveats == []


def test_correction_passes_correct_total_energy_to_compat() -> None:
    """Raw entry total energy = ml_energy_per_atom * num_sites."""
    structure = _simple_structure("Fe")
    ml_energy = -4.0

    with patch(
        "agentic_discovery_workbench.materials.energy_correction._COMPAT"
    ) as mock_compat:
        mock_compat.process_entries.return_value = []
        apply_mp2020_correction(structure, ml_energy)

    call_args = mock_compat.process_entries.call_args
    entries_passed = call_args[0][0]
    assert len(entries_passed) == 1
    assert entries_passed[0].energy == pytest.approx(-4.0)


# --- build_candidate_entry ---------------------------------------------------


def test_build_entry_total_energy() -> None:
    """Entry total energy = corrected_per_atom * num_sites."""
    structure = _simple_structure("Fe")
    entry = build_candidate_entry(structure, -4.2)
    assert entry.energy == pytest.approx(-4.2 * structure.num_sites)


def test_build_entry_composition_matches_structure() -> None:
    """Entry composition matches the input structure."""
    structure = _simple_structure("Li")
    entry = build_candidate_entry(structure, -1.5)
    assert entry.composition == structure.composition


def test_build_entry_multi_atom_structure() -> None:
    """Entry handles multi-atom structures with correct total energy."""
    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        ["Li", "Fe", "P", "O"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
    )
    entry = build_candidate_entry(structure, -5.0)
    assert entry.energy == pytest.approx(-20.0)
    assert entry.energy_per_atom == pytest.approx(-5.0)
