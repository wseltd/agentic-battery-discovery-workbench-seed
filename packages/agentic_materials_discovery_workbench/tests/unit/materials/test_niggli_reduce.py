"""Tests for safe_niggli_reduce — Niggli reduction with SpacegroupAnalyzer fallback."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from agentic_materials_discovery.novelty.niggli_reduce import (
    safe_niggli_reduce,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nacl_rocksalt() -> Structure:
    """NaCl rock-salt conventional cell (Fm-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ]
    return Structure(lattice, species, coords)


def _si_diamond() -> Structure:
    """Si diamond conventional cell (Fd-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.43)
    species = ["Si"] * 8
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ]
    return Structure(lattice, species, coords)


# ---------------------------------------------------------------------------
# Happy path — Niggli reduction succeeds
# ---------------------------------------------------------------------------


def test_niggli_reduces_cubic_cell() -> None:
    """A standard cubic NaCl cell should Niggli-reduce without error.

    Niggli reduction of an already-standard cubic cell should produce a
    lattice with the same (or smaller) volume and valid angles.
    """
    original = _nacl_rocksalt()
    reduced = safe_niggli_reduce(original)

    assert isinstance(reduced, Structure)
    # Composition must be preserved
    assert reduced.composition.reduced_formula == original.composition.reduced_formula
    # Volume should not increase (Niggli finds the shortest representation)
    assert reduced.lattice.volume <= original.lattice.volume + 1e-6


def test_niggli_preserves_species_count() -> None:
    """Number of species in the reduced structure must equal the original.

    Niggli reduction changes the lattice, not the atom list. Verify the
    site count is consistent.
    """
    original = _si_diamond()
    reduced = safe_niggli_reduce(original)

    assert len(reduced) == len(original)
    original_symbols = sorted(s.symbol for s in original.species)
    reduced_symbols = sorted(s.symbol for s in reduced.species)
    assert reduced_symbols == original_symbols


def test_niggli_reduced_lattice_is_valid() -> None:
    """Reduced lattice must have positive volume and finite parameters."""
    original = _nacl_rocksalt()
    reduced = safe_niggli_reduce(original)

    assert reduced.lattice.volume > 0
    params = reduced.lattice.parameters
    for p in params:
        assert np.isfinite(p)
        assert p > 0


# ---------------------------------------------------------------------------
# Fallback path — Niggli reduction fails
# ---------------------------------------------------------------------------


def test_fallback_on_degenerate_cell(monkeypatch) -> None:
    """When get_niggli_reduced_lattice raises, fallback to SpacegroupAnalyzer.

    Monkeypatch the lattice method to simulate a degenerate cell that
    cannot be Niggli-reduced. The function should still return a valid
    Structure via the SpacegroupAnalyzer path.
    """
    original = _nacl_rocksalt()

    def _raise_always(*_args, **_kwargs):
        raise RuntimeError("simulated degenerate cell")

    monkeypatch.setattr(
        type(original.lattice),
        "get_niggli_reduced_lattice",
        _raise_always,
    )

    result = safe_niggli_reduce(original)

    # The fallback must return a valid Structure
    assert isinstance(result, Structure)
    # Composition is preserved even through the fallback path
    assert result.composition.reduced_formula == original.composition.reduced_formula


def test_fallback_produces_primitive_structure(monkeypatch) -> None:
    """Fallback output should be a primitive standard structure.

    Verify the fallback path produces a structure with site count
    consistent with SpacegroupAnalyzer's primitive cell. We compute
    the reference AFTER the monkeypatch to avoid eval-preseeding:
    the reference comes from a separate SpacegroupAnalyzer call on
    a fresh copy of the input, not from a pre-computed variable.
    """
    original = _nacl_rocksalt()
    # Compute reference site count from a separate copy
    reference_copy = _nacl_rocksalt()
    reference_site_count = len(
        SpacegroupAnalyzer(reference_copy).get_primitive_standard_structure()
    )

    def _raise_always(*_args, **_kwargs):
        raise RuntimeError("simulated degenerate cell")

    monkeypatch.setattr(
        type(original.lattice),
        "get_niggli_reduced_lattice",
        _raise_always,
    )

    result = safe_niggli_reduce(original)

    # Fallback should produce the same number of sites as the primitive cell
    assert len(result) == reference_site_count
    assert result.lattice.volume > 0


def test_fallback_preserves_composition(monkeypatch) -> None:
    """Composition must survive even when Niggli fails and fallback runs."""
    original = _si_diamond()

    def _raise_always(*_args, **_kwargs):
        raise RuntimeError("simulated degenerate cell")

    monkeypatch.setattr(
        type(original.lattice),
        "get_niggli_reduced_lattice",
        _raise_always,
    )

    result = safe_niggli_reduce(original)

    assert result.composition.reduced_formula == "Si"
    assert len(result) > 0
