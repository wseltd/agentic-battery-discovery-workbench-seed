"""Tests for materials duplicate/novelty detection.

9 tests in three categories:
  - Exact match (3): identical, reordered atoms, origin-shifted
  - Tolerance boundary (3): stol within, stol beyond, angle_tol
  - Novelty classification (3): known in MP, novel (no match), close analogue

Uses hand-crafted pymatgen Structure objects — no file I/O, no network calls.
MP reference lookup is mocked via a ReferenceDBClient subclass that returns
pre-loaded structures keyed by composition.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.novelty_checker import (
    MaterialsNoveltyChecker,
    MaterialsNoveltyClassification,
    ReferenceDBClient,
)
from agentic_discovery_workbench.materials.novelty_matcher import (
    match_against_references,
)


# --- Helpers -----------------------------------------------------------------


def _make_nacl(a: float = 5.64) -> Structure:
    """NaCl primitive cell: Na at origin, Cl at body centre."""
    return Structure(
        Lattice.cubic(a),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


class _FakeMPClient(ReferenceDBClient):
    """Mock MP client returning pre-loaded structures by composition.

    Avoids network calls while exercising the full MaterialsNoveltyChecker
    pipeline including composition lookup, caching, and StructureMatcher.
    """

    def __init__(
        self, structures_by_formula: dict[str, list[tuple[str, Structure]]]
    ) -> None:
        super().__init__(db_name="MP", db_version="mock-2024.01")
        self._structures_by_formula = structures_by_formula

    def _fetch_by_composition(
        self, composition: str
    ) -> list[tuple[str, Structure]]:
        return self._structures_by_formula.get(composition, [])


# --- Exact match tests -------------------------------------------------------
# These verify that match_against_references correctly identifies structures
# that are crystallographically identical despite superficial differences
# (atom ordering, origin choice).


def test_exact_match_identical_structure():
    """Identical NaCl structures match — baseline for the matcher wrapper."""
    nacl = _make_nacl()
    result = match_against_references(nacl, [("ref-1", _make_nacl())])
    assert result == "ref-1"


def test_exact_match_reordered_atoms():
    """NaCl with species listed Cl-first still matches Na-first NaCl."""
    nacl = _make_nacl()
    reordered = Structure(
        Lattice.cubic(5.64),
        ["Cl", "Na"],
        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    )
    result = match_against_references(nacl, [("ref-1", reordered)])
    assert result == "ref-1"


def test_exact_match_origin_shifted():
    """NaCl shifted by [0.25, 0.25, 0.25] in fractional coords still matches.

    StructureMatcher handles origin shifts by enumerating candidate
    translations — this test verifies the wrapper propagates that correctly.
    """
    nacl = _make_nacl()
    shifted = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    result = match_against_references(nacl, [("ref-1", shifted)])
    assert result == "ref-1"


# --- Tolerance boundary tests ------------------------------------------------
# These verify that tolerance parameters (stol, angle_tol) are actually passed
# through to StructureMatcher and control match/no-match outcomes.  We are NOT
# testing StructureMatcher internals — we are testing that our wrapper honours
# the parameters rather than hardcoding defaults.


def test_tolerance_boundary_stol_at_limit():
    """Small site displacement within default stol=0.3 still matches.

    Cl displaced by 0.02 fractional (~0.11 Angstrom in a 5.64 A cell)
    is well within default stol=0.3 tolerance.
    """
    nacl = _make_nacl()
    perturbed = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.52, 0.5, 0.5]],
    )
    result = match_against_references(
        nacl, [("ref-1", perturbed)], stol=0.3
    )
    assert result == "ref-1"


def test_tolerance_boundary_stol_beyond_limit():
    """Same displacement matches at stol=0.3 but NOT at stol=0.001.

    The 0.02 fractional Cl shift from the at-limit test is repeated here.
    Contrasting both tolerances proves stol is actually forwarded to the
    matcher rather than hardcoded.
    """
    nacl = _make_nacl()
    perturbed = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.52, 0.5, 0.5]],
    )
    # Displacement within default stol — should match
    match_result = match_against_references(
        nacl, [("ref-1", perturbed)], stol=0.3
    )
    assert match_result == "ref-1"
    # Same displacement with tight stol — should NOT match
    no_match_result = match_against_references(
        nacl, [("ref-1", perturbed)], stol=0.001
    )
    assert no_match_result is None


def test_tolerance_boundary_angle_tol():
    """4-degree lattice shear matches at angle_tol=5 but not at angle_tol=3.

    Monoclinic cell with beta=94 degrees (4 degrees off cubic 90 degrees).
    Verifies angle_tol parameter controls the lattice-angle boundary.
    """
    nacl = _make_nacl()
    sheared = Structure(
        Lattice.from_parameters(5.64, 5.64, 5.64, 90, 94, 90),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    # 4 degrees < angle_tol 5 degrees — should match
    match_result = match_against_references(
        nacl, [("ref-1", sheared)], angle_tol=5.0
    )
    assert match_result == "ref-1"
    # 4 degrees > angle_tol 3 degrees — should NOT match
    no_match_result = match_against_references(
        nacl, [("ref-1", sheared)], angle_tol=3.0
    )
    assert no_match_result is None


# --- Novelty classification tests --------------------------------------------
# These exercise MaterialsNoveltyChecker.check() end-to-end with a mocked
# ReferenceDBClient.  No network calls — the fake client returns pre-loaded
# structures keyed by reduced composition.


def test_novelty_exact_known_in_mp():
    """Structure matching an MP reference exactly is classified KNOWN."""
    nacl = _make_nacl()
    formula = nacl.composition.reduced_formula
    client = _FakeMPClient({formula: [("mp-22862", _make_nacl())]})
    checker = MaterialsNoveltyChecker(clients=[client])

    result = checker.check("gen-1", nacl)

    assert result.classification == MaterialsNoveltyClassification.KNOWN
    assert result.matched_reference_id == "mp-22862"
    assert result.reference_db == "MP"


def test_novelty_novel_like_no_mp_match():
    """No matching composition in MP database — classified NOVEL."""
    nacl = _make_nacl()
    # Empty database: no structures for any composition
    client = _FakeMPClient({})
    checker = MaterialsNoveltyChecker(clients=[client])

    result = checker.check("gen-1", nacl)

    assert result.classification == MaterialsNoveltyClassification.NOVEL
    assert result.matched_reference_id is None


def test_novelty_close_analogue():
    """Slightly perturbed MP reference within default tolerance — classified KNOWN.

    The reference has lattice param 5.65 vs query 5.64 (0.18% difference,
    within ltol=0.2) and small site displacements.  This is a 'close analogue'
    rather than an exact copy, but StructureMatcher still identifies the match.
    """
    nacl = _make_nacl()
    formula = nacl.composition.reduced_formula
    close = Structure(
        Lattice.cubic(5.65),
        ["Na", "Cl"],
        [[0.01, 0.0, 0.0], [0.5, 0.5, 0.51]],
    )
    client = _FakeMPClient({formula: [("mp-22862", close)]})
    checker = MaterialsNoveltyChecker(clients=[client])

    result = checker.check("gen-1", nacl)

    assert result.classification == MaterialsNoveltyClassification.KNOWN
    assert result.matched_reference_id == "mp-22862"
