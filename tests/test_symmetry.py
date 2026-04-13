"""Tests for symmetry handling: SG range mapping, P1 policy, spglib verification, tolerance sensitivity.

Tests target the symmetry module (ammd.materials.symmetry) and use real spglib
through pymatgen's SpacegroupAnalyzer — no mocking.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ammd.materials.symmetry import (
    CRYSTAL_SYSTEM_SG_RANGES,
    crystal_system_to_sg_range,
    enforce_p1_policy,
)


# ---------------------------------------------------------------------------
# SG range mapping
# ---------------------------------------------------------------------------


def test_sg_range_mapping_completeness():
    """All 230 space groups must be covered with no gaps or overlaps."""
    all_sgs: set[int] = set()
    sorted_ranges = sorted(CRYSTAL_SYSTEM_SG_RANGES.values())

    for lo, hi in sorted_ranges:
        for sg in range(lo, hi + 1):
            assert sg not in all_sgs, f"SG {sg} assigned to multiple systems"
            all_sgs.add(sg)

    assert all_sgs == set(range(1, 231)), (
        f"Missing SGs: {set(range(1, 231)) - all_sgs}, "
        f"Extra SGs: {all_sgs - set(range(1, 231))}"
    )

    # Verify contiguity — no gap between consecutive ranges.
    for i in range(len(sorted_ranges) - 1):
        _, hi = sorted_ranges[i]
        lo_next, _ = sorted_ranges[i + 1]
        assert lo_next == hi + 1, f"Gap between SG {hi} and {lo_next}"


def test_sg_range_mapping_known_systems():
    """Crystal system names map to textbook SG number ranges."""
    # ITA Vol. A reference ranges
    expected = {
        "triclinic": (1, 2),
        "monoclinic": (3, 15),
        "orthorhombic": (16, 74),
        "tetragonal": (75, 142),
        "trigonal": (143, 167),
        "hexagonal": (168, 194),
        "cubic": (195, 230),
    }
    for system, expected_range in expected.items():
        actual = crystal_system_to_sg_range(system)
        assert actual == expected_range, (
            f"{system}: expected {expected_range}, got {actual}"
        )


# ---------------------------------------------------------------------------
# P1 policy enforcement
# ---------------------------------------------------------------------------


def test_p1_policy_reject_when_disallowed():
    """P1 structure rejected when allow_P1=false, accepted when true.

    Also verifies non-P1 structures pass regardless.
    """
    # P1 (SG 1) blocked when disallowed
    ok, reason = enforce_p1_policy(1, allow_p1=False)
    assert ok is False
    assert "P1" in reason

    # P1 (SG 1) permitted when allowed
    ok, reason = enforce_p1_policy(1, allow_p1=True)
    assert ok is True
    assert reason == ""

    # Non-P1 passes regardless of the flag
    ok_disallowed, _ = enforce_p1_policy(225, allow_p1=False)
    ok_allowed, _ = enforce_p1_policy(225, allow_p1=True)
    assert ok_disallowed is True
    assert ok_allowed is True


# ---------------------------------------------------------------------------
# spglib verification — known crystals (3x test weight, riskiest logic)
# ---------------------------------------------------------------------------


def _nacl_fm3m() -> Structure:
    """NaCl conventional cell: Fm-3m (SG 225), a = 5.64 Å, 8 atoms.

    The rock-salt structure has Na on the FCC lattice and Cl on the
    octahedral interstitial sites (or vice versa).
    """
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


def _diamond_si() -> Structure:
    """Diamond Si conventional cell: Fd-3m (SG 227), a = 5.431 Å, 8 atoms.

    Diamond cubic has two interpenetrating FCC sublattices offset by (1/4,1/4,1/4).
    """
    lattice = Lattice.cubic(5.431)
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


def test_spglib_verify_nacl_fm3m():
    """NaCl conventional cell must be detected as Fm-3m (SG 225) by spglib."""
    structure = _nacl_fm3m()
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    sg_number = analyzer.get_space_group_number()
    sg_symbol = analyzer.get_space_group_symbol()

    assert sg_number == 225, f"Expected SG 225 (Fm-3m), got {sg_number} ({sg_symbol})"
    assert "Fm" in sg_symbol, f"Symbol should contain 'Fm', got {sg_symbol!r}"


def test_spglib_verify_diamond_si():
    """Diamond Si conventional cell must be detected as Fd-3m (SG 227) by spglib."""
    structure = _diamond_si()
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    sg_number = analyzer.get_space_group_number()
    sg_symbol = analyzer.get_space_group_symbol()

    assert sg_number == 227, f"Expected SG 227 (Fd-3m), got {sg_number} ({sg_symbol})"
    assert "Fd" in sg_symbol, f"Symbol should contain 'Fd', got {sg_symbol!r}"


# ---------------------------------------------------------------------------
# Tolerance sensitivity (3x test weight, riskiest logic)
# ---------------------------------------------------------------------------


def test_tolerance_sensitivity_sg_change():
    """Tightening symprec on a near-boundary structure must change the detected SG.

    A slightly distorted BCC-like cell with two different species and
    off-ideal positions: loose tolerance detects higher symmetry,
    tight tolerance falls to P1.
    """
    # Lattice angles slightly off cubic, positions slightly off high-symmetry sites.
    # Two different species prevent accidental high symmetry from composition alone.
    lattice = Lattice([[4.0, 0.0, 0.0], [0.0, 4.01, 0.0], [0.0, 0.0, 4.02]])
    species = ["Si", "Ge"]
    coords = [[0.0, 0.0, 0.0], [0.501, 0.502, 0.503]]
    structure = Structure(lattice, species, coords)

    tight = SpacegroupAnalyzer(structure, symprec=1e-5)
    loose = SpacegroupAnalyzer(structure, symprec=0.1)

    sg_tight = tight.get_space_group_number()
    sg_loose = loose.get_space_group_number()

    # Tight tolerance should find P1 (SG 1) — the distortions break symmetry.
    assert sg_tight == 1, (
        f"Tight symprec should yield P1 (SG 1), got SG {sg_tight}"
    )

    # Loose tolerance should find higher symmetry.
    assert sg_loose > 1, (
        f"Loose symprec should detect symmetry above P1, got SG {sg_loose}"
    )

    # The key assertion: different tolerances yield different space groups.
    assert sg_tight != sg_loose, (
        f"Expected different SGs at different tolerances, both gave SG {sg_tight}"
    )
