"""Tests for composite pre-relaxation structure validation (validate_structure).

9 tests covering three categories:
  - Valid-pass: NaCl rocksalt, LiFePO4-like olivine, BCC iron
  - Reject: technetium (forbidden element), 25 atoms (exceeds cap), neon (noble gas)
  - Flag: short interatomic distance, extreme coordination, 2D layered in 3D cell
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from agentic_materials_discovery.validation.core import (
    WARNING_EXTREME_COORDINATION,
    WARNING_LOW_DIMENSIONALITY,
    WARNING_SHORT_DISTANCE,
    validate_structure,
)


# ---------------------------------------------------------------------------
# Valid-pass tests — is_valid must be True
# ---------------------------------------------------------------------------


def test_valid_nacl_rocksalt_passes():
    """NaCl rocksalt conventional cell: 8 atoms, cubic, common elements."""
    nacl = Structure(
        Lattice.cubic(5.64),
        ["Na"] * 4 + ["Cl"] * 4,
        [
            # Na sublattice (FCC positions)
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            # Cl sublattice (offset FCC)
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ],
    )
    result = validate_structure(nacl)
    assert result.is_valid is True
    assert result.rejection_reason is None


def test_valid_lifepo4_olivine_passes():
    """Olivine-type structure with Li, Fe, P, O in orthorhombic cell."""
    # Simplified olivine-like arrangement — 8 atoms in an orthorhombic cell
    # with realistic lattice parameters.  Positions are spaced to avoid
    # distance violations while keeping a 3D bonded network.
    olivine = Structure(
        Lattice.orthorhombic(10.33, 6.01, 4.69),
        ["Li", "Fe", "P", "O", "O", "O", "O", "Fe"],
        [
            [0.00, 0.00, 0.00],
            [0.28, 0.25, 0.50],
            [0.60, 0.75, 0.00],
            [0.40, 0.25, 0.25],
            [0.10, 0.75, 0.75],
            [0.80, 0.50, 0.25],
            [0.60, 0.00, 0.75],
            [0.72, 0.25, 0.50],
        ],
    )
    result = validate_structure(olivine)
    assert result.is_valid is True
    assert result.rejection_reason is None


def test_valid_bcc_iron_passes():
    """BCC iron: 2 atoms, cubic, single element — minimal valid structure."""
    bcc_fe = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    result = validate_structure(bcc_fe)
    assert result.is_valid is True
    assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# Reject tests — is_valid must be False with an informative rejection_reason
# ---------------------------------------------------------------------------


def test_reject_excluded_element_technetium():
    """Tc (Z=43) has no stable isotopes — must be rejected at the element stage."""
    tc_struct = Structure(
        Lattice.cubic(5.0),
        ["Tc", "Tc"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    result = validate_structure(tc_struct)
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "Tc" in result.rejection_reason


def test_reject_atom_count_exceeds_limit():
    """25 atoms exceeds the 20-atom cap — must be rejected at atom-count stage."""
    n = 25
    # Spread atoms in a large cell so distance checks would not be the
    # cause of failure — atom count is checked first anyway.
    coords = [[i / n, 0.0, 0.0] for i in range(n)]
    big = Structure(Lattice.cubic(50.0), ["Si"] * n, coords)
    result = validate_structure(big)
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "25" in result.rejection_reason or "20" in result.rejection_reason


def test_reject_noble_gas_element():
    """Ne (Z=10) is a noble gas — chemically inert, must be rejected."""
    ne_struct = Structure(
        Lattice.cubic(5.0),
        ["Ne", "Ne"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    result = validate_structure(ne_struct)
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "Ne" in result.rejection_reason


# ---------------------------------------------------------------------------
# Flag tests — is_valid is True but specific warnings are present.
# These are the risky tests: synthetic structures must pass all hard checks
# yet reliably trigger exactly one soft-check category.
# ---------------------------------------------------------------------------


def test_flag_short_interatomic_distance():
    """Two Si atoms 0.4 Å apart — well below 0.5× summed covalent radii.

    Si covalent radius ≈ 1.11 Å → threshold = 0.5 × 2.22 = 1.11 Å.
    0.4 Å < 1.11 Å triggers the distance warning.  Hard checks still
    pass because Si is allowed and 2 atoms ≤ 20.
    """
    # 0.4 Å separation in a 10 Å cubic cell = fractional offset of 0.04
    short_distance = Structure(
        Lattice.cubic(10.0),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]],
    )
    result = validate_structure(short_distance)
    assert result.is_valid is True
    assert WARNING_SHORT_DISTANCE in result.warnings


def test_flag_extreme_coordination_number():
    """Single Fe atom in a 50 Å cell — CrystalNN sees no neighbours (CN=0).

    One atom passes all hard checks (Fe allowed, 1 ≤ 20, lattice fine).
    CrystalNN cannot find any coordinated neighbours at this distance,
    producing a CN=0 anomaly that triggers the coordination warning.
    """
    isolated = Structure(
        Lattice.cubic(50.0),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    result = validate_structure(isolated)
    assert result.is_valid is True
    assert WARNING_EXTREME_COORDINATION in result.warnings


def test_flag_2d_layered_in_3d_cell():
    """Two C atoms in the z=0 plane with a 25 Å c-axis — effectively 2D.

    In-plane bonds form through periodic images of the 3 Å a/b axes,
    but the 25 Å c-axis gap has no bonding across it.  Larsen's method
    classifies this as 2D (layered), triggering the dimensionality warning.
    Hard checks pass: C is allowed, 2 atoms ≤ 20, no distance violation
    (nearest C–C ≈ 2.12 Å, threshold ≈ 0.77 Å).
    """
    layered = Structure(
        Lattice([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 25.0]]),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    result = validate_structure(layered)
    assert result.is_valid is True
    assert WARNING_LOW_DIMENSIONALITY in result.warnings
