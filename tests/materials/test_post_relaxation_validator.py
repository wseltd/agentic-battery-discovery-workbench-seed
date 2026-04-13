"""Tests for post-relaxation validation of crystal structures."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pymatgen.core import Lattice, Structure

from pymatgen.core import Element

from agentic_discovery_workbench.materials.post_relaxation_validator import (
    POST_RELAX_COVALENT_RATIO,
    PostRelaxationReport,
    validate_post_relaxation,
)


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------


def _nacl_rocksalt() -> Structure:
    """NaCl rock-salt conventional cell (Fm-3m, SG 225, 8 atoms)."""
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
    """Si diamond conventional cell (Fd-3m, SG 227, 8 atoms)."""
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


def _two_atom_structure(distance_angstrom: float) -> Structure:
    """Two Si atoms at a controlled interatomic distance.

    Uses a large cubic cell so the direct distance is always shorter
    than any periodic image.

    Args:
        distance_angstrom: Desired minimum interatomic distance.
    """
    a = max(distance_angstrom * 4, 10.0)
    lattice = Lattice.cubic(a)
    frac = distance_angstrom / a
    return Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [frac, 0.0, 0.0]])


def _perturbed_nacl() -> Structure:
    """NaCl with one atom displaced enough to break Fm-3m symmetry.

    Displacement of ~0.85 Å is well above any reasonable symprec,
    guaranteeing a different spacegroup than the ideal structure.
    """
    s = _nacl_rocksalt()
    s.translate_sites([0], [0.15, 0.05, 0.03], frac_coords=True)
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_relaxed_structure_passes() -> None:
    """A well-formed structure with no duplicates should pass all checks."""
    nacl = _nacl_rocksalt()
    reports = validate_post_relaxation([("nacl_1", nacl, nacl)])

    assert len(reports) == 1
    r = reports[0]
    assert r.structure_id == "nacl_1"
    assert r.distance_valid is True
    assert r.is_duplicate is False
    assert r.duplicate_of is None
    assert r.symmetry_changed is False
    assert r.passed is True


def test_short_distance_rejects() -> None:
    """Structures with atoms closer than the threshold must fail."""
    pre = _nacl_rocksalt()
    # 0.3 Å is well below the 0.5 Å default threshold
    short = _two_atom_structure(0.3)

    reports = validate_post_relaxation([("short", pre, short)])

    assert len(reports) == 1
    r = reports[0]
    assert r.distance_valid is False
    assert r.min_distance_angstrom < 0.5
    assert r.min_distance_angstrom == pytest.approx(0.3, abs=0.05)
    assert r.passed is False


def test_duplicate_relaxed_structure_rejects() -> None:
    """The second of two identical structures must be flagged as duplicate."""
    nacl = _nacl_rocksalt()
    reports = validate_post_relaxation(
        [("orig", nacl, nacl), ("copy", nacl, nacl)]
    )

    assert len(reports) == 2
    assert reports[0].is_duplicate is False
    assert reports[0].passed is True

    assert reports[1].is_duplicate is True
    assert reports[1].duplicate_of == "orig"
    assert reports[1].passed is False


def test_symmetry_change_recorded_not_rejected() -> None:
    """Symmetry change is informational — it alone must not cause rejection."""
    pre = _nacl_rocksalt()
    post = _perturbed_nacl()

    reports = validate_post_relaxation([("s1", pre, post)])

    r = reports[0]
    assert r.symmetry_changed is True
    assert r.pre_relax_spacegroup != r.post_relax_spacegroup
    # Symmetry change does not reject — only distance and duplicate do
    assert r.distance_valid is True
    assert r.is_duplicate is False
    assert r.passed is True


def test_symmetry_preserved_after_relaxation() -> None:
    """When the same structure is pre and post, symmetry must be unchanged."""
    nacl = _nacl_rocksalt()
    reports = validate_post_relaxation([("nacl", nacl, nacl)])

    r = reports[0]
    assert r.pre_relax_spacegroup == r.post_relax_spacegroup
    assert r.symmetry_changed is False


def test_report_fields_populated() -> None:
    """Every field in PostRelaxationReport must have the correct type."""
    nacl = _nacl_rocksalt()
    reports = validate_post_relaxation([("nacl", nacl, nacl)])
    r = reports[0]

    assert isinstance(r.structure_id, str)
    assert isinstance(r.distance_valid, bool)
    assert isinstance(r.min_distance_angstrom, float)
    assert isinstance(r.pre_relax_spacegroup, int)
    assert isinstance(r.post_relax_spacegroup, int)
    assert isinstance(r.symmetry_changed, bool)
    assert isinstance(r.is_duplicate, bool)
    assert r.duplicate_of is None or isinstance(r.duplicate_of, str)
    assert isinstance(r.passed, bool)

    # Numeric fields carry meaningful values
    assert r.min_distance_angstrom > 0
    assert 1 <= r.pre_relax_spacegroup <= 230
    assert 1 <= r.post_relax_spacegroup <= 230


def test_passed_flag_logic() -> None:
    """The __post_init__ invariant: passed = distance_valid AND NOT is_duplicate."""
    # Valid: distance_valid=True, is_duplicate=False → passed=True
    r_a = PostRelaxationReport(
        structure_id="a", distance_valid=True, min_distance_angstrom=2.0,
        pre_relax_spacegroup=225, post_relax_spacegroup=225,
        symmetry_changed=False, is_duplicate=False, duplicate_of=None,
        passed=True,
    )
    assert r_a.passed is True

    # Valid: distance_valid=False, is_duplicate=False → passed=False
    r_b = PostRelaxationReport(
        structure_id="b", distance_valid=False, min_distance_angstrom=0.1,
        pre_relax_spacegroup=225, post_relax_spacegroup=225,
        symmetry_changed=False, is_duplicate=False, duplicate_of=None,
        passed=False,
    )
    assert r_b.passed is False

    # Valid: distance_valid=True, is_duplicate=True → passed=False
    r_c = PostRelaxationReport(
        structure_id="c", distance_valid=True, min_distance_angstrom=2.0,
        pre_relax_spacegroup=225, post_relax_spacegroup=225,
        symmetry_changed=False, is_duplicate=True, duplicate_of="x",
        passed=False,
    )
    assert r_c.passed is False

    # Valid: distance_valid=False, is_duplicate=True → passed=False
    r_d = PostRelaxationReport(
        structure_id="d", distance_valid=False, min_distance_angstrom=0.1,
        pre_relax_spacegroup=225, post_relax_spacegroup=225,
        symmetry_changed=False, is_duplicate=True, duplicate_of="x",
        passed=False,
    )
    assert r_d.passed is False

    # Invalid: passed=True when distance_valid=False — __post_init__ rejects
    with pytest.raises(ValueError, match="passed must equal"):
        PostRelaxationReport(
            structure_id="e", distance_valid=False, min_distance_angstrom=0.1,
            pre_relax_spacegroup=225, post_relax_spacegroup=225,
            symmetry_changed=False, is_duplicate=False, duplicate_of=None,
            passed=True,
        )

    # Invalid: passed=True when is_duplicate=True — __post_init__ rejects
    with pytest.raises(ValueError, match="passed must equal"):
        PostRelaxationReport(
            structure_id="f", distance_valid=True, min_distance_angstrom=2.0,
            pre_relax_spacegroup=225, post_relax_spacegroup=225,
            symmetry_changed=False, is_duplicate=True, duplicate_of="x",
            passed=True,
        )


def test_consistent_symprec_between_phases() -> None:
    """The same symprec must be forwarded to both pre and post spacegroup analysis."""
    nacl = _nacl_rocksalt()
    custom_symprec = 0.042

    with patch(
        "agentic_discovery_workbench.materials.post_relaxation_validator"
        "._get_spacegroup_number",
        return_value=225,
    ) as mock_fn:
        validate_post_relaxation(
            [("s1", nacl, nacl)], symprec=custom_symprec
        )

    # Exactly two calls: one for pre-relax, one for post-relax
    assert mock_fn.call_count == 2
    for call_obj in mock_fn.call_args_list:
        # _get_spacegroup_number(structure, symprec) — symprec is second positional arg
        assert call_obj[0][1] == custom_symprec


def test_batch_validation_mixed_results() -> None:
    """A batch with valid, short-distance, and duplicate structures."""
    nacl = _nacl_rocksalt()
    si = _si_diamond()
    short = _two_atom_structure(0.3)

    reports = validate_post_relaxation([
        ("good_nacl", nacl, nacl),
        ("good_si", si, si),
        ("bad_short", nacl, short),
        ("dup_nacl", nacl, nacl),  # duplicate of good_nacl
    ])

    assert len(reports) == 4

    # First NaCl passes
    assert reports[0].passed is True
    assert reports[0].is_duplicate is False
    assert reports[0].distance_valid is True

    # Si passes (different composition, not a duplicate)
    assert reports[1].passed is True
    assert reports[1].is_duplicate is False

    # Short distance fails
    assert reports[2].passed is False
    assert reports[2].distance_valid is False

    # Duplicate NaCl fails
    assert reports[3].passed is False
    assert reports[3].is_duplicate is True
    assert reports[3].duplicate_of == "good_nacl"


def test_distance_threshold_boundary() -> None:
    """Distance at covalent-ratio threshold passes; just below fails."""
    pre = _nacl_rocksalt()

    # Si-Si covalent-ratio threshold:
    # POST_RELAX_COVALENT_RATIO * (r_Si + r_Si)
    si_radius = Element("Si").atomic_radius
    assert si_radius is not None
    cov_threshold = POST_RELAX_COVALENT_RATIO * 2 * float(si_radius)

    at_threshold = _two_atom_structure(cov_threshold)
    below = _two_atom_structure(cov_threshold - 0.01)
    above = _two_atom_structure(cov_threshold + 0.01)

    reports = validate_post_relaxation([
        ("at", pre, at_threshold),
        ("below", pre, below),
        ("above", pre, above),
    ])

    # Exactly at threshold: dist >= threshold → passes
    assert reports[0].distance_valid is True
    assert reports[0].min_distance_angstrom == pytest.approx(
        cov_threshold, abs=0.02
    )

    # Just below: fails
    assert reports[1].distance_valid is False
    assert reports[1].min_distance_angstrom < cov_threshold

    # Just above: passes
    assert reports[2].distance_valid is True
    assert reports[2].min_distance_angstrom > cov_threshold
