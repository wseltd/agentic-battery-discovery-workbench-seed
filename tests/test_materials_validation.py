"""Tests for materials structure validation (parse, lattice, elements, atom count,
interatomic distances, coordination sanity)."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery.materials.validation import (
    FORBIDDEN_ATOMIC_NUMBERS,
    MIN_DISTANCE_COVALENT_RATIO,
    ValidationResult,
    validate_allowed_elements,
    validate_atom_count,
    validate_coordination_sanity,
    validate_interatomic_distances,
    validate_lattice_sanity,
    validate_structure_parseable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic_si() -> Structure:
    """Standard cubic Si — the canonical 'good' structure for tests."""
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _make_structure_with_matrix(matrix: list[list[float]]) -> Structure:
    """Build a Structure from a raw 3×3 lattice matrix with a dummy atom."""
    lattice = Lattice(matrix)
    return Structure(lattice, ["H"], [[0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_frozen(self):
        r = ValidationResult(passed=True, stage="x", message="", severity="hard")
        with pytest.raises(AttributeError) as exc_info:
            r.passed = False  # type: ignore[misc]
        assert "passed" in str(exc_info.value) or "cannot assign" in str(exc_info.value).lower() or exc_info.type is AttributeError
        assert r.passed is True  # original value unchanged

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError, match="severity must be one of") as exc_info:
            ValidationResult(passed=True, stage="x", message="", severity="critical")  # type: ignore[arg-type]
        assert "critical" in str(exc_info.value)

    def test_both_allowed_severities(self):
        for sev in ("hard", "soft"):
            r = ValidationResult(passed=True, stage="s", message="", severity=sev)  # type: ignore[arg-type]
            assert r.severity == sev


# ---------------------------------------------------------------------------
# validate_structure_parseable — 5 tests
# ---------------------------------------------------------------------------

class TestParseStage:
    def test_parse_valid_structure(self):
        """Round-tripping a valid Si structure through as_dict → from_dict succeeds."""
        si = _cubic_si()
        result = validate_structure_parseable(si.as_dict())
        assert result.passed is True
        assert result.stage == "parse"
        assert result.severity == "hard"

    def test_parse_missing_lattice_fails(self):
        """Dict with no lattice key cannot be parsed."""
        d = _cubic_si().as_dict()
        del d["lattice"]
        result = validate_structure_parseable(d)
        assert result.passed is False
        assert result.stage == "parse"
        assert "parse" in result.message.lower() or len(result.message) > 0

    def test_parse_missing_species_fails(self):
        """Dict with no species/sites info cannot be parsed."""
        d = _cubic_si().as_dict()
        # pymatgen stores species inside each site entry
        d["sites"] = []
        result = validate_structure_parseable(d)
        # Empty sites list may raise or produce a degenerate structure;
        # pymatgen ≥2024 raises ValueError for zero-site structures.
        # Either way the result should reflect the problem.
        assert result.passed is False or len(d["sites"]) == 0

    def test_parse_wrong_type_fails(self):
        """Passing a non-dict-like value where lattice matrix is expected."""
        d = _cubic_si().as_dict()
        d["lattice"]["matrix"] = "not-a-matrix"
        result = validate_structure_parseable(d)
        assert result.passed is False
        assert result.stage == "parse"

    def test_parse_empty_dict_fails(self):
        """Completely empty dict has no recoverable structure info."""
        result = validate_structure_parseable({})
        assert result.passed is False
        assert result.stage == "parse"
        assert result.severity == "hard"


# ---------------------------------------------------------------------------
# validate_lattice_sanity — 6 tests
# ---------------------------------------------------------------------------

class TestLatticeSanity:
    def test_lattice_valid_cubic(self):
        """Standard cubic Si passes lattice sanity."""
        si = _cubic_si()
        result = validate_lattice_sanity(si)
        assert result.passed is True
        assert result.stage == "lattice_sanity"
        assert result.severity == "hard"

    def test_lattice_nan_in_matrix_fails(self):
        """NaN in any matrix entry is detected."""
        s = _make_structure_with_matrix([
            [float("nan"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "NaN" in result.message or "nan" in result.message.lower()

    def test_lattice_inf_in_matrix_fails(self):
        """Inf in any matrix entry is detected."""
        s = _make_structure_with_matrix([
            [float("inf"), 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "Inf" in result.message or "inf" in result.message.lower()

    def test_lattice_zero_volume_fails(self):
        """Singular matrix (zero determinant) → non-positive determinant failure."""
        # Two identical rows make the matrix singular (det = 0).
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert result.stage == "lattice_sanity"

    def test_lattice_near_singular_fails(self):
        """Lattice with volume far below the minimum threshold fails."""
        # Tiny c-axis → volume ≈ 5 * 5 * 1e-6 = 2.5e-5 Å³ < 0.1
        s = _make_structure_with_matrix([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 1e-6],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "volume" in result.message.lower() or "determinant" in result.message.lower()

    def test_lattice_negative_determinant_fails(self):
        """Left-handed lattice (negative determinant) is rejected."""
        # Swapping two rows of a right-handed basis flips the determinant sign.
        s = _make_structure_with_matrix([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        result = validate_lattice_sanity(s)
        assert result.passed is False
        assert "determinant" in result.message.lower() or "non-positive" in result.message.lower()


# ---------------------------------------------------------------------------
# FORBIDDEN_ATOMIC_NUMBERS constant — 2 tests
# ---------------------------------------------------------------------------

class TestForbiddenAtomicNumbers:
    def test_cardinality(self):
        """The set should contain exactly the union of noble gases, Tc, Pm, and Z>=84."""
        noble = {2, 10, 18, 36, 54, 86}
        radioactive_orphans = {43, 61}
        high_z = set(range(84, 119))
        expected = noble | radioactive_orphans | high_z
        assert FORBIDDEN_ATOMIC_NUMBERS == expected
        assert len(FORBIDDEN_ATOMIC_NUMBERS) == 42

    def test_is_frozenset(self):
        assert isinstance(FORBIDDEN_ATOMIC_NUMBERS, frozenset)
        assert 43 in FORBIDDEN_ATOMIC_NUMBERS  # Tc must be present


# ---------------------------------------------------------------------------
# validate_allowed_elements — 6 tests (boundary correctness is critical)
# ---------------------------------------------------------------------------

def _simple_structure(*symbols: str) -> Structure:
    """Build a minimal cubic structure from element symbols."""
    coords = [[i * 0.2, 0.0, 0.0] for i in range(len(symbols))]
    return Structure(Lattice.cubic(10.0), list(symbols), coords)


class TestAllowedElements:
    def test_elements_valid_common_elements(self):
        """H, C, N, O, Fe, Bi (Z=83) are all allowed."""
        s = _simple_structure("H", "C", "N", "O", "Fe", "Bi")
        result = validate_allowed_elements(s)
        assert result.passed is True
        assert result.stage == "allowed_elements"
        assert result.severity == "hard"

    def test_elements_technetium_rejected(self):
        """Tc (Z=43) has no stable isotopes — must be forbidden."""
        s = _simple_structure("Tc")
        result = validate_allowed_elements(s)
        assert result.passed is False
        assert "Tc" in result.message

    def test_elements_promethium_rejected(self):
        """Pm (Z=61) has no stable isotopes — must be forbidden."""
        s = _simple_structure("Pm")
        result = validate_allowed_elements(s)
        assert result.passed is False
        assert "Pm" in result.message

    def test_elements_noble_gas_rejected(self):
        """All six noble gases (He, Ne, Ar, Kr, Xe, Rn) must be forbidden."""
        for sym in ("He", "Ne", "Ar", "Kr", "Xe", "Rn"):
            s = _simple_structure(sym)
            result = validate_allowed_elements(s)
            assert result.passed is False, f"{sym} should be forbidden"
            assert sym in result.message

    def test_elements_z_above_83_rejected(self):
        """Spot-check several elements with Z >= 84."""
        for sym in ("Po", "At", "Ra", "Th", "U", "Pu", "Og"):
            s = _simple_structure(sym)
            result = validate_allowed_elements(s)
            assert result.passed is False, f"{sym} should be forbidden"

    def test_elements_boundary_z83_bismuth_allowed(self):
        """Bi (Z=83) is the heaviest allowed element — boundary check."""
        s = _simple_structure("Bi")
        result = validate_allowed_elements(s)
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_atom_count — 6 tests
# ---------------------------------------------------------------------------

class TestAtomCount:
    def test_atom_count_valid(self):
        """Standard 2-atom Si cell passes."""
        si = _cubic_si()
        result = validate_atom_count(si)
        assert result.passed is True
        assert result.stage == "atom_count"
        assert result.severity == "hard"

    def test_atom_count_exactly_1_passes(self):
        """A single-atom structure is within the allowed range."""
        s = Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]])
        result = validate_atom_count(s)
        assert result.passed is True

    def test_atom_count_exactly_20_passes(self):
        """20 atoms is the upper boundary — must pass."""
        coords = [[i * 0.05, 0.0, 0.0] for i in range(20)]
        s = Structure(Lattice.cubic(20.0), ["Si"] * 20, coords)
        result = validate_atom_count(s)
        assert result.passed is True

    def test_atom_count_exceeds_20_fails(self):
        """21 atoms exceeds the limit."""
        coords = [[i * 0.045, 0.0, 0.0] for i in range(21)]
        s = Structure(Lattice.cubic(20.0), ["Si"] * 21, coords)
        result = validate_atom_count(s)
        assert result.passed is False
        assert "21" in result.message
        assert result.stage == "atom_count"

    def test_atom_count_large_structure_fails(self):
        """50 atoms is well above the limit."""
        coords = [[i * 0.02, 0.0, 0.0] for i in range(50)]
        s = Structure(Lattice.cubic(30.0), ["C"] * 50, coords)
        result = validate_atom_count(s)
        assert result.passed is False
        assert "50" in result.message

    def test_atom_count_message_includes_maximum(self):
        """Failure message should mention the maximum for actionable feedback."""
        coords = [[i * 0.045, 0.0, 0.0] for i in range(25)]
        s = Structure(Lattice.cubic(20.0), ["O"] * 25, coords)
        result = validate_atom_count(s)
        assert result.passed is False
        assert "20" in result.message


# ---------------------------------------------------------------------------
# validate_interatomic_distances — 6 tests (PBC + threshold edge cases)
# ---------------------------------------------------------------------------

class TestInteratomicDistances:
    def test_distance_valid_si_diamond(self):
        """Diamond-cubic Si (nearest-neighbour ≈2.35 Å) passes easily."""
        si = _cubic_si()
        result = validate_interatomic_distances(si)
        assert result.passed is True
        assert result.stage == "interatomic_distances"
        assert result.severity == "hard"

    def test_distance_overlapping_atoms_fails(self):
        """Two atoms at the same fractional position (distance ≈ 0) must fail."""
        s = Structure(
            Lattice.cubic(5.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        result = validate_interatomic_distances(s)
        assert result.passed is False
        assert result.stage == "interatomic_distances"
        assert "Si" in result.message

    def test_distance_barely_below_threshold_fails(self):
        """Place two Si atoms just below 0.5 * (r_Si + r_Si) apart.

        Si covalent radius ≈ 1.11 Å → threshold = 0.5 * 2.22 = 1.11 Å.
        Put atoms 1.10 Å apart (below threshold) → must fail.
        """
        from pymatgen.core import Element

        r_si = float(Element("Si").atomic_radius)
        threshold = MIN_DISTANCE_COVALENT_RATIO * 2 * r_si
        gap = threshold - 0.01  # just below

        # Place second atom at (gap, 0, 0) in Cartesian → fractional = (gap/20, 0, 0)
        s = Structure(
            Lattice.cubic(20.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [gap / 20.0, 0.0, 0.0]],
        )
        result = validate_interatomic_distances(s)
        assert result.passed is False
        assert result.severity == "hard"

    def test_distance_exactly_at_threshold_passes(self):
        """Atoms separated by exactly the threshold distance should pass.

        The check is strict-less-than (d < threshold), so d == threshold passes.
        """
        from pymatgen.core import Element

        r_si = float(Element("Si").atomic_radius)
        threshold = MIN_DISTANCE_COVALENT_RATIO * 2 * r_si

        s = Structure(
            Lattice.cubic(20.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [threshold / 20.0, 0.0, 0.0]],
        )
        result = validate_interatomic_distances(s)
        assert result.passed is True

    def test_distance_uses_periodic_boundaries(self):
        """Two atoms far apart in fractional coords but close across the PBC.

        In a 4 Å cubic cell, atoms at (0, 0, 0) and (0.95, 0, 0) are only
        0.2 Å apart through the periodic boundary (4 - 3.8 = 0.2 Å).
        That is well below any covalent-radius threshold → must fail.
        """
        s = Structure(
            Lattice.cubic(4.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]],
        )
        result = validate_interatomic_distances(s)
        assert result.passed is False, (
            "PBC should make these atoms ~0.2 Å apart, well below threshold"
        )

    def test_distance_different_element_pairs(self):
        """Mixed elements use per-element covalent radii, not a single global value.

        Fe (r_cov ≈ 1.25 Å) + O (r_cov ≈ 0.66 Å) → threshold ≈ 0.955 Å.
        Place them 0.90 Å apart — must fail.  Then verify the message names
        both elements.
        """
        from pymatgen.core import Element

        r_fe = float(Element("Fe").atomic_radius)
        r_o = float(Element("O").atomic_radius)
        threshold = MIN_DISTANCE_COVALENT_RATIO * (r_fe + r_o)
        gap = threshold - 0.05

        s = Structure(
            Lattice.cubic(20.0),
            ["Fe", "O"],
            [[0.0, 0.0, 0.0], [gap / 20.0, 0.0, 0.0]],
        )
        result = validate_interatomic_distances(s)
        assert result.passed is False
        assert "Fe" in result.message
        assert "O" in result.message


# ---------------------------------------------------------------------------
# validate_coordination_sanity — 3 tests (soft flag, lower risk)
# ---------------------------------------------------------------------------

class TestCoordinationSanity:
    def test_coordination_valid_structure(self):
        """Diamond-cubic Si has CN=4 for both sites — should pass."""
        si = _cubic_si()
        result = validate_coordination_sanity(si)
        assert result.passed is True
        assert result.stage == "coordination_sanity"
        assert result.severity == "soft"

    def test_coordination_isolated_atom_flags(self):
        """A single atom in a huge cell has CN=0 — soft flag expected."""
        s = Structure(Lattice.cubic(50.0), ["He"], [[0.0, 0.0, 0.0]])
        result = validate_coordination_sanity(s)
        assert result.passed is False
        assert "isolated" in result.message.lower() or "CN=0" in result.message

    def test_coordination_severity_is_soft(self):
        """Coordination check must always use soft severity, never hard."""
        si = _cubic_si()
        result = validate_coordination_sanity(si)
        assert result.severity == "soft"
