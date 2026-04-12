"""Tests for check_allowed_elements and check_atom_count — stage 3 validation.

Exercises element-whitelist boundary correctness (the critical risk surface)
and atom-count edge cases through the stage-function wrappers.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from validation.stages import check_allowed_elements, check_atom_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic_si() -> Structure:
    """Standard cubic Si — canonical 'good' structure."""
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _simple_structure(*symbols: str) -> Structure:
    """Build a minimal cubic structure from element symbols."""
    coords = [[i * 0.2, 0.0, 0.0] for i in range(len(symbols))]
    return Structure(Lattice.cubic(10.0), list(symbols), coords)


# ---------------------------------------------------------------------------
# check_allowed_elements — 6 tests (boundary correctness is critical)
# ---------------------------------------------------------------------------

class TestCheckAllowedElements:
    def test_elements_valid_common_elements(self):
        """H, C, N, O, Fe, Bi (Z=83) are all allowed."""
        s = _simple_structure("H", "C", "N", "O", "Fe", "Bi")
        result = check_allowed_elements(s)
        assert result.passed is True
        assert result.stage == "allowed_elements"
        assert result.severity == "hard"

    def test_elements_technetium_rejected(self):
        """Tc (Z=43) has no stable isotopes — must be forbidden."""
        s = _simple_structure("Tc")
        result = check_allowed_elements(s)
        assert result.passed is False
        assert "Tc" in result.message

    def test_elements_promethium_rejected(self):
        """Pm (Z=61) has no stable isotopes — must be forbidden."""
        s = _simple_structure("Pm")
        result = check_allowed_elements(s)
        assert result.passed is False
        assert "Pm" in result.message

    def test_elements_noble_gas_rejected(self):
        """All six noble gases (He, Ne, Ar, Kr, Xe, Rn) must be forbidden."""
        for sym in ("He", "Ne", "Ar", "Kr", "Xe", "Rn"):
            s = _simple_structure(sym)
            result = check_allowed_elements(s)
            assert result.passed is False, f"{sym} should be forbidden"
            assert sym in result.message

    def test_elements_z_above_83_rejected(self):
        """Spot-check several elements with Z >= 84."""
        for sym in ("Po", "At", "Ra", "Th", "U", "Pu", "Og"):
            s = _simple_structure(sym)
            result = check_allowed_elements(s)
            assert result.passed is False, f"{sym} should be forbidden"

    def test_elements_boundary_z83_bismuth_allowed(self):
        """Bi (Z=83) is the heaviest allowed element — boundary check."""
        s = _simple_structure("Bi")
        result = check_allowed_elements(s)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_atom_count — 6 tests
# ---------------------------------------------------------------------------

class TestCheckAtomCount:
    def test_atom_count_valid(self):
        """Standard 2-atom Si cell passes."""
        si = _cubic_si()
        result = check_atom_count(si)
        assert result.passed is True
        assert result.stage == "atom_count"
        assert result.severity == "hard"

    def test_atom_count_exactly_1_passes(self):
        """A single-atom structure is within the allowed range."""
        s = Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]])
        result = check_atom_count(s)
        assert result.passed is True

    def test_atom_count_exactly_20_passes(self):
        """20 atoms is the upper boundary — must pass."""
        coords = [[i * 0.05, 0.0, 0.0] for i in range(20)]
        s = Structure(Lattice.cubic(20.0), ["Si"] * 20, coords)
        result = check_atom_count(s)
        assert result.passed is True

    def test_atom_count_exceeds_20_fails(self):
        """21 atoms exceeds the limit."""
        coords = [[i * 0.045, 0.0, 0.0] for i in range(21)]
        s = Structure(Lattice.cubic(20.0), ["Si"] * 21, coords)
        result = check_atom_count(s)
        assert result.passed is False
        assert "21" in result.message
        assert result.stage == "atom_count"

    def test_atom_count_large_structure_fails(self):
        """50 atoms is well above the limit."""
        coords = [[i * 0.02, 0.0, 0.0] for i in range(50)]
        s = Structure(Lattice.cubic(30.0), ["C"] * 50, coords)
        result = check_atom_count(s)
        assert result.passed is False
        assert "50" in result.message

    def test_atom_count_message_includes_maximum(self):
        """Failure message should mention the maximum for actionable feedback."""
        coords = [[i * 0.045, 0.0, 0.0] for i in range(25)]
        s = Structure(Lattice.cubic(20.0), ["O"] * 25, coords)
        result = check_atom_count(s)
        assert result.passed is False
        assert "20" in result.message
