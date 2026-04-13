"""Tests for agentic_materials_discovery.structure.symmetry."""

from __future__ import annotations

import pytest

from agentic_materials_discovery.structure.symmetry import (
    CRYSTAL_SYSTEM_SG_RANGES,
    crystal_system_to_sg_range,
    enforce_p1_policy,
    is_p1,
    sg_number_to_crystal_system,
)


# --- CRYSTAL_SYSTEM_SG_RANGES constant ---


def test_all_seven_systems_present():
    expected = {
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    }
    assert set(CRYSTAL_SYSTEM_SG_RANGES.keys()) == expected


def test_ranges_cover_all_230_sgs():
    total = sum(hi - lo + 1 for lo, hi in CRYSTAL_SYSTEM_SG_RANGES.values())
    assert total == 230, f"Ranges cover {total} SGs, expected 230"


def test_ranges_no_gaps():
    """Consecutive systems must leave no gap between max_i and min_{i+1}."""
    sorted_ranges = sorted(CRYSTAL_SYSTEM_SG_RANGES.values())
    for i in range(len(sorted_ranges) - 1):
        _, hi = sorted_ranges[i]
        lo_next, _ = sorted_ranges[i + 1]
        assert lo_next == hi + 1, (
            f"Gap between SG {hi} and {lo_next}"
        )


def test_ranges_no_overlaps():
    """No two systems should claim the same space-group number."""
    seen: dict[int, str] = {}
    for system, (lo, hi) in CRYSTAL_SYSTEM_SG_RANGES.items():
        for sg in range(lo, hi + 1):
            assert sg not in seen, (
                f"SG {sg} claimed by both {seen[sg]!r} and {system!r}"
            )
            seen[sg] = system


# --- crystal_system_to_sg_range ---


def test_cubic_range():
    assert crystal_system_to_sg_range("cubic") == (195, 230)


def test_triclinic_range():
    assert crystal_system_to_sg_range("triclinic") == (1, 2)


def test_orthorhombic_range():
    assert crystal_system_to_sg_range("orthorhombic") == (16, 74)


def test_crystal_system_case_insensitive():
    assert crystal_system_to_sg_range("CUBIC") == (195, 230)
    assert crystal_system_to_sg_range("Triclinic") == (1, 2)
    assert crystal_system_to_sg_range("  Hexagonal  ") == (168, 194)


def test_unknown_system_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown crystal system") as exc_info:
        crystal_system_to_sg_range("rhombohedral")
    assert "rhombohedral" in str(exc_info.value)


# --- sg_number_to_crystal_system ---


def test_sg_to_system_boundary_values():
    """Spot-check every boundary SG from the ticket list."""
    assert sg_number_to_crystal_system(1) == "triclinic"
    assert sg_number_to_crystal_system(2) == "triclinic"
    assert sg_number_to_crystal_system(3) == "monoclinic"
    assert sg_number_to_crystal_system(15) == "monoclinic"
    assert sg_number_to_crystal_system(16) == "orthorhombic"
    assert sg_number_to_crystal_system(74) == "orthorhombic"
    assert sg_number_to_crystal_system(75) == "tetragonal"
    assert sg_number_to_crystal_system(142) == "tetragonal"
    assert sg_number_to_crystal_system(143) == "trigonal"
    assert sg_number_to_crystal_system(167) == "trigonal"
    assert sg_number_to_crystal_system(168) == "hexagonal"
    assert sg_number_to_crystal_system(194) == "hexagonal"
    assert sg_number_to_crystal_system(195) == "cubic"
    assert sg_number_to_crystal_system(230) == "cubic"


def test_sg_to_system_sg1_triclinic():
    assert sg_number_to_crystal_system(1) == "triclinic"


def test_sg_to_system_sg230_cubic():
    assert sg_number_to_crystal_system(230) == "cubic"


def test_sg_to_system_invalid_raises():
    with pytest.raises(ValueError, match="outside the valid range") as exc_info:
        sg_number_to_crystal_system(0)
    assert "0" in str(exc_info.value)

    with pytest.raises(ValueError, match="outside the valid range") as exc_info:
        sg_number_to_crystal_system(231)
    assert "231" in str(exc_info.value)

    with pytest.raises(ValueError, match="outside the valid range") as exc_info:
        sg_number_to_crystal_system(-1)
    assert "-1" in str(exc_info.value)


# --- is_p1 ---


def test_is_p1_true():
    assert is_p1(1) is True


def test_is_p1_false():
    assert is_p1(2) is False
    assert is_p1(230) is False


# --- enforce_p1_policy ---


def test_enforce_p1_policy_blocked():
    ok, reason = enforce_p1_policy(1, allow_p1=False)
    assert ok is False
    assert reason == "P1 structures require allow_P1=true; auto-down-ranked per Q20 policy"


def test_enforce_p1_policy_allowed():
    ok, reason = enforce_p1_policy(1, allow_p1=True)
    assert ok is True
    assert reason == ""


def test_enforce_p1_policy_non_p1_always_allowed():
    """Non-P1 structures pass regardless of the allow_p1 flag."""
    ok_false, _ = enforce_p1_policy(225, allow_p1=False)
    ok_true, _ = enforce_p1_policy(225, allow_p1=True)
    assert ok_false is True
    assert ok_true is True
