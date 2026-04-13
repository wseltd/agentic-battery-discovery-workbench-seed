"""Tests for dimensionality analysis: DimensionalityResult and check_dimensionality."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from discovery_workbench.materials.validation.dimensionality import (
    DimensionalityResult,
    check_dimensionality,
)


# ---------------------------------------------------------------------------
# Fixtures — representative crystal structures
# ---------------------------------------------------------------------------

def _nacl_structure() -> Structure:
    """NaCl rock-salt primitive cell — 3D ionic framework."""
    return Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _fcc_cu() -> Structure:
    """FCC copper single-atom primitive cell — 3D metallic framework."""
    return Structure(Lattice.cubic(3.6), ["Cu"], [[0.0, 0.0, 0.0]])


def _graphite_structure() -> Structure:
    """Graphite 4-atom hexagonal cell — 2D layered structure.

    AB stacking with two layers, each containing two C atoms.
    MinimumDistanceNN detects in-plane C–C bonds (1.42 Å) but not
    the interlayer van der Waals gap (~3.35 Å), giving dim=2.
    """
    return Structure(
        Lattice.hexagonal(2.46, 6.71),
        ["C", "C", "C", "C"],
        [
            [0.0, 0.0, 0.25],
            [1 / 3, 2 / 3, 0.25],
            [0.0, 0.0, 0.75],
            [2 / 3, 1 / 3, 0.75],
        ],
    )


# ---------------------------------------------------------------------------
# 3D structure tests
# ---------------------------------------------------------------------------

def test_nacl_is_3d() -> None:
    result = check_dimensionality(_nacl_structure())
    assert result.is_3d is True
    assert result.dimensionality == 3


def test_fcc_single_atom_is_3d() -> None:
    result = check_dimensionality(_fcc_cu())
    assert result.is_3d is True
    assert result.dimensionality == 3


# ---------------------------------------------------------------------------
# Layered (2D) structure test
# ---------------------------------------------------------------------------

def test_graphite_layered_not_3d() -> None:
    result = check_dimensionality(_graphite_structure())
    assert result.is_3d is False
    assert result.dimensionality == 2
    assert result.component_count >= 1
    assert result.method == "larsen"


# ---------------------------------------------------------------------------
# Result field tests
# ---------------------------------------------------------------------------

def test_result_fields_populated() -> None:
    result = check_dimensionality(_nacl_structure())
    assert result.dimensionality == 3
    assert result.is_3d is True
    assert result.component_count >= 1
    assert result.method == "larsen"


def test_dimensionality_returns_integer() -> None:
    result = check_dimensionality(_nacl_structure())
    assert isinstance(result.dimensionality, int)
    assert result.dimensionality == 3


def test_component_count_positive() -> None:
    result = check_dimensionality(_nacl_structure())
    assert result.component_count > 0


def test_method_field_is_larsen() -> None:
    result = check_dimensionality(_fcc_cu())
    assert result.method == "larsen"


# ---------------------------------------------------------------------------
# Dataclass validation tests
# ---------------------------------------------------------------------------

def test_dimensionality_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="dimensionality must be between") as exc_info:
        DimensionalityResult(dimensionality=4, is_3d=False, component_count=1, method="larsen")
    assert "4" in str(exc_info.value)


def test_dimensionality_negative_raises() -> None:
    with pytest.raises(ValueError, match="dimensionality must be between") as exc_info:
        DimensionalityResult(dimensionality=-1, is_3d=False, component_count=1, method="larsen")
    assert "-1" in str(exc_info.value)


def test_negative_component_count_raises() -> None:
    with pytest.raises(ValueError, match="component_count must be >= 0") as exc_info:
        DimensionalityResult(dimensionality=3, is_3d=True, component_count=-1, method="larsen")
    assert "-1" in str(exc_info.value)


def test_type_error_on_non_structure() -> None:
    with pytest.raises(TypeError, match="Expected pymatgen Structure") as exc_info:
        check_dimensionality({"not": "a structure"})  # type: ignore[arg-type]
    assert "dict" in str(exc_info.value)
