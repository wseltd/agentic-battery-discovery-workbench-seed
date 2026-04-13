"""Focused tests for CIF and POSCAR export edge cases.

Complements the main test_dft_handoff.py suite with adversarial
inputs: empty structures, multi-element cells, large lattice
parameters, and symmetry preservation.
"""

import pytest
from pymatgen.core import Lattice, Structure

from agentic_materials_discovery.handoff.dft_handoff import (
    export_cif,
    export_poscar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def empty_structure():
    """Structure with zero sites — invalid for export."""
    lattice = Lattice.cubic(5.0)
    return Structure(lattice, [], [])


@pytest.fixture()
def ternary_structure():
    """SrTiO3 perovskite — three elements, five sites."""
    lattice = Lattice.cubic(3.905)
    return Structure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )


@pytest.fixture()
def large_cell_structure():
    """Structure with large lattice constants to check numeric formatting."""
    lattice = Lattice.cubic(50.0)
    return Structure(lattice, ["Ar"], [[0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# export_cif — edge cases
# ---------------------------------------------------------------------------

class TestExportCifEdgeCases:
    """CIF export validation and edge-case coverage."""

    def test_empty_structure_raises(self, empty_structure, tmp_path):
        """Zero-site structure must raise ValueError, not write a file."""
        path = tmp_path / "empty.cif"
        with pytest.raises(ValueError, match="no sites"):
            export_cif(empty_structure, path)
        assert not path.exists(), "File should not be created on error"

    def test_ternary_composition_preserved(self, ternary_structure, tmp_path):
        """All three elements of SrTiO3 must appear in the CIF."""
        path = tmp_path / "perovskite.cif"
        export_cif(ternary_structure, path)
        content = path.read_text()
        for elem in ("Sr", "Ti", "O"):
            assert elem in content, f"Element {elem} missing from CIF"

    def test_ternary_roundtrip_site_count(self, ternary_structure, tmp_path):
        """CIF roundtrip must preserve site count for multi-element cell."""
        path = tmp_path / "roundtrip.cif"
        export_cif(ternary_structure, path)
        parsed = Structure.from_file(str(path))
        assert len(parsed) == len(ternary_structure)

    def test_large_cell_lattice_in_cif(self, large_cell_structure, tmp_path):
        """Large lattice constant (50 A) must survive CIF export."""
        path = tmp_path / "large.cif"
        export_cif(large_cell_structure, path)
        parsed = Structure.from_file(str(path))
        assert abs(parsed.lattice.a - 50.0) < 0.01

    def test_cif_contains_symmetry_header(self, ternary_structure, tmp_path):
        """CIF output must contain a space-group header line."""
        path = tmp_path / "sym.cif"
        export_cif(ternary_structure, path)
        content = path.read_text()
        assert "_symmetry_space_group_name_H-M" in content

    def test_return_value_is_path(self, ternary_structure, tmp_path):
        """Return value must be the exact path written to."""
        path = tmp_path / "ret.cif"
        result = export_cif(ternary_structure, path)
        assert result == path


# ---------------------------------------------------------------------------
# export_poscar — edge cases
# ---------------------------------------------------------------------------

class TestExportPoscarEdgeCases:
    """POSCAR export validation and edge-case coverage."""

    def test_empty_structure_raises(self, empty_structure, tmp_path):
        """Zero-site structure must raise ValueError, not write a file."""
        path = tmp_path / "POSCAR_empty"
        with pytest.raises(ValueError, match="no sites"):
            export_poscar(empty_structure, path)
        assert not path.exists(), "File should not be created on error"

    def test_ternary_element_order(self, ternary_structure, tmp_path):
        """POSCAR must list all species from the ternary structure."""
        path = tmp_path / "POSCAR_ternary"
        export_poscar(ternary_structure, path)
        content = path.read_text()
        for elem in ("Sr", "Ti", "O"):
            assert elem in content, f"Element {elem} missing from POSCAR"

    def test_ternary_roundtrip_site_count(self, ternary_structure, tmp_path):
        """POSCAR roundtrip must preserve site count."""
        path = tmp_path / "POSCAR_rt"
        export_poscar(ternary_structure, path)
        parsed = Structure.from_file(str(path))
        assert len(parsed) == len(ternary_structure)

    def test_large_cell_lattice_in_poscar(self, large_cell_structure, tmp_path):
        """Large lattice constant (50 A) must survive POSCAR export."""
        path = tmp_path / "POSCAR_large"
        export_poscar(large_cell_structure, path)
        parsed = Structure.from_file(str(path))
        assert abs(parsed.lattice.a - 50.0) < 0.01

    def test_poscar_roundtrip_composition(self, ternary_structure, tmp_path):
        """POSCAR roundtrip must preserve composition formula."""
        path = tmp_path / "POSCAR_comp"
        export_poscar(ternary_structure, path)
        parsed = Structure.from_file(str(path))
        assert (
            parsed.composition.reduced_formula
            == ternary_structure.composition.reduced_formula
        )

    def test_return_value_is_path(self, ternary_structure, tmp_path):
        """Return value must be the exact path written to."""
        path = tmp_path / "POSCAR_ret"
        result = export_poscar(ternary_structure, path)
        assert result == path
