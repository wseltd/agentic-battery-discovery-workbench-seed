"""Canonical crystal representation built from pymatgen Structure."""

from __future__ import annotations

import dataclasses

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclasses.dataclass
class CrystalCanonical:
    """Immutable-ish canonical crystal snapshot extracted from a pymatgen Structure.

    Fields are set exclusively by the ``from_pymatgen_structure`` factory.
    Direct construction is not supported — ``__init__`` is disabled so that
    callers cannot create an inconsistent instance.
    """

    composition: str
    lattice: Lattice
    fractional_coords: np.ndarray
    species: list[Element | Species]
    space_group_number: int
    space_group_symbol: str
    site_properties: dict[str, list[object]]
    num_atoms: int
    _structure: Structure = dataclasses.field(repr=False)

    def __init__(self) -> None:  # noqa: D107
        raise TypeError(
            "CrystalCanonical cannot be constructed directly. "
            "Use CrystalCanonical.from_pymatgen_structure()."
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pymatgen_structure(
        cls,
        structure: Structure,
        symprec: float = 0.1,
    ) -> CrystalCanonical:
        """Build a CrystalCanonical from a pymatgen Structure.

        Args:
            structure: Any pymatgen ``Structure``.
            symprec: Symmetry tolerance passed to ``SpacegroupAnalyzer``.
                     Lower values detect less symmetry; higher values are
                     more forgiving of small distortions.

        Returns:
            A fully-populated ``CrystalCanonical``.
        """
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        sg_number = analyzer.get_space_group_number()
        sg_symbol = analyzer.get_space_group_symbol()

        # Wrap fractional coordinates into [0, 1)
        frac_coords = structure.frac_coords % 1.0

        # Site properties: convert each per-site array to a plain list
        site_props: dict[str, list[object]] = {
            key: list(vals) for key, vals in structure.site_properties.items()
        }

        obj = object.__new__(cls)
        obj.composition = structure.composition.reduced_formula
        obj.lattice = structure.lattice
        obj.fractional_coords = frac_coords
        obj.species = [site.specie for site in structure]
        obj.space_group_number = sg_number
        obj.space_group_symbol = sg_symbol
        obj.site_properties = site_props
        obj.num_atoms = len(structure)
        obj._structure = structure.copy()
        return obj

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_cif(self) -> str:
        """Serialise to CIF format, including space-group metadata."""
        writer = CifWriter(self._structure, symprec=0.1)
        return str(writer)

    def to_poscar(self) -> str:
        """Serialise to VASP POSCAR format."""
        poscar = Poscar(self._structure)
        return poscar.get_str()
