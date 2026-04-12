"""Molecular property calculators.

Re-exports all public calculator functions from sub-modules.
Imports are deferred so that ``import molecular_properties`` succeeds
even when optional native dependencies (e.g. RDKit) are not installed;
individual calculator functions are resolved on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atom_and_ring_counts import (
        calc_aromatic_ring_count as calc_aromatic_ring_count,
        calc_heavy_atom_count as calc_heavy_atom_count,
        calc_ring_count as calc_ring_count,
    )
    from .clogp import calc_clogp as calc_clogp
    from .hydrogen_bonds import calc_hba as calc_hba, calc_hbd as calc_hbd
    from .molecular_weight import calc_molecular_weight as calc_molecular_weight
    from .rotatable_bonds import calc_rotatable_bonds as calc_rotatable_bonds
    from .tpsa import calc_tpsa as calc_tpsa

__all__ = [
    "calc_aromatic_ring_count",
    "calc_clogp",
    "calc_hba",
    "calc_hbd",
    "calc_heavy_atom_count",
    "calc_molecular_weight",
    "calc_ring_count",
    "calc_rotatable_bonds",
    "calc_tpsa",
]

# Map each public name to (submodule, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "calc_aromatic_ring_count": (".atom_and_ring_counts", "calc_aromatic_ring_count"),
    "calc_clogp": (".clogp", "calc_clogp"),
    "calc_hba": (".hydrogen_bonds", "calc_hba"),
    "calc_hbd": (".hydrogen_bonds", "calc_hbd"),
    "calc_heavy_atom_count": (".atom_and_ring_counts", "calc_heavy_atom_count"),
    "calc_molecular_weight": (".molecular_weight", "calc_molecular_weight"),
    "calc_ring_count": (".atom_and_ring_counts", "calc_ring_count"),
    "calc_rotatable_bonds": (".rotatable_bonds", "calc_rotatable_bonds"),
    "calc_tpsa": (".tpsa", "calc_tpsa"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
