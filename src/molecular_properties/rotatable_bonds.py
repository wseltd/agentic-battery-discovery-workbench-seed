"""Rotatable bonds count using RDKit.

Wraps ``Descriptors.NumRotatableBonds`` with input validation so callers
get a clear error instead of a cryptic RDKit segfault or AttributeError
when passing invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_rotatable_bonds(mol: Chem.Mol) -> int:
    """Return the number of rotatable bonds in *mol*.

    Uses RDKit's ``Descriptors.NumRotatableBonds``, which counts
    single bonds between heavy atoms that are not in a ring and not
    adjacent to a triple bond (Lipinski definition).

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Number of rotatable bonds (always >= 0).

    Raises
    ------
    TypeError
        If *mol* is not an ``rdkit.Chem.Mol`` instance.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError(
            f"Expected rdkit.Chem.Mol, got {type(mol).__name__}. "
            "Pass a valid RDKit molecule object."
        )
    return Descriptors.NumRotatableBonds(mol)
