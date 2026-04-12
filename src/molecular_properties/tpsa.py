"""Topological polar surface area calculation using RDKit.

Wraps ``Descriptors.TPSA`` with input validation so callers get a clear
error instead of a cryptic RDKit segfault or AttributeError when passing
invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_tpsa(mol: Chem.Mol) -> float:
    """Return the topological polar surface area of *mol* in angstroms squared.

    Uses RDKit's ``Descriptors.TPSA``, which sums fragment-based
    contributions from polar atoms (N, O, S, P) and their attached
    hydrogens using the Ertl parameterisation.

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    float
        TPSA in square angstroms (Å²).

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
    return Descriptors.TPSA(mol)
