"""Molecular weight calculation using RDKit exact molecular weight.

Wraps ``Descriptors.ExactMolWt`` with input validation so callers get
a clear error instead of a cryptic RDKit segfault or AttributeError
when passing invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_molecular_weight(mol: Chem.Mol) -> float:
    """Return the exact molecular weight of *mol* in daltons.

    Uses RDKit's ``Descriptors.ExactMolWt``, which sums isotopic masses
    of the most abundant isotope for each atom (including implicit
    hydrogens).

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    float
        Exact molecular weight in daltons.

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
    return Descriptors.ExactMolWt(mol)
