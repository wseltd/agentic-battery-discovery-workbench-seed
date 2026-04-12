"""Hydrogen bond donor and acceptor counts using RDKit.

Wraps ``Descriptors.NumHDonors`` and ``Descriptors.NumHAcceptors`` with input
validation so callers get a clear error instead of a cryptic RDKit segfault
or AttributeError when passing invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_hbd(mol: Chem.Mol) -> int:
    """Return the number of hydrogen bond donors in *mol*.

    Uses RDKit's ``Descriptors.NumHDonors``, which counts N-H and O-H
    bonds using Lipinski's definition.

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Number of hydrogen bond donors.

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
    return Descriptors.NumHDonors(mol)


def calc_hba(mol: Chem.Mol) -> int:
    """Return the number of hydrogen bond acceptors in *mol*.

    Uses RDKit's ``Descriptors.NumHAcceptors``, which counts N and O
    atoms using Lipinski's definition.

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Number of hydrogen bond acceptors.

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
    return Descriptors.NumHAcceptors(mol)
