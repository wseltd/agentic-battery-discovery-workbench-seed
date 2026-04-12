"""Atom and ring count calculators using RDKit.

Wraps ``Mol.GetNumHeavyAtoms``, ``Descriptors.RingCount``, and
``rdMolDescriptors.CalcNumAromaticRings`` with input validation so callers
get a clear error instead of a cryptic RDKit segfault or AttributeError
when passing invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def _validate_mol(mol: Chem.Mol) -> None:
    """Reject non-Mol inputs early with a helpful message."""
    if not isinstance(mol, Chem.Mol):
        raise TypeError(
            f"Expected rdkit.Chem.Mol, got {type(mol).__name__}. "
            "Pass a valid RDKit molecule object."
        )


def calc_heavy_atom_count(mol: Chem.Mol) -> int:
    """Return the number of heavy (non-hydrogen) atoms in *mol*.

    Uses ``Mol.GetNumHeavyAtoms()``, which counts all atoms except
    explicit and implicit hydrogens.

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Number of heavy atoms (always >= 0).

    Raises
    ------
    TypeError
        If *mol* is not an ``rdkit.Chem.Mol`` instance.
    """
    _validate_mol(mol)
    return mol.GetNumHeavyAtoms()


def calc_ring_count(mol: Chem.Mol) -> int:
    """Return the total number of rings in *mol* (SSSR count).

    Uses RDKit's ``Descriptors.RingCount``, which returns the number
    of rings in the smallest set of smallest rings (SSSR).

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Total ring count (always >= 0).

    Raises
    ------
    TypeError
        If *mol* is not an ``rdkit.Chem.Mol`` instance.
    """
    _validate_mol(mol)
    return Descriptors.RingCount(mol)


def calc_aromatic_ring_count(mol: Chem.Mol) -> int:
    """Return the number of aromatic rings in *mol*.

    Uses ``rdMolDescriptors.CalcNumAromaticRings``, which counts
    rings where every bond in the ring is aromatic.

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    int
        Number of aromatic rings (always >= 0).

    Raises
    ------
    TypeError
        If *mol* is not an ``rdkit.Chem.Mol`` instance.
    """
    _validate_mol(mol)
    return rdMolDescriptors.CalcNumAromaticRings(mol)
