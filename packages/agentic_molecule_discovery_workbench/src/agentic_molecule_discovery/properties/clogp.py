"""Crippen cLogP calculation using RDKit's Wildman-Crippen partition coefficient.

Wraps ``Crippen.MolLogP`` with input validation so callers get a clear
error instead of a cryptic RDKit segfault or AttributeError when passing
invalid input.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen


def calc_clogp(mol: Chem.Mol) -> float:
    """Return the Wildman-Crippen LogP estimate for *mol*.

    Uses RDKit's ``Crippen.MolLogP``, which assigns atom contributions
    based on the Wildman-Crippen parameterisation and sums them.  The
    result is a unitless partition coefficient (log10 of the
    octanol/water concentration ratio).

    Parameters
    ----------
    mol:
        An RDKit molecule object.  Must not be None.

    Returns
    -------
    float
        Estimated cLogP (unitless).

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
    return Crippen.MolLogP(mol)  # type: ignore[attr-defined]
