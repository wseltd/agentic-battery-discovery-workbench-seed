"""Salt stripping for molecular structures.

Decomposes a molecule into disconnected fragments, filters out purely
inorganic fragments (those containing no carbon), and returns the largest
organic fragment by heavy atom count.  Ties are broken by molecular weight.

This is a pre-processing step — many compound databases store molecules as
salt forms (e.g. sodium salts, hydrochloride salts) where the counterion
is irrelevant to the drug-like properties of the organic component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, GetMolFrags

# Atomic number of carbon — used to detect organic fragments
_CARBON_ATOMIC_NUM = 6


@dataclass(frozen=True, slots=True)
class SaltStripResult:
    """Result of salt stripping a molecule.

    Attributes
    ----------
    mol:
        The largest organic fragment, or None if no organic fragment exists.
    fragments_removed:
        Number of fragments discarded (inorganic + smaller organic).
    original_fragment_count:
        Total number of disconnected fragments in the input molecule.
    """

    mol: Optional[Chem.Mol]
    fragments_removed: int
    original_fragment_count: int


def _has_carbon(mol: Chem.Mol) -> bool:
    """Return True if *mol* contains at least one carbon atom."""
    return any(atom.GetAtomicNum() == _CARBON_ATOMIC_NUM for atom in mol.GetAtoms())


def _heavy_atom_count(mol: Chem.Mol) -> int:
    """Return the number of heavy (non-hydrogen) atoms."""
    return mol.GetNumHeavyAtoms()


def strip_salts(mol: Chem.Mol) -> SaltStripResult:
    """Strip salt counterions, keeping the largest organic fragment.

    Decomposes *mol* into disconnected fragments via ``GetMolFrags``,
    discards any fragment that contains no carbon (purely inorganic),
    then selects the largest remaining fragment by heavy atom count.
    Ties in heavy atom count are broken by molecular weight (heavier wins).

    Parameters
    ----------
    mol:
        RDKit molecule to process.  May contain multiple disconnected
        fragments (e.g. from SMILES like ``[Na+].[O-]c1ccccc1``).

    Returns
    -------
    SaltStripResult
        The selected fragment (or None if all fragments are inorganic),
        plus bookkeeping counts.
    """
    fragments = GetMolFrags(mol, asMols=True)
    original_count = len(fragments)

    organic = [frag for frag in fragments if _has_carbon(frag)]

    if not organic:
        return SaltStripResult(
            mol=None,
            fragments_removed=original_count,
            original_fragment_count=original_count,
        )

    # Sort by heavy atom count descending, then molecular weight descending
    # to break ties deterministically.
    organic.sort(
        key=lambda m: (_heavy_atom_count(m), Descriptors.ExactMolWt(m)),
        reverse=True,
    )
    best = organic[0]

    return SaltStripResult(
        mol=best,
        fragments_removed=original_count - 1,
        original_fragment_count=original_count,
    )
