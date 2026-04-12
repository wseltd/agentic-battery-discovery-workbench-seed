"""Stereocentre flagging for molecular structures.

Identifies chiral centres in a molecule using RDKit's FindMolChiralCenters
and partitions them into specified (R/S assigned) and unspecified (ambiguous).
This is an informational flag, not a pass/fail validation — molecules with
unspecified stereocentres may be intentional (racemates, unknown chirality).
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import FindMolChiralCenters


@dataclass(frozen=True, slots=True)
class StereocentreReport:
    """Summary of chiral centres found in a molecule.

    Attributes
    ----------
    has_stereocentres:
        True if the molecule contains at least one chiral centre.
    unspecified:
        Atom indices of chiral centres without assigned R/S configuration.
    specified:
        Atom indices of chiral centres with assigned R/S configuration.
    total:
        Total number of chiral centres (len(specified) + len(unspecified)).
    """

    has_stereocentres: bool
    unspecified: list[int]
    specified: list[int]
    total: int


def flag_stereocentres(mol: Chem.Mol) -> StereocentreReport:
    """Flag chiral centres in *mol*, partitioned by assignment status.

    Uses RDKit's ``FindMolChiralCenters`` with ``includeUnassigned=True``
    so that both specified (R/S) and unspecified (?) centres are detected.

    Parameters
    ----------
    mol:
        RDKit molecule to analyse.

    Returns
    -------
    StereocentreReport
        Atom indices split into specified and unspecified lists.
    """
    # includeUnassigned=True ensures we see '?' centres as well as R/S
    centres = FindMolChiralCenters(mol, includeUnassigned=True)

    specified: list[int] = []
    unspecified: list[int] = []

    for atom_idx, label in centres:
        if label == "?":
            unspecified.append(atom_idx)
        else:
            specified.append(atom_idx)

    total = len(specified) + len(unspecified)

    return StereocentreReport(
        has_stereocentres=total > 0,
        unspecified=unspecified,
        specified=specified,
        total=total,
    )
