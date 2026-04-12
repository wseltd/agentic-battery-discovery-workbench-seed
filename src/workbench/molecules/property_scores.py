"""QED and SA score property calculators.

Thin wrappers around RDKit's QED and SA score functions that return
structured PropertyScore results with evidence-level metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import rdkit
from rdkit.Chem import Mol
from rdkit.Chem.QED import qed
from rdkit.Contrib.SA_Score.sascorer import calculateScore as sa_score

from workbench.shared.evidence import EvidenceLevel


@dataclass(frozen=True, slots=True)
class PropertyScore:
    """A scored molecular property with evidence metadata.

    Attributes
    ----------
    name:
        Human-readable property name (e.g. ``'QED'``, ``'SA_Score'``).
    value:
        Numeric score value.
    evidence_level:
        Credibility tier from the shared evidence taxonomy.
    tool_version:
        Version string of the tool that produced the score.
    """

    name: str
    value: float
    evidence_level: EvidenceLevel
    tool_version: str


def compute_qed(mol: Mol) -> PropertyScore:
    """Compute quantitative estimate of drug-likeness (QED).

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    PropertyScore
        QED value in [0.0, 1.0] with heuristic evidence level.

    Raises
    ------
    TypeError
        If *mol* is None or not an RDKit Mol.
    """
    if not isinstance(mol, Mol):
        raise TypeError(
            f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
        )
    return PropertyScore(
        name="QED",
        value=qed(mol),
        evidence_level=EvidenceLevel.HEURISTIC_ESTIMATED,
        tool_version=rdkit.__version__,
    )


def compute_sa_score(mol: Mol) -> PropertyScore:
    """Compute synthetic accessibility score.

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    PropertyScore
        SA score in [1.0, 10.0] where lower means easier to synthesise.
        Evidence level is heuristic — the score is based on fragment
        contributions, not a retrosynthetic analysis.

    Raises
    ------
    TypeError
        If *mol* is None or not an RDKit Mol.
    """
    if not isinstance(mol, Mol):
        raise TypeError(
            f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
        )
    return PropertyScore(
        name="SA_Score",
        value=sa_score(mol),
        evidence_level=EvidenceLevel.HEURISTIC_ESTIMATED,
        tool_version=rdkit.__version__,
    )
