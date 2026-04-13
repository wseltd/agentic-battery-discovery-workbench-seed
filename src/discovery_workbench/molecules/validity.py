"""Molecular validity pipeline — ordered validation chain.

Implements the syntax -> valence -> charge -> stereochemistry -> salt/fragment
validation chain for SMILES input.  Fail-fast on hard rejections (syntax,
valence, charge); collect-all for informational stages (stereocentre,
salt stripping).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdkit import Chem

from molecular_validity.formal_charge import check_formal_charge
from molecular_validity.salt_strip import strip_salts
from molecular_validity.stereocentre import flag_stereocentres

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of the full molecular validation pipeline.

    Attributes:
        is_valid: Whether the molecule passed all rejection stages.
        rejection_stage: Name of the first stage that rejected the molecule,
            or None if the molecule is valid.
        warnings: Informational messages from non-rejecting stages
            (e.g. undefined stereocentres, stripped counterions).
    """

    is_valid: bool
    rejection_stage: str | None = None
    warnings: list[str] = field(default_factory=list)


def validate_molecule(
    smiles: str,
    min_formal_charge: int = -1,
    max_formal_charge: int = 1,
) -> ValidationResult:
    """Validate a SMILES string through the ordered rejection chain.

    Stages run in order; the first hard-rejection stage terminates early.
    Informational stages run only after all rejection stages pass.

    Stage order:
      1. syntax — parse SMILES with RDKit
      2. valence — sanitize, checking for impossible valences
      3. charge — formal charge per atom within [min, max]
      4. stereocentre — flag undefined chiral centres (warning only)
      5. salt/fragment — strip counterions, keep largest organic piece (warning only)

    Args:
        smiles: SMILES string to validate.
        min_formal_charge: Minimum allowed formal charge per atom (inclusive).
        max_formal_charge: Maximum allowed formal charge per atom (inclusive).

    Returns:
        ValidationResult with is_valid flag, optional rejection_stage,
        and informational warnings list.

    Raises:
        ValueError: If min_formal_charge > max_formal_charge.
    """
    if not isinstance(smiles, str) or not smiles:
        return ValidationResult(is_valid=False, rejection_stage="syntax")

    # Stage 1: syntax — can RDKit parse this at all?
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        logger.info("Syntax rejection for SMILES: %s", smiles)
        return ValidationResult(is_valid=False, rejection_stage="syntax")

    # Stage 2: valence — does the molecule have chemically valid valences?
    try:
        Chem.SanitizeMol(mol)
    except ValueError as exc:
        logger.info("Valence rejection for SMILES %s: %s", smiles, exc)
        return ValidationResult(is_valid=False, rejection_stage="valence")

    # Stage 3: charge — are all formal charges within allowed range?
    charge_result = check_formal_charge(
        mol, min_charge=min_formal_charge, max_charge=max_formal_charge,
    )
    if not charge_result.is_valid:
        logger.info("Charge rejection for SMILES: %s", smiles)
        return ValidationResult(is_valid=False, rejection_stage="charge")

    # Informational stages — collect warnings, never reject
    warnings: list[str] = []

    # Stage 4: stereocentre — flag undefined chiral centres
    stereo = flag_stereocentres(mol)
    if stereo.unspecified:
        warnings.append(
            f"undefined stereocentre(s) at atom(s) {stereo.unspecified}"
        )

    # Stage 5: salt/fragment — strip counterions
    salt_result = strip_salts(mol)
    if salt_result.fragments_removed > 0:
        warnings.append(
            f"salt stripped: {salt_result.fragments_removed} fragment(s) removed"
        )

    return ValidationResult(is_valid=True, warnings=warnings)
