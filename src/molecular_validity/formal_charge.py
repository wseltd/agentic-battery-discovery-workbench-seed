"""Formal charge sanity checker for molecular structures.

Validates that every atom in a molecule has a formal charge within a
configurable range.  Designed as a fast pre-screen before expensive
downstream computation — catches obviously over-charged atoms early.
"""

from __future__ import annotations

from rdkit import Chem

from validation.models import ValidationError, ValidationResult


def check_formal_charge(
    mol: Chem.Mol,
    min_charge: int = -1,
    max_charge: int = 1,
) -> ValidationResult:
    """Check whether every atom's formal charge falls within [min_charge, max_charge].

    Parameters
    ----------
    mol:
        RDKit molecule to validate.
    min_charge:
        Minimum allowed formal charge (inclusive).
    max_charge:
        Maximum allowed formal charge (inclusive).

    Returns
    -------
    ValidationResult
        ``is_valid=True`` with empty errors when all atoms are in range.
        ``is_valid=False`` with one error per violating atom otherwise.

    Raises
    ------
    ValueError
        If *min_charge* > *max_charge*.
    """
    if min_charge > max_charge:
        raise ValueError(
            f"min_charge ({min_charge}) must be <= max_charge ({max_charge})"
        )

    errors: list[ValidationError] = []
    smiles = Chem.MolToSmiles(mol)

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge < min_charge or charge > max_charge:
            errors.append(
                ValidationError(
                    error_type="formal_charge",
                    message=(
                        f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) has formal "
                        f"charge {charge}, outside allowed range "
                        f"[{min_charge}, {max_charge}]"
                    ),
                    smiles=smiles,
                )
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        mol=mol,
        errors=errors,
    )
