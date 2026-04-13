"""Validation error and result types for molecular structure validation.

Provides frozen dataclasses for reporting validation outcomes: individual
errors (syntax, valence, etc.) and aggregate results pairing a validity
flag with the parsed molecule and any accumulated errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem


@dataclass(frozen=True)
class ValidationError:
    """A single validation failure for a SMILES string.

    Parameters
    ----------
    error_type:
        Category of the error (e.g. ``'syntax'``, ``'valence'``).
    message:
        Human-readable description of what went wrong.
    smiles:
        The SMILES string that failed validation.
    """

    error_type: str
    message: str
    smiles: str


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate outcome of validating a SMILES string.

    Parameters
    ----------
    is_valid:
        Whether the molecule passed all validation checks.
    mol:
        The parsed RDKit Mol object, or ``None`` if parsing failed.
    errors:
        Validation errors encountered.  Empty when ``is_valid`` is ``True``.
    """

    is_valid: bool
    mol: Optional[Chem.Mol]
    # list rather than tuple per contract — callers must not mutate after
    # construction, but the ticket specifies list for interface consistency.
    errors: list[ValidationError]
