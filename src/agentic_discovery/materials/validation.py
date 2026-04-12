"""Materials structure validation: parseability and lattice sanity checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

_ALLOWED_SEVERITIES = frozenset({"hard", "soft"})

# Minimum credible unit-cell volume in Å³ — anything below this
# is numerically degenerate rather than physically meaningful.
_MIN_LATTICE_VOLUME = 0.1


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation stage.

    Args:
        passed: Whether the check succeeded.
        stage: Short label identifying which validation stage produced this result.
        message: Human-readable explanation (empty string on success).
        severity: 'hard' for fatal failures, 'soft' for warnings.
    """

    passed: bool
    stage: str
    message: str
    severity: Literal["hard", "soft"]

    def __post_init__(self) -> None:
        if self.severity not in _ALLOWED_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_ALLOWED_SEVERITIES)}, "
                f"got {self.severity!r}"
            )


def validate_structure_parseable(structure_dict: dict) -> ValidationResult:
    """Check whether *structure_dict* can be round-tripped into a pymatgen Structure.

    Accepts a dict in pymatgen's ``as_dict()`` format and attempts to
    reconstruct a :class:`~pymatgen.core.Structure` from it.  Any failure
    (missing keys, wrong types, malformed lattice) is reported as a hard
    parse error.

    Args:
        structure_dict: A dictionary representation of a pymatgen Structure.

    Returns:
        A :class:`ValidationResult` with ``stage='parse'``.
    """
    try:
        Structure.from_dict(structure_dict)
    except Exception as exc:
        logger.debug("Structure parse failed: %s", exc)
        return ValidationResult(
            passed=False,
            stage="parse",
            message=f"Cannot parse structure dict: {exc}",
            severity="hard",
        )
    return ValidationResult(
        passed=True, stage="parse", message="", severity="hard"
    )


def validate_lattice_sanity(structure: Structure) -> ValidationResult:
    """Check that the lattice of *structure* is numerically well-formed.

    Catches degenerate lattices that pymatgen can technically represent but
    that have no physical meaning: NaN/Inf entries, near-zero volume, and
    non-positive determinant.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`ValidationResult` with ``stage='lattice_sanity'``.
    """
    matrix = np.array(structure.lattice.matrix)

    if not np.all(np.isfinite(matrix)):
        return ValidationResult(
            passed=False,
            stage="lattice_sanity",
            message="Lattice matrix contains NaN or Inf",
            severity="hard",
        )

    det = float(np.linalg.det(matrix))

    if det <= 0:
        return ValidationResult(
            passed=False,
            stage="lattice_sanity",
            message=f"Lattice determinant is non-positive ({det:.6g})",
            severity="hard",
        )

    volume = abs(det)
    if volume < _MIN_LATTICE_VOLUME:
        return ValidationResult(
            passed=False,
            stage="lattice_sanity",
            message=f"Lattice volume {volume:.6g} Å³ is below minimum {_MIN_LATTICE_VOLUME}",
            severity="hard",
        )

    return ValidationResult(
        passed=True, stage="lattice_sanity", message="", severity="hard"
    )
