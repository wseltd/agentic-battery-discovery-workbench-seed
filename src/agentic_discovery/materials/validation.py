"""Materials structure validation: parseability, lattice sanity, element, and atom-count checks."""

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

# Maximum atom count for a single structure in the validation pipeline.
_MAX_ATOM_COUNT = 20

# Elements forbidden from generated inorganic structures.
# Noble gases are chemically inert and rarely form stable crystalline compounds.
# Tc (43) and Pm (61) have no stable isotopes — synthesising them is impractical.
# All Z >= 84 are radioactive with short-lived or difficult-to-handle isotopes.
FORBIDDEN_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    {2, 10, 18, 36, 54, 86}  # noble gases
    | {43, 61}                 # Tc, Pm — no stable isotopes
    | set(range(84, 119))      # Po through Og — all radioactive
)


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
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
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


def validate_allowed_elements(structure: Structure) -> ValidationResult:
    """Reject structures containing forbidden elements.

    Checks every site in *structure* against :data:`FORBIDDEN_ATOMIC_NUMBERS`.
    Any match is a hard failure — the element cannot appear in a practical
    inorganic-materials search.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`ValidationResult` with ``stage='allowed_elements'``.
    """
    offending: set[str] = set()
    for site in structure:
        for species, _ in site.species.items():
            if species.Z in FORBIDDEN_ATOMIC_NUMBERS:
                offending.add(species.symbol)

    if offending:
        names = ", ".join(sorted(offending))
        return ValidationResult(
            passed=False,
            stage="allowed_elements",
            message=f"Forbidden element(s): {names}",
            severity="hard",
        )

    return ValidationResult(
        passed=True, stage="allowed_elements", message="", severity="hard"
    )


def validate_atom_count(structure: Structure) -> ValidationResult:
    """Reject structures with zero atoms or more than the allowed maximum.

    Structures must contain between 1 and :data:`_MAX_ATOM_COUNT` atoms
    (inclusive).  Empty structures are physically meaningless; very large
    cells are outside the scope of the fast-validation pipeline.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`ValidationResult` with ``stage='atom_count'``.
    """
    n = len(structure)

    if n == 0:
        return ValidationResult(
            passed=False,
            stage="atom_count",
            message="Structure has 0 atoms",
            severity="hard",
        )

    if n > _MAX_ATOM_COUNT:
        return ValidationResult(
            passed=False,
            stage="atom_count",
            message=f"Structure has {n} atoms, exceeds maximum of {_MAX_ATOM_COUNT}",
            severity="hard",
        )

    return ValidationResult(
        passed=True, stage="atom_count", message="", severity="hard"
    )
