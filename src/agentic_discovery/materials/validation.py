"""Materials structure validation: parseability, lattice sanity, element, atom-count,
interatomic-distance, and coordination-sanity checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pymatgen.core import Element, Structure

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
# Fraction of summed covalent radii below which two atoms are considered
# unphysically close.  0.5 catches overlapping/collapsed sites while
# tolerating compressed bonds that MatterGen occasionally proposes.
MIN_DISTANCE_COVALENT_RATIO: float = 0.5

FORBIDDEN_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    {2, 10, 18, 36, 54, 86}  # noble gases
    | {43, 61}                 # Tc, Pm — no stable isotopes
    | set(range(84, 119))      # Po through Og — all radioactive
)


@dataclass(frozen=True)
class MaterialsValidationResult:
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


def validate_lattice_sanity(structure: Structure) -> MaterialsValidationResult:
    """Check that the lattice of *structure* is numerically well-formed.

    Catches degenerate lattices that pymatgen can technically represent but
    that have no physical meaning: NaN/Inf entries, near-zero volume, and
    non-positive determinant.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='lattice_sanity'``.
    """
    matrix = np.array(structure.lattice.matrix)

    if not np.all(np.isfinite(matrix)):
        return MaterialsValidationResult(
            passed=False,
            stage="lattice_sanity",
            message="Lattice matrix contains NaN or Inf",
            severity="hard",
        )

    det = float(np.linalg.det(matrix))

    if det <= 0:
        return MaterialsValidationResult(
            passed=False,
            stage="lattice_sanity",
            message=f"Lattice determinant is non-positive ({det:.6g})",
            severity="hard",
        )

    volume = abs(det)
    if volume < _MIN_LATTICE_VOLUME:
        return MaterialsValidationResult(
            passed=False,
            stage="lattice_sanity",
            message=f"Lattice volume {volume:.6g} Å³ is below minimum {_MIN_LATTICE_VOLUME}",
            severity="hard",
        )

    return MaterialsValidationResult(
        passed=True, stage="lattice_sanity", message="", severity="hard"
    )


def validate_allowed_elements(structure: Structure) -> MaterialsValidationResult:
    """Reject structures containing forbidden elements.

    Checks every site in *structure* against :data:`FORBIDDEN_ATOMIC_NUMBERS`.
    Any match is a hard failure — the element cannot appear in a practical
    inorganic-materials search.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='allowed_elements'``.
    """
    offending: set[str] = set()
    for site in structure:
        for species, _ in site.species.items():
            if species.Z in FORBIDDEN_ATOMIC_NUMBERS:
                offending.add(species.symbol)

    if offending:
        names = ", ".join(sorted(offending))
        return MaterialsValidationResult(
            passed=False,
            stage="allowed_elements",
            message=f"Forbidden element(s): {names}",
            severity="hard",
        )

    return MaterialsValidationResult(
        passed=True, stage="allowed_elements", message="", severity="hard"
    )


def validate_atom_count(structure: Structure) -> MaterialsValidationResult:
    """Reject structures with zero atoms or more than the allowed maximum.

    Structures must contain between 1 and :data:`_MAX_ATOM_COUNT` atoms
    (inclusive).  Empty structures are physically meaningless; very large
    cells are outside the scope of the fast-validation pipeline.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='atom_count'``.
    """
    n = len(structure)

    if n == 0:
        return MaterialsValidationResult(
            passed=False,
            stage="atom_count",
            message="Structure has 0 atoms",
            severity="hard",
        )

    if n > _MAX_ATOM_COUNT:
        return MaterialsValidationResult(
            passed=False,
            stage="atom_count",
            message=f"Structure has {n} atoms, exceeds maximum of {_MAX_ATOM_COUNT}",
            severity="hard",
        )

    return MaterialsValidationResult(
        passed=True, stage="atom_count", message="", severity="hard"
    )


def validate_interatomic_distances(structure: Structure) -> MaterialsValidationResult:
    """Hard-reject structures where any atom pair is unphysically close.

    Uses pymatgen's periodic-aware :attr:`Structure.distance_matrix` (which
    accounts for minimum-image convention) and compares each pair distance
    against ``MIN_DISTANCE_COVALENT_RATIO * (r_cov_i + r_cov_j)`` where
    *r_cov* is :pyattr:`Element.atomic_radius`.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='interatomic_distances'``
        and ``severity='hard'``.
    """
    n = len(structure)
    if n < 2:
        return MaterialsValidationResult(
            passed=True,
            stage="interatomic_distances",
            message="",
            severity="hard",
        )

    # Periodic-aware distances — not raw Cartesian.
    dist_matrix = structure.distance_matrix

    cov_radii = [
        float(Element(site.specie.symbol).atomic_radius or 0.0)
        for site in structure
    ]

    violations: list[str] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            threshold = MIN_DISTANCE_COVALENT_RATIO * (cov_radii[i] + cov_radii[j])
            if d < threshold:
                sym_i = structure[i].specie.symbol
                sym_j = structure[j].specie.symbol
                violations.append(
                    f"{sym_i}(#{i})–{sym_j}(#{j}): {d:.4f} Å < {threshold:.4f} Å"
                )

    if violations:
        detail = "; ".join(violations)
        return MaterialsValidationResult(
            passed=False,
            stage="interatomic_distances",
            message=f"Atoms too close: {detail}",
            severity="hard",
        )

    return MaterialsValidationResult(
        passed=True,
        stage="interatomic_distances",
        message="",
        severity="hard",
    )


def validate_coordination_sanity(structure: Structure) -> MaterialsValidationResult:
    """Soft-flag structures with implausible coordination numbers.

    Uses pymatgen :class:`~pymatgen.analysis.local_env.CrystalNN` to compute
    the coordination number of each site.  A site with CN 0 (isolated) or
    CN > 12 (hyper-coordinated) is flagged as suspicious.  This is a *soft*
    check — MatterGen may intentionally produce unusual coordination, so we
    warn rather than reject.

    Args:
        structure: A pymatgen :class:`~pymatgen.core.Structure`.

    Returns:
        A :class:`MaterialsValidationResult` with ``stage='coordination_sanity'``
        and ``severity='soft'``.
    """
    # Import here — CrystalNN is expensive to import and only needed for this
    # stage, not the fast-path distance/element checks.
    from pymatgen.analysis.local_env import CrystalNN

    nn = CrystalNN()
    anomalies: list[str] = []

    for idx, site in enumerate(structure):
        try:
            cn = nn.get_cn(structure, idx)
        except Exception:
            # CrystalNN can fail on pathological geometries; treat as CN 0.
            logger.warning(
                "CrystalNN failed for site %d (%s); treating as isolated",
                idx,
                site.specie.symbol,
            )
            cn = 0

        if cn == 0:
            anomalies.append(f"{site.specie.symbol}(#{idx}): CN=0 (isolated)")
        elif cn > 12:
            anomalies.append(f"{site.specie.symbol}(#{idx}): CN={cn} (>12)")

    if anomalies:
        detail = "; ".join(anomalies)
        return MaterialsValidationResult(
            passed=False,
            stage="coordination_sanity",
            message=f"Unusual coordination: {detail}",
            severity="soft",
        )

    return MaterialsValidationResult(
        passed=True,
        stage="coordination_sanity",
        message="",
        severity="soft",
    )
