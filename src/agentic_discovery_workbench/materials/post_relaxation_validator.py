"""Post-relaxation validation for crystal structures.

Validates relaxed structures by checking interatomic distances, tracking
spacegroup symmetry changes, and detecting duplicates among the batch.
Produces one PostRelaxationReport per structure.

Symmetry changes between pre- and post-relaxation are *recorded* but do
not cause rejection — relaxation routinely breaks ideal symmetry, and the
important thing is to know it happened.  Only short distances and
duplicates cause rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from agentic_discovery.materials.validation import MIN_DISTANCE_COVALENT_RATIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard floor for minimum interatomic distance (Å).  Anything below this
# is numerical junk regardless of element pair.
DEFAULT_MIN_DISTANCE_THRESHOLD: float = 0.5

# Fraction of summed covalent radii below which two atoms in a relaxed
# structure are considered unphysically close.  Tighter than the
# pre-relaxation ratio (0.5) because relaxed structures should not
# retain compressed bonds.
POST_RELAX_COVALENT_RATIO: float = max(MIN_DISTANCE_COVALENT_RATIO, 0.7)

# Default symmetry precision for SpacegroupAnalyzer (Å).  Applied
# identically to both pre- and post-relaxation phases so that the
# symmetry comparison is internally consistent.
DEFAULT_SYMPREC: float = 0.1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PostRelaxationReport:
    """Validation report for a single relaxed structure.

    Attributes:
        structure_id: Identifier of the structure.
        distance_valid: True if minimum interatomic distance meets threshold.
        min_distance_angstrom: Shortest pairwise distance in the relaxed
            structure (periodic-aware).
        pre_relax_spacegroup: Space group number before relaxation.
        post_relax_spacegroup: Space group number after relaxation.
        symmetry_changed: True if the spacegroup changed during relaxation.
        is_duplicate: True if this structure duplicates an earlier one.
        duplicate_of: ID of the earlier structure, or None.
        passed: Overall verdict — True iff distance_valid AND NOT is_duplicate.
    """

    structure_id: str
    distance_valid: bool
    min_distance_angstrom: float
    pre_relax_spacegroup: int
    post_relax_spacegroup: int
    symmetry_changed: bool
    is_duplicate: bool
    duplicate_of: str | None
    passed: bool

    def __post_init__(self) -> None:
        """Enforce invariant: passed = distance_valid AND NOT is_duplicate."""
        expected = self.distance_valid and not self.is_duplicate
        if self.passed != expected:
            raise ValueError(
                f"passed must equal (distance_valid and not is_duplicate): "
                f"expected {expected}, got {self.passed}"
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_distances_post_relax(relaxed: Structure) -> tuple[bool, float]:
    """Check interatomic distances against covalent-radii-based threshold.

    Uses pymatgen Element.atomic_radius — the same covalent-radius source as
    the pre-relaxation validator — but applies POST_RELAX_COVALENT_RATIO
    (0.7) instead of the pre-relaxation ratio (0.5).

    Args:
        relaxed: Post-relaxation pymatgen Structure.

    Returns:
        Tuple of (distance_valid, min_distance_angstrom).  distance_valid is
        True iff every atom pair exceeds its per-pair covalent-radii threshold.
        min_distance_angstrom is the shortest pairwise distance (periodic-aware),
        or inf if the structure has fewer than 2 atoms.
    """
    n = len(relaxed)
    if n < 2:
        return True, float("inf")

    dist_matrix = relaxed.distance_matrix
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = float(np.min(dist_matrix))

    cov_radii: list[float] = []
    for site in relaxed:
        radius = Element(site.specie.symbol).atomic_radius
        if radius is None:
            raise ValueError(
                f"No atomic radius data for {site.specie.symbol}; "
                f"cannot compute covalent-ratio distance threshold"
            )
        cov_radii.append(float(radius))

    for i in range(n):
        for j in range(i + 1, n):
            threshold = POST_RELAX_COVALENT_RATIO * (
                cov_radii[i] + cov_radii[j]
            )
            if dist_matrix[i, j] < threshold:
                logger.info(
                    "Post-relax distance violation: %s(#%d)-%s(#%d) "
                    "%.4f Å < %.4f Å threshold",
                    relaxed[i].specie.symbol, i,
                    relaxed[j].specie.symbol, j,
                    dist_matrix[i, j],
                    threshold,
                )
                return False, min_dist

    return True, min_dist


def _get_spacegroup_number(structure: Structure, symprec: float) -> int:
    """Get the international space group number for a structure.

    Args:
        structure: Pymatgen Structure.
        symprec: Symmetry precision in angstroms for SpacegroupAnalyzer.

    Returns:
        Space group number (1–230).
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return analyzer.get_space_group_number()


def _check_symmetry(
    pre_relax: Structure,
    post_relax: Structure,
    symprec: float = DEFAULT_SYMPREC,
) -> tuple[int, int, bool]:
    """Compare spacegroup symmetry before and after relaxation.

    Uses SpacegroupAnalyzer with the *same* symprec for both structures
    so that the comparison is internally consistent.

    Args:
        pre_relax: Structure before relaxation.
        post_relax: Structure after relaxation.
        symprec: Symmetry precision in angstroms, forwarded to
            SpacegroupAnalyzer for both structures.

    Returns:
        Tuple of (pre_relax_spacegroup, post_relax_spacegroup,
        symmetry_changed).  symmetry_changed is True when the two
        spacegroup numbers differ.
    """
    pre_sg = _get_spacegroup_number(pre_relax, symprec)
    post_sg = _get_spacegroup_number(post_relax, symprec)
    return pre_sg, post_sg, pre_sg != post_sg


def _find_duplicate(
    struct_id: str,
    structure: Structure,
    canonical: list[tuple[str, Structure]],
    matcher: StructureMatcher,
) -> str | None:
    """Find the first earlier canonical structure that matches.

    Args:
        struct_id: Identifier of the query structure (for logging).
        structure: Post-relaxation query structure.
        canonical: Previously seen non-duplicate structures.
        matcher: Configured StructureMatcher.

    Returns:
        ID of the first matching canonical structure, or None.
    """
    for earlier_id, earlier_struct in canonical:
        if matcher.fit(structure, earlier_struct):
            logger.info(
                "Post-relax structure %s is duplicate of %s",
                struct_id,
                earlier_id,
            )
            return earlier_id
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_post_relaxation(
    structures: list[tuple[str, Structure, Structure]],
    *,
    min_distance_threshold: float = DEFAULT_MIN_DISTANCE_THRESHOLD,
    symprec: float = DEFAULT_SYMPREC,
) -> list[PostRelaxationReport]:
    """Validate a batch of relaxed crystal structures.

    For each ``(id, pre_relax, post_relax)`` triple:

    1. Check interatomic distances against covalent-radii-based threshold
       (POST_RELAX_COVALENT_RATIO) and a hard floor (min_distance_threshold).
    2. Compare spacegroups before and after relaxation (same *symprec*).
    3. Check for duplicates among post-relax structures via StructureMatcher.

    The overall ``passed`` flag is ``distance_valid AND NOT is_duplicate``.
    Symmetry changes are informational and do not cause rejection.

    Args:
        structures: List of ``(structure_id, pre_relax, post_relax)`` triples.
        min_distance_threshold: Hard floor for minimum interatomic distance
            in angstroms.  Applied alongside the covalent-radii-based check.
            Must be non-negative.
        symprec: Symmetry precision for SpacegroupAnalyzer (Å).  Applied
            identically to both pre- and post-relaxation analysis.

    Returns:
        One PostRelaxationReport per input structure, in input order.

    Raises:
        ValueError: If min_distance_threshold is negative.
    """
    if min_distance_threshold < 0:
        raise ValueError(
            f"min_distance_threshold must be non-negative, "
            f"got {min_distance_threshold}"
        )

    logger.info(
        "Post-relaxation validation on %d structures "
        "(min_dist_threshold=%.3f, symprec=%.3f)",
        len(structures),
        min_distance_threshold,
        symprec,
    )

    if not structures:
        return []

    matcher = StructureMatcher()
    canonical: list[tuple[str, Structure]] = []
    reports: list[PostRelaxationReport] = []

    for struct_id, pre_relax, post_relax in structures:
        cov_valid, min_dist = _check_distances_post_relax(post_relax)
        distance_valid = cov_valid and min_dist >= min_distance_threshold

        pre_sg, post_sg, symmetry_changed = _check_symmetry(
            pre_relax, post_relax, symprec
        )

        duplicate_of = _find_duplicate(
            struct_id, post_relax, canonical, matcher
        )
        is_dup = duplicate_of is not None

        if not is_dup:
            canonical.append((struct_id, post_relax))

        passed = distance_valid and not is_dup

        report = PostRelaxationReport(
            structure_id=struct_id,
            distance_valid=distance_valid,
            min_distance_angstrom=min_dist,
            pre_relax_spacegroup=pre_sg,
            post_relax_spacegroup=post_sg,
            symmetry_changed=symmetry_changed,
            is_duplicate=is_dup,
            duplicate_of=duplicate_of,
            passed=passed,
        )
        reports.append(report)

        logger.info(
            "Structure %s: dist_valid=%s sg=%d->%d dup=%s passed=%s",
            struct_id,
            distance_valid,
            pre_sg,
            post_sg,
            is_dup,
            passed,
        )

    return reports
