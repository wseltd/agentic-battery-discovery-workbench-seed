"""Materials property ranker with heuristic scoring.

Ranks candidate materials by composite scores derived from stability,
symmetry, complexity, and target property satisfaction.  All scores are
heuristic estimates unless DFT evidence is provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STABILITY_THRESHOLD_EV = 0.1

NONCONVERGED_STABILITY_SCORE = 0.1

CRYSTAL_SYSTEM_RANGES: list[tuple[int, int]] = [
    (1, 2),       # Triclinic
    (3, 15),      # Monoclinic
    (16, 74),     # Orthorhombic
    (75, 142),    # Tetragonal
    (143, 167),   # Trigonal
    (168, 194),   # Hexagonal
    (195, 230),   # Cubic
]

PARTIAL_SYMMETRY_SCORE = 0.5

MIN_VOLUME_PER_ATOM = 7.0
MAX_VOLUME_PER_ATOM = 40.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "stability": 0.40,
    "symmetry": 0.20,
    "complexity": 0.10,
    "target_satisfaction": 0.30,
}

CRYSTAL_SYSTEM_NAMES: list[str] = [
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
]

CRYSTAL_SYSTEM_BY_NAME: dict[str, tuple[int, int]] = dict(
    zip(CRYSTAL_SYSTEM_NAMES, CRYSTAL_SYSTEM_RANGES)
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RankedCandidate:
    """A ranked materials candidate with heuristic property scores."""

    DEFERRED_EVIDENCE_LEVEL: ClassVar[str] = "requested"

    candidate_id: str
    composition: str
    space_group_number: int
    stability_score: float
    symmetry_score: float
    complexity_score: float
    target_satisfaction_score: float
    composite_score: float
    evidence_level: str = "heuristic_estimated"
    band_gap_eV: float | None = None
    bulk_modulus_GPa: float | None = None
    rank: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crystal_system_index(space_group: int) -> int:
    """Return crystal system index (0-6) for a space group number."""
    for i, (low, high) in enumerate(CRYSTAL_SYSTEM_RANGES):
        if low <= space_group <= high:
            return i
    raise ValueError(f"space_group must be 1-230, got {space_group}")


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def compute_stability_score(
    energy_above_hull: float,
    converged: bool = True,
) -> float:
    """Score thermodynamic stability from energy above hull.

    Args:
        energy_above_hull: Energy above convex hull in eV/atom.
        converged: Whether the relaxation converged.

    Returns:
        Stability score in [0, 1].

    Raises:
        ValueError: If energy_above_hull is negative.
    """
    if energy_above_hull < 0:
        raise ValueError(
            f"energy_above_hull must be >= 0, got {energy_above_hull}"
        )
    if not converged:
        logger.info("Non-converged relaxation, assigning penalty score")
        return NONCONVERGED_STABILITY_SCORE

    return max(0.0, 1.0 - energy_above_hull / STABILITY_THRESHOLD_EV)


def compute_symmetry_score(
    candidate_space_group: int,
    target_space_group: int | None,
) -> float:
    """Score how well a candidate's symmetry matches the target.

    Args:
        candidate_space_group: Candidate's space group number (1-230).
        target_space_group: Desired space group, or None if no preference.

    Returns:
        Symmetry score in {0.0, 0.5, 1.0}.
    """
    if target_space_group is None:
        return 1.0

    candidate_system = _crystal_system_index(candidate_space_group)
    target_system = _crystal_system_index(target_space_group)

    if candidate_space_group == target_space_group:
        return 1.0
    if candidate_system == target_system:
        return PARTIAL_SYMMETRY_SCORE
    return 0.0


def compute_complexity_score(num_atoms: int, volume: float) -> float:
    """Score structural complexity from unit cell geometry.

    Args:
        num_atoms: Number of atoms in the unit cell.
        volume: Unit cell volume in angstrom-cubed.

    Returns:
        Complexity score in [0, 1].

    Raises:
        ValueError: If num_atoms or volume is not positive.
    """
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    if volume <= 0:
        raise ValueError(f"volume must be positive, got {volume}")

    volume_per_atom = volume / num_atoms

    if volume_per_atom < MIN_VOLUME_PER_ATOM:
        return max(0.0, volume_per_atom / MIN_VOLUME_PER_ATOM)

    if volume_per_atom > MAX_VOLUME_PER_ATOM:
        return max(0.0, MAX_VOLUME_PER_ATOM / volume_per_atom)

    return 1.0


def compute_target_satisfaction_score(
    targets: dict[str, float],
    achieved: dict[str, float],
) -> float:
    """Score what fraction of target property thresholds are met.

    Args:
        targets: Property name to minimum acceptable value.
        achieved: Property name to achieved value.

    Returns:
        Fraction of targets met, in [0, 1].

    Raises:
        ValueError: If targets is empty.
    """
    if not targets:
        raise ValueError("targets must not be empty")

    met = sum(
        1 for prop, threshold in targets.items()
        if prop in achieved and achieved[prop] >= threshold
    )
    return met / len(targets)


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class MaterialsPropertyRanker:
    """Ranks materials candidates by weighted heuristic property scores."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        logger.info("Initialising MaterialsPropertyRanker")
        raw = dict(DEFAULT_WEIGHTS)
        if weights:
            for key in weights:
                if key not in DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"Unknown weight key '{key}', "
                        f"valid keys: {sorted(DEFAULT_WEIGHTS)}"
                    )
            raw.update(weights)
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        self.weights = {k: v / total for k, v in raw.items()}

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        target_space_group: int | None = None,
        targets: dict[str, float] | None = None,
    ) -> list[RankedCandidate]:
        """Score, rank, and return candidates in descending composite order."""
        logger.info(
            "Ranking %d candidates with weights=%s",
            len(candidates), self.weights,
        )
        if not candidates:
            return []

        ranked: list[RankedCandidate] = []
        for cand in candidates:
            stability = compute_stability_score(
                cand["energy_above_hull"],
                cand.get("converged", True),
            )
            symmetry = compute_symmetry_score(
                cand["space_group_number"],
                target_space_group,
            )
            complexity = compute_complexity_score(
                cand["num_atoms"],
                cand["volume"],
            )
            if targets:
                target_sat = compute_target_satisfaction_score(
                    targets,
                    cand.get("achieved", {}),
                )
            else:
                target_sat = 1.0

            composite = (
                self.weights["stability"] * stability
                + self.weights["symmetry"] * symmetry
                + self.weights["complexity"] * complexity
                + self.weights["target_satisfaction"] * target_sat
            )

            ranked.append(RankedCandidate(
                candidate_id=cand["candidate_id"],
                composition=cand["composition"],
                space_group_number=cand["space_group_number"],
                stability_score=stability,
                symmetry_score=symmetry,
                complexity_score=complexity,
                target_satisfaction_score=target_sat,
                composite_score=composite,
            ))

        ranked.sort(key=lambda r: (-r.composite_score, r.candidate_id))

        for i, rc in enumerate(ranked, start=1):
            rc.rank = i

        return ranked


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    stability_threshold: float = STABILITY_THRESHOLD_EV,
    requested_sg: int | None = None,
    requested_crystal_system: str | None = None,
    complexity_bounds: dict[str, float] | None = None,
    constraints: dict[str, float] | None = None,
) -> list[RankedCandidate]:
    """Rank candidate materials by weighted heuristic property scores."""
    logger.info("rank_candidates called with %d candidates", len(candidates))
    if stability_threshold <= 0:
        raise ValueError(
            f"stability_threshold must be positive, got {stability_threshold}"
        )
    if not candidates:
        return []

    ranker = MaterialsPropertyRanker(weights=weights)

    min_vpa = MIN_VOLUME_PER_ATOM
    max_vpa = MAX_VOLUME_PER_ATOM
    if complexity_bounds is not None:
        min_vpa = complexity_bounds.get(
            "min_volume_per_atom", MIN_VOLUME_PER_ATOM,
        )
        max_vpa = complexity_bounds.get(
            "max_volume_per_atom", MAX_VOLUME_PER_ATOM,
        )
        if min_vpa <= 0 or max_vpa <= 0:
            raise ValueError("complexity_bounds values must be positive")
        if min_vpa >= max_vpa:
            raise ValueError(
                "min_volume_per_atom must be less than max_volume_per_atom"
            )

    cs_range: tuple[int, int] | None = None
    if requested_sg is None and requested_crystal_system is not None:
        cs_lower = requested_crystal_system.lower()
        if cs_lower not in CRYSTAL_SYSTEM_BY_NAME:
            raise ValueError(
                f"Unknown crystal system '{requested_crystal_system}', "
                f"valid: {sorted(CRYSTAL_SYSTEM_BY_NAME)}"
            )
        cs_range = CRYSTAL_SYSTEM_BY_NAME[cs_lower]

    ranked: list[RankedCandidate] = []
    for cand in candidates:
        energy = cand["energy_above_hull"]
        converged = cand.get("converged", True)
        if not converged:
            stability = NONCONVERGED_STABILITY_SCORE
        else:
            if energy < 0:
                raise ValueError(
                    f"energy_above_hull must be >= 0, got {energy}"
                )
            stability = max(0.0, 1.0 - energy / stability_threshold)

        if requested_sg is not None:
            symmetry = compute_symmetry_score(
                cand["space_group_number"], requested_sg,
            )
        elif cs_range is not None:
            low, high = cs_range
            sg = cand["space_group_number"]
            symmetry = 1.0 if low <= sg <= high else 0.0
        else:
            symmetry = 1.0

        num_atoms = cand["num_atoms"]
        volume = cand["volume"]
        if num_atoms <= 0:
            raise ValueError(f"num_atoms must be positive, got {num_atoms}")
        if volume <= 0:
            raise ValueError(f"volume must be positive, got {volume}")
        vpa = volume / num_atoms
        if vpa < min_vpa:
            complexity = max(0.0, vpa / min_vpa)
        elif vpa > max_vpa:
            complexity = max(0.0, max_vpa / vpa)
        else:
            complexity = 1.0

        if constraints:
            target_sat = compute_target_satisfaction_score(
                constraints, cand.get("achieved", {}),
            )
        else:
            target_sat = 1.0

        composite = (
            ranker.weights["stability"] * stability
            + ranker.weights["symmetry"] * symmetry
            + ranker.weights["complexity"] * complexity
            + ranker.weights["target_satisfaction"] * target_sat
        )

        ranked.append(RankedCandidate(
            candidate_id=cand["candidate_id"],
            composition=cand["composition"],
            space_group_number=cand["space_group_number"],
            stability_score=stability,
            symmetry_score=symmetry,
            complexity_score=complexity,
            target_satisfaction_score=target_sat,
            composite_score=composite,
        ))

    ranked.sort(key=lambda r: (-r.composite_score, r.candidate_id))
    for i, rc in enumerate(ranked, start=1):
        rc.rank = i

    return ranked
