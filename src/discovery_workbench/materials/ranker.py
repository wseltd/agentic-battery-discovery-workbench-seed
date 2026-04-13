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

# Energy above hull threshold in eV/atom — candidates below this are
# considered thermodynamically accessible.  Includes metastable phases
# that are still synthesisable in practice.
STABILITY_THRESHOLD_EV = 0.1

# Score assigned to non-converged relaxation results.  Low but non-zero:
# the candidate might still be viable, we just cannot confirm stability
# from a failed relaxation.
NONCONVERGED_STABILITY_SCORE = 0.1

# Space group ranges defining crystal systems (inclusive bounds).
CRYSTAL_SYSTEM_RANGES: list[tuple[int, int]] = [
    (1, 2),       # Triclinic
    (3, 15),      # Monoclinic
    (16, 74),     # Orthorhombic
    (75, 142),    # Tetragonal
    (143, 167),   # Trigonal
    (168, 194),   # Hexagonal
    (195, 230),   # Cubic
]

# Partial credit for matching crystal system but not exact space group.
PARTIAL_SYMMETRY_SCORE = 0.5

# Volume-per-atom bounds in angstrom-cubed.  Outside this range structures
# are penalised as unrealistic.  Range covers dense metals (~7) to open
# frameworks (~40).
MIN_VOLUME_PER_ATOM = 7.0
MAX_VOLUME_PER_ATOM = 40.0

# Default composite-score weights.  Stability dominates because an
# unstable material is useless regardless of other properties.
DEFAULT_WEIGHTS: dict[str, float] = {
    "stability": 0.40,
    "symmetry": 0.20,
    "complexity": 0.10,
    "target_satisfaction": 0.30,
}

# Crystal system names, indexed to match CRYSTAL_SYSTEM_RANGES.
CRYSTAL_SYSTEM_NAMES: list[str] = [
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
]

# Name-to-range lookup for crystal system matching by name.
CRYSTAL_SYSTEM_BY_NAME: dict[str, tuple[int, int]] = dict(
    zip(CRYSTAL_SYSTEM_NAMES, CRYSTAL_SYSTEM_RANGES)
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RankedCandidate:
    """A ranked materials candidate with heuristic property scores.

    Args:
        candidate_id: Unique identifier for this candidate.
        composition: Chemical composition formula (e.g. "BaTiO3").
        space_group_number: International space group number (1-230).
        stability_score: Score from energy-above-hull assessment (0-1).
        symmetry_score: Score from space group matching (0-1).
        complexity_score: Score from structural complexity assessment (0-1).
        target_satisfaction_score: Fraction of target properties met (0-1).
        composite_score: Weighted combination of component scores.
        evidence_level: How scores were obtained.
        band_gap_eV: Band gap in electronvolts, None if not yet computed.
        bulk_modulus_GPa: Bulk modulus in gigapascals, None if not yet computed.
        rank: Position in ranked list (1 = best), 0 if unranked.
    """

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
    """Return crystal system index (0-6) for a space group number.

    Args:
        space_group: International space group number (1-230).

    Returns:
        Index into CRYSTAL_SYSTEM_RANGES.

    Raises:
        ValueError: If space_group is outside 1-230.
    """
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

    Candidates below STABILITY_THRESHOLD_EV score linearly from 1.0 (on
    hull) to 0.0 (at threshold).  Above threshold the score is 0.0.
    Non-converged relaxations get a fixed low score.

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

    Exact space group match scores 1.0, same crystal system scores
    PARTIAL_SYMMETRY_SCORE, different system scores 0.0.  If no target
    is specified returns 1.0 (no preference means no penalty).

    Args:
        candidate_space_group: Candidate's space group number (1-230).
        target_space_group: Desired space group, or None if no preference.

    Returns:
        Symmetry score in {0.0, 0.5, 1.0}.

    Raises:
        ValueError: If either space group is outside 1-230.
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

    Penalises extreme atomic density (volume per atom outside the
    physically reasonable range).  Within the reasonable range the score
    is 1.0 — we deliberately do not penalise atom count alone because
    large unit cells with reasonable density are valid candidates.

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

    Each target defines a minimum acceptable value.  A target is "met"
    if the property exists in *achieved* and its value >= the target.

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
    """Ranks materials candidates by weighted heuristic property scores.

    Args:
        weights: Override default component weights.  Keys must be from
            {"stability", "symmetry", "complexity", "target_satisfaction"}.
            Missing keys keep default values.  Weights are normalised
            internally so they sum to 1.0.
    """

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
        """Score, rank, and return candidates in descending composite order.

        Args:
            candidates: Each dict must have keys: candidate_id, composition,
                space_group_number, energy_above_hull, num_atoms, volume.
                Optional: converged (default True), achieved (default {}).
            target_space_group: Desired space group for symmetry scoring.
            targets: Property targets for satisfaction scoring.  If None,
                target_satisfaction_score is 1.0 for all candidates.

        Returns:
            List of RankedCandidate sorted by composite_score descending,
            with rank positions assigned starting at 1.
        """
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

        # Descending composite score; ties broken by candidate_id for determinism
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
    """Rank candidate materials by weighted heuristic property scores.

    Parameterised convenience function.  Scores each candidate on stability,
    symmetry, complexity, and target-property satisfaction, then returns them
    sorted by descending composite score with 1-based rank positions.

    Unlike MaterialsPropertyRanker.rank_candidates, this function accepts
    custom scoring parameters (threshold, crystal system, bounds) directly
    rather than relying on module-level constants.

    Args:
        candidates: Each dict must have keys: candidate_id, composition,
            space_group_number, energy_above_hull, num_atoms, volume.
            Optional: converged (default True), achieved (default {}).
        weights: Override default component weights.  Keys must be from
            {"stability", "symmetry", "complexity", "target_satisfaction"}.
        stability_threshold: Energy-above-hull cutoff in eV/atom.  Candidates
            with energy below this score linearly from 1.0 to 0.0.
        requested_sg: Desired space group number for symmetry scoring.
        requested_crystal_system: Desired crystal system name (e.g. "cubic").
            Used for symmetry scoring when requested_sg is None.
        complexity_bounds: Override volume-per-atom bounds.  Recognised keys:
            min_volume_per_atom, max_volume_per_atom.
        constraints: Property targets for satisfaction scoring.  Each key is
            a property name and the value is the minimum acceptable threshold.

    Returns:
        List of RankedCandidate sorted by composite_score descending,
        with 1-based rank positions assigned.

    Raises:
        ValueError: If stability_threshold is not positive, weights contain
            unknown keys, or candidate data is invalid.
    """
    logger.info("rank_candidates called with %d candidates", len(candidates))
    if stability_threshold <= 0:
        raise ValueError(
            f"stability_threshold must be positive, got {stability_threshold}"
        )
    if not candidates:
        return []

    # Reuse class weight normalisation logic
    ranker = MaterialsPropertyRanker(weights=weights)

    # Resolve volume-per-atom bounds for complexity scoring
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

    # Validate crystal system name early to fail before iterating candidates
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
        # --- Stability ---
        # Uses threshold parameter so callers can override the default cutoff
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

        # --- Symmetry ---
        if requested_sg is not None:
            symmetry = compute_symmetry_score(
                cand["space_group_number"], requested_sg,
            )
        elif cs_range is not None:
            low, high = cs_range
            sg = cand["space_group_number"]
            symmetry = 1.0 if low <= sg <= high else 0.0
        else:
            symmetry = 1.0  # No preference means no penalty

        # --- Complexity ---
        # Uses bounds parameters so callers can override the density range
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

        # --- Target satisfaction ---
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
