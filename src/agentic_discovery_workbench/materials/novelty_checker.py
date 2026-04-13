"""Materials novelty classification against reference crystal databases.

Checks whether a post-relaxation crystal structure is already known in
Materials Project (MP) or Alexandria databases by querying for structures
with matching reduced composition, then applying pymatgen StructureMatcher.

Two-step check per database:
1. Query API for all structures with the same reduced formula.
2. Run StructureMatcher.fit() against each candidate.

If any candidate matches in any database, the structure is classified as
'known'. If no match is found across all databases, it is 'novel'. If an
API call fails, the result is flagged with reference_incomplete=True to
indicate partial coverage — the caller can decide whether to re-check
later or accept the partial result.

Chose pymatgen-default StructureMatcher tolerances (ltol=0.2, stol=0.3,
angle_tol=5) for consistency with T038 duplicate detection. Changing them
here would create silent divergence between the dedup and novelty stages.

No base class protocol for clients — only two concrete databases exist
(MP, Alexandria), and a Protocol would just add a governance-flagged
ellipsis body for zero benefit. Duck typing is sufficient.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)


# --- Constants ---------------------------------------------------------------

# Pymatgen-default StructureMatcher tolerances, same as T038 duplicate
# detection. Keeping them identical avoids silent divergence between
# the dedup and novelty stages of the pipeline.
DEFAULT_LTOL: float = 0.2
DEFAULT_STOL: float = 0.3
DEFAULT_ANGLE_TOL: float = 5.0

# Minimum seconds between consecutive API calls to avoid rate-limit bans.
# Conservative default — real MP/Alexandria limits are higher, but bursting
# during batch checks risks transient 429s.
RATE_LIMIT_DELAY_SECONDS: float = 0.1


# --- Data structures ---------------------------------------------------------


class MaterialsNoveltyClassification(StrEnum):
    """Binary novelty classification for a crystal structure."""

    KNOWN = "known"
    NOVEL = "novel"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@dataclass(frozen=True, slots=True)
class MaterialsNoveltyResult:
    """Result of a novelty check for one crystal structure.

    Attributes:
        structure_id: Identifier of the checked structure.
        classification: Whether the structure is known or novel.
        matched_reference_id: ID of the matching reference structure,
            or None if classified as novel.
        reference_db: Name of the reference database (e.g. 'MP').
        reference_db_version: Version string of the reference database.
        matcher_tolerances: StructureMatcher tolerances used
            (keys: ltol, stol, angle_tol).
        match_stage: Pipeline stage at which matching was done
            (always 'post_relax').
        reference_incomplete: True if an API failure prevented full
            database coverage during the check.
    """

    structure_id: str
    classification: MaterialsNoveltyClassification
    matched_reference_id: str | None
    reference_db: str
    reference_db_version: str
    matcher_tolerances: dict
    match_stage: str
    reference_incomplete: bool = False


class ReferenceDBClient:
    """Client for querying a reference crystal structure database by composition.

    Subclasses override ``_fetch_by_composition`` to provide actual database
    access.  The public ``query_by_composition`` method wraps the fetch with
    per-client rate limiting so callers do not need to manage timing.

    Args:
        db_name: Short name of the database (e.g. 'MP', 'Alexandria').
        db_version: Version string for the database snapshot.
    """

    def __init__(self, db_name: str, db_version: str) -> None:
        self._db_name = db_name
        self._db_version = db_version
        self._last_call_time: float = 0.0

    @property
    def db_name(self) -> str:
        """Short name of the reference database."""
        return self._db_name

    @property
    def db_version(self) -> str:
        """Version string for the database snapshot."""
        return self._db_version

    def __repr__(self) -> str:
        return (
            f"ReferenceDBClient(db_name={self._db_name!r}, "
            f"db_version={self._db_version!r})"
        )

    def query_by_composition(
        self, composition: str
    ) -> list[tuple[str, Structure]]:
        """Rate-limited query for structures matching a reduced composition.

        Enforces a minimum delay between consecutive calls to avoid
        API rate-limit bans, then delegates to ``_fetch_by_composition``.

        Args:
            composition: Reduced composition formula (e.g. 'NaCl').

        Returns:
            List of (reference_id, Structure) pairs from the database.
        """
        self._respect_rate_limit()
        return self._fetch_by_composition(composition)

    def _fetch_by_composition(
        self, composition: str
    ) -> list[tuple[str, Structure]]:
        """Fetch structures from the database.

        Base implementation returns an empty list and logs a warning.
        Subclasses override this to provide actual database access
        (e.g. MP API, Alexandria API).

        Args:
            composition: Reduced composition formula.

        Returns:
            List of (reference_id, Structure) pairs.  Empty in the base
            class — override to provide real data.
        """
        logger.warning(
            "No database backend for %s — override _fetch_by_composition "
            "in a subclass to query real data (composition=%s)",
            self._db_name,
            composition,
        )
        return []

    def _respect_rate_limit(self) -> None:
        """Block until the per-client rate-limit window has elapsed."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < RATE_LIMIT_DELAY_SECONDS:
            time.sleep(RATE_LIMIT_DELAY_SECONDS - elapsed)
        self._last_call_time = time.monotonic()


# --- Checker -----------------------------------------------------------------


class MaterialsNoveltyChecker:
    """Check crystal structures for novelty against reference databases.

    Queries each configured reference database for structures with matching
    reduced composition, then runs StructureMatcher to detect crystallographic
    matches.  Results are cached by (db_name, composition) to avoid redundant
    API calls when multiple candidate structures share a composition.

    Args:
        clients: Reference database clients to query.
        ltol: Fractional length tolerance for StructureMatcher.
        stol: Site tolerance for StructureMatcher.
        angle_tol: Angle tolerance in degrees for StructureMatcher.

    Raises:
        ValueError: If *clients* is empty.
    """

    def __init__(
        self,
        clients: list[ReferenceDBClient],
        ltol: float = DEFAULT_LTOL,
        stol: float = DEFAULT_STOL,
        angle_tol: float = DEFAULT_ANGLE_TOL,
    ) -> None:
        if not clients:
            raise ValueError("At least one ReferenceDBClient is required")
        self._clients = list(clients)
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol
        # Built once, reused for every check — StructureMatcher is stateless
        # after construction so this is safe.
        self._matcher = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol
        )
        # (db_name, composition) -> [(ref_id, Structure)]
        self._composition_cache: dict[
            tuple[str, str], list[tuple[str, Structure]]
        ] = {}

    @property
    def matcher_tolerances(self) -> dict:
        """Current StructureMatcher tolerance parameters."""
        return {
            "ltol": self._ltol,
            "stol": self._stol,
            "angle_tol": self._angle_tol,
        }

    def check(
        self, structure_id: str, structure: Structure
    ) -> MaterialsNoveltyResult:
        """Check one structure for novelty against all reference databases.

        Iterates through clients in order.  On the first crystallographic
        match the structure is classified as 'known' and the matching
        database's metadata is recorded.  If no match is found across all
        databases the structure is 'novel'.

        API failures are logged at WARNING and set ``reference_incomplete``
        on the result — the remaining databases are still checked so partial
        coverage is better than no coverage.

        Args:
            structure_id: Identifier for the structure being checked.
            structure: Post-relaxation pymatgen Structure to classify.

        Returns:
            MaterialsNoveltyResult with classification and match details.
        """
        logger.info("Novelty check for structure_id=%s", structure_id)

        composition = structure.composition.reduced_formula
        reference_incomplete = False
        tolerances = self.matcher_tolerances

        for client in self._clients:
            cache_key = (client.db_name, composition)

            if cache_key in self._composition_cache:
                candidates = self._composition_cache[cache_key]
            else:
                try:
                    candidates = client.query_by_composition(composition)
                    self._composition_cache[cache_key] = candidates
                except Exception as exc:
                    # API failure is not fatal — flag and continue to next DB
                    # so we still get partial coverage.
                    logger.warning(
                        "API query failed for db=%s composition=%s: %s",
                        client.db_name,
                        composition,
                        exc,
                    )
                    reference_incomplete = True
                    continue

            matched_id = self._find_match(structure, candidates)
            if matched_id is not None:
                return MaterialsNoveltyResult(
                    structure_id=structure_id,
                    classification=MaterialsNoveltyClassification.KNOWN,
                    matched_reference_id=matched_id,
                    reference_db=client.db_name,
                    reference_db_version=client.db_version,
                    matcher_tolerances=tolerances,
                    match_stage="post_relax",
                    reference_incomplete=reference_incomplete,
                )

        # No match found in any database — report first client's metadata
        # as the primary reference (arbitrary but deterministic).
        first = self._clients[0]
        return MaterialsNoveltyResult(
            structure_id=structure_id,
            classification=MaterialsNoveltyClassification.NOVEL,
            matched_reference_id=None,
            reference_db=first.db_name,
            reference_db_version=first.db_version,
            matcher_tolerances=tolerances,
            match_stage="post_relax",
            reference_incomplete=reference_incomplete,
        )

    def _find_match(
        self,
        structure: Structure,
        candidates: list[tuple[str, Structure]],
    ) -> str | None:
        """Return the ID of the first crystallographic match, or None.

        Args:
            structure: Query structure to match.
            candidates: Reference (id, Structure) pairs to compare against.

        Returns:
            ID of the first matching reference structure, or None.
        """
        for ref_id, ref_struct in candidates:
            if self._matcher.fit(structure, ref_struct):
                return ref_id
        return None


# --- Convenience function ----------------------------------------------------


def check_novelty(
    structure_id: str,
    structure: Structure,
    clients: list[ReferenceDBClient],
    ltol: float = DEFAULT_LTOL,
    stol: float = DEFAULT_STOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
) -> MaterialsNoveltyResult:
    """Check a single structure for novelty (convenience wrapper).

    Creates a fresh ``MaterialsNoveltyChecker`` and runs one check.  For
    batch processing, construct the checker once and call ``check()``
    repeatedly to benefit from composition caching.

    Args:
        structure_id: Identifier for the structure.
        structure: Post-relaxation pymatgen Structure.
        clients: Reference database clients to query.
        ltol: Fractional length tolerance.
        stol: Site tolerance.
        angle_tol: Angle tolerance in degrees.

    Returns:
        MaterialsNoveltyResult with classification and match details.
    """
    checker = MaterialsNoveltyChecker(
        clients=clients, ltol=ltol, stol=stol, angle_tol=angle_tol
    )
    return checker.check(structure_id, structure)
