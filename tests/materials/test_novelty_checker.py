"""Tests for materials novelty checking against reference databases."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from pymatgen.core import Lattice, Structure

from agentic_discovery_workbench.materials.novelty_checker import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    MaterialsNoveltyChecker,
    MaterialsNoveltyClassification,
    MaterialsNoveltyResult,
    RATE_LIMIT_DELAY_SECONDS,
    ReferenceDBClient,
    check_novelty,
)


# ---------------------------------------------------------------------------
# Helpers — reusable structure builders
# ---------------------------------------------------------------------------


def _nacl_rocksalt() -> Structure:
    """NaCl rock-salt conventional cell (Fm-3m, 8 atoms)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ]
    return Structure(lattice, species, coords)


def _fe_bcc() -> Structure:
    """BCC iron primitive cell (Im-3m, 2 atoms)."""
    lattice = Lattice.cubic(2.87)
    return Structure(lattice, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def _mock_client(
    db_name: str = "MP",
    db_version: str = "v2022.10.28",
    candidates: list[tuple[str, Structure]] | None = None,
) -> Mock:
    """Create a mock ReferenceDBClient with controlled query responses.

    Args:
        db_name: Database name the mock reports.
        db_version: Database version the mock reports.
        candidates: Structures returned by query_by_composition.
            Defaults to empty list (no matches in DB).
    """
    client = Mock(spec=ReferenceDBClient)
    client.db_name = db_name
    client.db_version = db_version
    client.query_by_composition.return_value = (
        candidates if candidates is not None else []
    )
    return client


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    """Verify known/novel classification logic."""

    def test_known_structure_classified_as_known(self) -> None:
        """A structure matching a reference in the DB is classified as known."""
        nacl = _nacl_rocksalt()
        # Reference DB returns the same structure — StructureMatcher will match
        client = _mock_client(candidates=[("mp-22862", _nacl_rocksalt())])
        checker = MaterialsNoveltyChecker([client])

        result = checker.check("gen_001", nacl)

        assert result.classification == MaterialsNoveltyClassification.KNOWN
        assert result.classification == "known"  # StrEnum string equality
        assert result.matched_reference_id == "mp-22862"
        assert result.reference_db == "MP"

    def test_novel_structure_classified_as_novel(self) -> None:
        """A structure with no matches in any DB is classified as novel."""
        nacl = _nacl_rocksalt()
        # Reference DB returns empty — no known structures with this composition
        client = _mock_client(candidates=[])
        checker = MaterialsNoveltyChecker([client])

        result = checker.check("gen_002", nacl)

        assert result.classification == MaterialsNoveltyClassification.NOVEL
        assert result.classification == "novel"
        assert result.matched_reference_id is None

    def test_known_in_either_db_is_known(self) -> None:
        """If known in the second DB but not the first, still classified known.

        The checker iterates all clients — a match in any one is sufficient.
        """
        nacl = _nacl_rocksalt()
        # First DB: no structures with this composition
        mp_client = _mock_client(db_name="MP", candidates=[])
        # Second DB: has a match
        alex_client = _mock_client(
            db_name="Alexandria",
            db_version="v2024.1",
            candidates=[("alex-12345", _nacl_rocksalt())],
        )
        checker = MaterialsNoveltyChecker([mp_client, alex_client])

        result = checker.check("gen_003", nacl)

        assert result.classification == MaterialsNoveltyClassification.KNOWN
        assert result.reference_db == "Alexandria"
        assert result.matched_reference_id == "alex-12345"


# ---------------------------------------------------------------------------
# Tolerance and stage tests
# ---------------------------------------------------------------------------


class TestTolerancesAndStage:
    """Verify tolerance defaults and match stage invariant."""

    def test_tolerances_match_t038_strict_defaults(self) -> None:
        """Checker defaults must match T038 duplicate-detector tolerances.

        Both stages of the pipeline should use the same StructureMatcher
        parameters to avoid silent divergence.
        """
        client = _mock_client()
        checker = MaterialsNoveltyChecker([client])
        tols = checker.matcher_tolerances

        assert tols["ltol"] == DEFAULT_LTOL
        assert tols["stol"] == DEFAULT_STOL
        assert tols["angle_tol"] == DEFAULT_ANGLE_TOL

        # Confirm these are pymatgen's standard defaults
        assert tols["ltol"] == pytest.approx(0.2)
        assert tols["stol"] == pytest.approx(0.3)
        assert tols["angle_tol"] == pytest.approx(5.0)

    def test_match_stage_always_post_relax(self) -> None:
        """match_stage must be 'post_relax' regardless of classification.

        Novelty checking operates on post-relaxation structures only —
        pre-relax geometry is too distorted for reliable comparison.
        """
        nacl = _nacl_rocksalt()

        # Novel case
        client_empty = _mock_client(candidates=[])
        checker_novel = MaterialsNoveltyChecker([client_empty])
        result_novel = checker_novel.check("gen_novel", nacl)
        assert result_novel.match_stage == "post_relax"

        # Known case
        client_match = _mock_client(
            candidates=[("mp-22862", _nacl_rocksalt())]
        )
        checker_known = MaterialsNoveltyChecker([client_match])
        result_known = checker_known.check("gen_known", nacl)
        assert result_known.match_stage == "post_relax"


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


class TestCompositionCache:
    """Verify that composition-level caching avoids redundant API calls."""

    def test_composition_cache_avoids_redundant_api_calls(self) -> None:
        """Two structures with the same composition should trigger one API call.

        The second check should hit the composition cache, not the client.
        """
        nacl_a = _nacl_rocksalt()
        nacl_b = _nacl_rocksalt()
        # Both have composition NaCl — second check should be cached
        client = _mock_client(candidates=[])
        checker = MaterialsNoveltyChecker([client])

        checker.check("nacl_a", nacl_a)
        checker.check("nacl_b", nacl_b)

        # Client was called exactly once despite two checks with same composition
        assert client.query_by_composition.call_count == 1
        assert client.query_by_composition.call_args[0][0] == "NaCl"


# ---------------------------------------------------------------------------
# API failure tests
# ---------------------------------------------------------------------------


class TestAPIFailures:
    """Verify graceful handling of reference database API failures."""

    def test_api_failure_sets_reference_incomplete_flag(self) -> None:
        """When a client raises, reference_incomplete must be True.

        The checker should log the failure and continue to other DBs,
        not crash. The caller can see partial coverage from the flag.
        """
        nacl = _nacl_rocksalt()
        client = _mock_client()
        client.query_by_composition.side_effect = ConnectionError(
            "MP API unreachable"
        )
        checker = MaterialsNoveltyChecker([client])

        result = checker.check("gen_err", nacl)

        assert result.reference_incomplete is True
        # Classification should still be produced (novel, since no data)
        assert result.classification == MaterialsNoveltyClassification.NOVEL


# ---------------------------------------------------------------------------
# Composition-filtered query tests
# ---------------------------------------------------------------------------


class TestCompositionFiltering:
    """Verify that API queries filter by reduced composition."""

    def test_mp_api_query_filters_by_composition(self) -> None:
        """MP client receives the reduced formula as the composition filter."""
        nacl = _nacl_rocksalt()
        mp_client = _mock_client(db_name="MP", candidates=[])
        checker = MaterialsNoveltyChecker([mp_client])

        checker.check("gen_mp", nacl)

        assert mp_client.query_by_composition.call_count == 1
        assert mp_client.query_by_composition.call_args[0][0] == "NaCl"

    def test_alexandria_api_query_filters_by_composition(self) -> None:
        """Alexandria client receives the reduced formula as the filter."""
        fe = _fe_bcc()
        alex_client = _mock_client(
            db_name="Alexandria", db_version="v2024.1", candidates=[]
        )
        checker = MaterialsNoveltyChecker([alex_client])

        checker.check("gen_alex", fe)

        assert alex_client.query_by_composition.call_count == 1
        assert alex_client.query_by_composition.call_args[0][0] == "Fe"


# ---------------------------------------------------------------------------
# Result field completeness
# ---------------------------------------------------------------------------


class TestResultFields:
    """Verify MaterialsNoveltyResult structure and field completeness."""

    def test_novelty_result_fields_complete(self) -> None:
        """All declared fields must be present and hold correct types."""
        tolerances = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
        result = MaterialsNoveltyResult(
            structure_id="test_struct",
            classification=MaterialsNoveltyClassification.NOVEL,
            matched_reference_id=None,
            reference_db="MP",
            reference_db_version="v2022.10.28",
            matcher_tolerances=tolerances,
            match_stage="post_relax",
            reference_incomplete=False,
        )

        assert result.structure_id == "test_struct"
        assert result.classification == "novel"
        assert result.matched_reference_id is None
        assert result.reference_db == "MP"
        assert result.reference_db_version == "v2022.10.28"
        assert result.matcher_tolerances == tolerances
        assert result.match_stage == "post_relax"
        assert result.reference_incomplete is False

        # Known variant — all fields populated
        result_known = MaterialsNoveltyResult(
            structure_id="test_known",
            classification=MaterialsNoveltyClassification.KNOWN,
            matched_reference_id="mp-22862",
            reference_db="MP",
            reference_db_version="v2022.10.28",
            matcher_tolerances=tolerances,
            match_stage="post_relax",
        )
        assert result_known.matched_reference_id == "mp-22862"
        assert result_known.reference_incomplete is False  # default

    def test_result_includes_reference_db_version(self) -> None:
        """Result must carry the database version from the matching client."""
        nacl = _nacl_rocksalt()
        client = _mock_client(
            db_name="MP",
            db_version="v2023.11.1",
            candidates=[("mp-999", _nacl_rocksalt())],
        )
        checker = MaterialsNoveltyChecker([client])

        result = checker.check("gen_ver", nacl)

        assert result.reference_db_version == "v2023.11.1"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Verify rate-limiting behaviour between API calls."""

    @patch(
        "agentic_discovery_workbench.materials.novelty_checker.time"
    )
    def test_rate_limiting_respected(self, mock_time: Mock) -> None:
        """Consecutive uncached queries must wait for the rate-limit window.

        Uses two clients with different compositions to force two actual
        API calls (bypassing the composition cache). The rate limiter
        in ReferenceDBClient._respect_rate_limit should call time.sleep
        when the elapsed time is below the threshold.
        """
        # _last_call_time starts at 0.0 in __init__.  First monotonic read
        # returns 10.0 so elapsed=10.0 — no sleep needed.  After the first
        # query, _last_call_time is set to 10.0.  Second monotonic read
        # returns 10.01 so elapsed=0.01 < RATE_LIMIT_DELAY — triggers sleep.
        mock_time.monotonic.side_effect = [
            10.0,   # first _respect_rate_limit: elapsed = 10.0 - 0.0, OK
            10.0,   # first _respect_rate_limit: sets _last_call_time
            10.01,  # second _respect_rate_limit: elapsed = 0.01, too soon
            10.15,  # second _respect_rate_limit: sets _last_call_time
        ]
        mock_time.sleep = Mock()

        # Base class _fetch_by_composition returns [] (with a warning),
        # so we can use it directly without overriding anything.
        client = ReferenceDBClient(db_name="TestDB", db_version="v1")

        # Two calls — second should trigger rate-limit sleep
        client.query_by_composition("NaCl")
        client.query_by_composition("Fe")

        assert mock_time.sleep.call_count == 1
        sleep_duration = mock_time.sleep.call_args[0][0]
        assert sleep_duration == pytest.approx(
            RATE_LIMIT_DELAY_SECONDS - 0.01
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestCheckNoveltyFunction:
    """Verify the module-level check_novelty convenience function."""

    def test_check_novelty_returns_novel_for_unmatched(self) -> None:
        """check_novelty should produce a MaterialsNoveltyResult via the checker."""
        nacl = _nacl_rocksalt()
        client = _mock_client(candidates=[])

        result = check_novelty("gen_fn", nacl, [client])

        assert result.classification == MaterialsNoveltyClassification.NOVEL
        assert result.structure_id == "gen_fn"
        assert result.match_stage == "post_relax"
