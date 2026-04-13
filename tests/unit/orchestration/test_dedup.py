"""Tests for cycle-level duplicate filtering.

Heaviest coverage on the core filtering logic and boundary validation,
since incorrect dedup silently corrupts downstream rankings.
"""

from __future__ import annotations

import pytest

from agentic_workbench.orchestration.dedup import deduplicate_across_cycles


# ---------------------------------------------------------------------------
# Core filtering — the main contract
# ---------------------------------------------------------------------------


class TestCoreFiltering:
    def test_novel_candidates_pass_through(self) -> None:
        """Candidates not in seen are returned and added to seen."""
        candidates = [("mol_a", 0.9), ("mol_b", 0.7)]
        filtered, updated = deduplicate_across_cycles(candidates, set())

        assert filtered == [("mol_a", 0.9), ("mol_b", 0.7)]
        assert "mol_a" in updated
        assert "mol_b" in updated

    def test_seen_candidates_removed(self) -> None:
        """Candidates already in seen are excluded from output."""
        seen = {"mol_a"}
        candidates = [("mol_a", 0.9), ("mol_b", 0.7)]
        filtered, updated = deduplicate_across_cycles(candidates, seen)

        assert len(filtered) == 1
        assert filtered[0] == ("mol_b", 0.7)
        assert "mol_a" in updated
        assert "mol_b" in updated

    def test_all_duplicates_returns_empty(self) -> None:
        """When every candidate is already seen, output is empty."""
        seen = {"a", "b", "c"}
        candidates = [("a", 0.5), ("b", 0.3), ("c", 0.1)]
        filtered, updated = deduplicate_across_cycles(candidates, seen)

        assert filtered == []
        assert updated == {"a", "b", "c"}

    def test_preserves_order_of_novel_candidates(self) -> None:
        """Novel candidates appear in the same order as the input."""
        seen = {"b"}
        candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        filtered, _ = deduplicate_across_cycles(candidates, seen)

        assert [c for c, _ in filtered] == ["a", "c", "d"]

    def test_scores_preserved_for_novel_candidates(self) -> None:
        """Scores are passed through unchanged."""
        candidates = [("x", 0.123456)]
        filtered, _ = deduplicate_across_cycles(candidates, set())

        assert filtered[0][1] == pytest.approx(0.123456)


# ---------------------------------------------------------------------------
# Seen set mutation — the set is updated in place AND returned
# ---------------------------------------------------------------------------


class TestSeenSetMutation:
    def test_seen_set_updated_in_place(self) -> None:
        """The caller's seen set is mutated, not just a copy returned."""
        seen: set = set()
        deduplicate_across_cycles([("new", 0.5)], seen)

        assert "new" in seen

    def test_returned_seen_is_same_object(self) -> None:
        """The returned set is the same object, not a copy."""
        seen: set = set()
        _, returned = deduplicate_across_cycles([("x", 0.1)], seen)

        assert returned is seen

    def test_seen_grows_across_sequential_calls(self) -> None:
        """Simulates two cycles: second call filters first-cycle candidates."""
        seen: set = set()
        _, seen = deduplicate_across_cycles([("a", 0.9), ("b", 0.8)], seen)
        filtered, seen = deduplicate_across_cycles(
            [("b", 0.7), ("c", 0.6)], seen,
        )

        assert len(filtered) == 1
        assert filtered[0] == ("c", 0.6)
        assert seen == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_empty_candidates_returns_empty(self) -> None:
        """No candidates in, no candidates out, seen unchanged."""
        seen = {"existing"}
        filtered, updated = deduplicate_across_cycles([], seen)

        assert filtered == []
        assert updated == {"existing"}

    def test_single_novel_candidate(self) -> None:
        """A single unseen candidate passes through."""
        filtered, seen = deduplicate_across_cycles([("only", 0.5)], set())

        assert filtered == [("only", 0.5)]
        assert seen == {"only"}

    def test_single_duplicate_candidate(self) -> None:
        """A single already-seen candidate is filtered out."""
        filtered, seen = deduplicate_across_cycles(
            [("dup", 0.5)], {"dup"},
        )

        assert filtered == []
        assert seen == {"dup"}

    def test_duplicate_within_same_batch(self) -> None:
        """If the same candidate appears twice in one batch, only the first passes."""
        candidates = [("x", 0.9), ("x", 0.3)]
        filtered, seen = deduplicate_across_cycles(candidates, set())

        assert len(filtered) == 1
        assert filtered[0] == ("x", 0.9)
        assert "x" in seen


# ---------------------------------------------------------------------------
# Input validation — reject non-set seen arguments at the boundary
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_set_seen_raises_type_error(self) -> None:
        """A list passed as seen is rejected with a descriptive message."""
        with pytest.raises(TypeError, match="seen must be a mutable set") as exc_info:
            deduplicate_across_cycles([("a", 0.5)], [])  # type: ignore[arg-type]

        assert "list" in str(exc_info.value)

    def test_frozenset_seen_raises_type_error(self) -> None:
        """A frozenset is immutable and cannot accumulate seen candidates."""
        with pytest.raises(TypeError, match="seen must be a mutable set") as exc_info:
            deduplicate_across_cycles(
                [("a", 0.5)], frozenset(),  # type: ignore[arg-type]
            )

        assert "frozenset" in str(exc_info.value)
