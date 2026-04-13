"""Tests for cycle-level duplicate filtering."""

from __future__ import annotations

import pytest

from agentic_discovery_core.orchestration.dedup import deduplicate_across_cycles


class TestCoreFiltering:
    def test_novel_candidates_pass_through(self) -> None:
        candidates = [("mol_a", 0.9), ("mol_b", 0.7)]
        filtered, updated = deduplicate_across_cycles(candidates, set())

        assert filtered == [("mol_a", 0.9), ("mol_b", 0.7)]
        assert "mol_a" in updated
        assert "mol_b" in updated

    def test_seen_candidates_removed(self) -> None:
        seen = {"mol_a"}
        candidates = [("mol_a", 0.9), ("mol_b", 0.7)]
        filtered, updated = deduplicate_across_cycles(candidates, seen)

        assert len(filtered) == 1
        assert filtered[0] == ("mol_b", 0.7)
        assert "mol_a" in updated
        assert "mol_b" in updated

    def test_all_duplicates_returns_empty(self) -> None:
        seen = {"a", "b", "c"}
        candidates = [("a", 0.5), ("b", 0.3), ("c", 0.1)]
        filtered, updated = deduplicate_across_cycles(candidates, seen)

        assert filtered == []
        assert updated == {"a", "b", "c"}

    def test_preserves_order_of_novel_candidates(self) -> None:
        seen = {"b"}
        candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        filtered, _ = deduplicate_across_cycles(candidates, seen)

        assert [c for c, _ in filtered] == ["a", "c", "d"]

    def test_scores_preserved_for_novel_candidates(self) -> None:
        candidates = [("x", 0.123456)]
        filtered, _ = deduplicate_across_cycles(candidates, set())

        assert filtered[0][1] == pytest.approx(0.123456)


class TestSeenSetMutation:
    def test_seen_set_updated_in_place(self) -> None:
        seen: set = set()
        deduplicate_across_cycles([("new", 0.5)], seen)

        assert "new" in seen

    def test_returned_seen_is_same_object(self) -> None:
        seen: set = set()
        _, returned = deduplicate_across_cycles([("x", 0.1)], seen)

        assert returned is seen

    def test_seen_grows_across_sequential_calls(self) -> None:
        seen: set = set()
        _, seen = deduplicate_across_cycles([("a", 0.9), ("b", 0.8)], seen)
        filtered, seen = deduplicate_across_cycles(
            [("b", 0.7), ("c", 0.6)], seen,
        )

        assert len(filtered) == 1
        assert filtered[0] == ("c", 0.6)
        assert seen == {"a", "b", "c"}


class TestBoundaryConditions:
    def test_empty_candidates_returns_empty(self) -> None:
        seen = {"existing"}
        filtered, updated = deduplicate_across_cycles([], seen)

        assert filtered == []
        assert updated == {"existing"}

    def test_single_novel_candidate(self) -> None:
        filtered, seen = deduplicate_across_cycles([("only", 0.5)], set())

        assert filtered == [("only", 0.5)]
        assert seen == {"only"}

    def test_single_duplicate_candidate(self) -> None:
        filtered, seen = deduplicate_across_cycles(
            [("dup", 0.5)], {"dup"},
        )

        assert filtered == []
        assert seen == {"dup"}

    def test_duplicate_within_same_batch(self) -> None:
        candidates = [("x", 0.9), ("x", 0.3)]
        filtered, seen = deduplicate_across_cycles(candidates, set())

        assert len(filtered) == 1
        assert filtered[0] == ("x", 0.9)
        assert "x" in seen


class TestInputValidation:
    def test_non_set_seen_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="seen must be a mutable set") as exc_info:
            deduplicate_across_cycles([("a", 0.5)], [])

        assert "list" in str(exc_info.value)

    def test_frozenset_seen_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="seen must be a mutable set") as exc_info:
            deduplicate_across_cycles(
                [("a", 0.5)], frozenset(),
            )

        assert "frozenset" in str(exc_info.value)
