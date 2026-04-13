"""Tests for the shortlist assembler.

Focuses on the tricky parts: deduplication correctness, tie-breaking
determinism, and boundary conditions.  The function is pure, so no
mocks are needed.
"""

from __future__ import annotations

import pytest

from agentic_workbench.orchestration.shortlist import assemble_shortlist


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_returns_top_candidates_by_score_descending(self) -> None:
        """Candidates are ranked highest-score-first."""
        items = [("low", 0.1), ("mid", 0.5), ("high", 0.9), ("med", 0.3)]
        result = assemble_shortlist(items, output_count=4)

        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0] == ("high", 0.9)
        assert result[-1] == ("low", 0.1)

    def test_truncates_to_output_count(self) -> None:
        """Only output_count entries are returned even when more exist."""
        items = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)]
        result = assemble_shortlist(items, output_count=3)

        assert len(result) == 3
        assert [c for c, _ in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Deduplication — the hardest part of this function
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_candidate_keeps_highest_score(self) -> None:
        """When the same candidate appears with different scores, keep the best."""
        items = [("mol_a", 0.3), ("mol_a", 0.9), ("mol_b", 0.5)]
        result = assemble_shortlist(items, output_count=10)

        candidates = [c for c, _ in result]
        assert candidates.count("mol_a") == 1
        # The retained entry has the higher score
        mol_a_entry = next(pair for pair in result if pair[0] == "mol_a")
        assert mol_a_entry[1] == 0.9

    def test_many_duplicates_reduced_to_uniques(self) -> None:
        """Five copies of the same candidate collapse to one entry."""
        items = [("dup", 0.1 * i) for i in range(1, 6)]
        result = assemble_shortlist(items, output_count=10)

        assert len(result) == 1
        assert result[0][0] == "dup"
        assert result[0][1] == pytest.approx(0.5)

    def test_dedup_with_output_count_smaller_than_uniques(self) -> None:
        """Dedup runs before truncation so output_count applies to uniques."""
        items = [("a", 0.9), ("a", 0.8), ("b", 0.7), ("b", 0.6), ("c", 0.5)]
        result = assemble_shortlist(items, output_count=2)

        assert len(result) == 2
        assert result[0] == ("a", 0.9)
        assert result[1] == ("b", 0.7)

    def test_unhashable_candidates_included_without_dedup(self) -> None:
        """Dicts (unhashable) are passed through without dedup tracking."""
        d1 = {"formula": "NaCl"}
        d2 = {"formula": "NaCl"}  # same content, different object
        items = [(d1, 0.8), (d2, 0.6), ("hashable", 0.7)]
        result = assemble_shortlist(items, output_count=10)

        # Both dicts included because unhashable — no dedup possible
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tie-breaking determinism — stable sort guarantee
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_tied_scores_preserve_insertion_order(self) -> None:
        """Candidates with identical scores keep their original ordering."""
        items = [("first", 0.5), ("second", 0.5), ("third", 0.5)]
        result = assemble_shortlist(items, output_count=3)

        candidates = [c for c, _ in result]
        assert candidates == ["first", "second", "third"]

    def test_ties_at_truncation_boundary_are_deterministic(self) -> None:
        """When truncation cuts through a tie, the cut is reproducible."""
        items = [("top", 0.9), ("tie_a", 0.5), ("tie_b", 0.5), ("tie_c", 0.5)]
        result_1 = assemble_shortlist(items, output_count=3)
        result_2 = assemble_shortlist(items, output_count=3)

        assert result_1 == result_2
        # "tie_c" is excluded because it comes last in insertion order
        assert len(result_1) == 3
        assert result_1[0] == ("top", 0.9)
        assert result_1[1] == ("tie_a", 0.5)
        assert result_1[2] == ("tie_b", 0.5)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_empty_input_returns_empty(self) -> None:
        """No candidates in, no candidates out."""
        result = assemble_shortlist([], output_count=10)
        assert result == []

    def test_output_count_larger_than_candidates(self) -> None:
        """When output_count exceeds available candidates, return all."""
        items = [("a", 0.9), ("b", 0.8)]
        result = assemble_shortlist(items, output_count=100)

        assert len(result) == 2
        assert result[0] == ("a", 0.9)

    def test_output_count_zero_returns_empty(self) -> None:
        """Requesting zero candidates returns an empty list."""
        items = [("a", 0.9)]
        result = assemble_shortlist(items, output_count=0)
        assert result == []

    def test_single_candidate(self) -> None:
        """A single candidate is returned as-is."""
        result = assemble_shortlist([("only", 0.42)], output_count=5)
        assert result == [("only", 0.42)]

    def test_negative_output_count_raises_value_error(self) -> None:
        """Negative output_count is rejected at the boundary."""
        with pytest.raises(ValueError, match="output_count must be >= 0"):
            assemble_shortlist([("a", 0.5)], output_count=-1)


# ---------------------------------------------------------------------------
# Score edge cases
# ---------------------------------------------------------------------------


class TestScoreEdgeCases:
    def test_negative_scores_sorted_correctly(self) -> None:
        """Negative scores are sorted descending like positive ones."""
        items = [("a", -0.1), ("b", -0.9), ("c", -0.5)]
        result = assemble_shortlist(items, output_count=3)

        scores = [s for _, s in result]
        assert scores == [-0.1, -0.5, -0.9]

    def test_mixed_positive_negative_scores(self) -> None:
        """Positive scores rank above negative scores."""
        items = [("neg", -0.5), ("zero", 0.0), ("pos", 0.5)]
        result = assemble_shortlist(items, output_count=3)

        assert result[0] == ("pos", 0.5)
        assert result[1] == ("zero", 0.0)
        assert result[2] == ("neg", -0.5)

    def test_very_close_scores_not_conflated(self) -> None:
        """Scores differing by a tiny epsilon remain in correct order."""
        items = [("lower", 0.5000000001), ("higher", 0.5000000002)]
        result = assemble_shortlist(items, output_count=2)

        assert result[0][0] == "higher"
        assert result[1][0] == "lower"
