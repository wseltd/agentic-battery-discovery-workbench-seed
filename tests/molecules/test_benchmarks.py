"""Tests for molecular benchmark metrics."""

from __future__ import annotations

import pytest

from discovery_workbench.molecules.benchmarks import (
    MolecularBenchmarkResult,
    compute_internal_diversity,
    compute_molecular_benchmarks,
    compute_novelty,
    compute_shortlist_quality,
    compute_target_satisfaction,
    compute_uniqueness,
    compute_validity,
)

# Pre-computed InChIKeys to avoid rdkit imports in test code.
INCHIKEY_ETHANOL = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"   # CCO
INCHIKEY_ETHYLAMINE = "QUSNBJAOOMFDIB-UHFFFAOYSA-N"  # CCN
INCHIKEY_BENZENE = "UHOVQNZJYSORNB-UHFFFAOYSA-N"    # c1ccccc1


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

class TestValidity:
    """Tests for compute_validity — RDKit SMILES parse success."""

    def test_validity_all_valid(self) -> None:
        """Every parseable SMILES produces 100% validity."""
        total, valid, pct, canonical = compute_validity(
            ["CCO", "CCN", "c1ccccc1"],
        )
        assert total == 3
        assert valid == 3
        assert pct == pytest.approx(100.0)
        assert len(canonical) == 3
        # Canonical forms should be deterministic
        assert "CCO" in canonical
        assert "CCN" in canonical

    def test_validity_some_invalid(self) -> None:
        """Mix of valid and invalid SMILES gives correct count and percentage."""
        total, valid, pct, canonical = compute_validity(
            ["CCO", "not_a_molecule", "c1ccccc1", "???"],
        )
        assert total == 4
        assert valid == 2
        assert pct == pytest.approx(50.0)
        assert len(canonical) == 2

    def test_validity_all_invalid(self) -> None:
        """No parseable SMILES gives 0% validity with empty canonical list."""
        total, valid, pct, canonical = compute_validity(
            ["invalid", "also_bad", "xyz123"],
        )
        assert total == 3
        assert valid == 0
        assert pct == pytest.approx(0.0)
        assert canonical == []

    def test_validity_empty_input(self) -> None:
        """Empty input gives zero counts and 0% validity."""
        total, valid, pct, canonical = compute_validity([])
        assert total == 0
        assert valid == 0
        assert pct == 0.0
        assert canonical == []


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

class TestUniqueness:
    """Tests for compute_uniqueness — canonical SMILES deduplication."""

    def test_uniqueness_all_unique(self) -> None:
        """Distinct canonical SMILES give 100% uniqueness."""
        count, pct, unique = compute_uniqueness(["CCO", "CCN", "c1ccccc1"])
        assert count == 3
        assert pct == pytest.approx(100.0)
        assert len(unique) == 3

    def test_uniqueness_with_duplicates(self) -> None:
        """Duplicate canonical SMILES are collapsed."""
        count, pct, unique = compute_uniqueness(["CCO", "CCN", "CCO", "CCN"])
        assert count == 2
        assert pct == pytest.approx(50.0)
        assert unique == ["CCO", "CCN"]

    def test_uniqueness_canonical_dedup(self) -> None:
        """Non-canonical variants that canonicalise to the same string merge.

        compute_validity canonicalises before passing to compute_uniqueness,
        so the same molecule written differently collapses to one entry.
        """
        # Both OCC and C(C)O canonicalise to CCO via compute_validity
        _, _, _, canonical = compute_validity(["OCC", "C(C)O", "CCN"])
        count, pct, unique = compute_uniqueness(canonical)
        assert count == 2  # CCO + CCN
        assert pct == pytest.approx(66.666, rel=0.01)


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

class TestNovelty:
    """Tests for compute_novelty — InChIKey set-difference against reference."""

    def test_novelty_all_novel(self) -> None:
        """Empty reference set means everything is novel."""
        count, pct, novel = compute_novelty(
            ["CCO", "CCN", "c1ccccc1"], set(),
        )
        assert count == 3
        assert pct == pytest.approx(100.0)
        assert novel == ["CCO", "CCN", "c1ccccc1"]

    def test_novelty_some_known(self) -> None:
        """Molecules whose InChIKeys are in the reference set are excluded."""
        count, pct, novel = compute_novelty(
            ["CCO", "CCN", "c1ccccc1"],
            {INCHIKEY_ETHANOL, INCHIKEY_BENZENE},
        )
        assert count == 1
        assert pct == pytest.approx(33.333, rel=0.01)
        assert novel == ["CCN"]

    def test_novelty_all_known(self) -> None:
        """All InChIKeys in reference set gives 0% novelty."""
        count, pct, novel = compute_novelty(
            ["CCO", "CCN"],
            {INCHIKEY_ETHANOL, INCHIKEY_ETHYLAMINE},
        )
        assert count == 0
        assert pct == pytest.approx(0.0)
        assert novel == []


# ---------------------------------------------------------------------------
# Target satisfaction
# ---------------------------------------------------------------------------

class TestTargetSatisfaction:
    """Tests for compute_target_satisfaction — (min, max) window checking."""

    def test_target_satisfaction_all_met(self) -> None:
        """All molecules within a wide MW window — 100% satisfaction."""
        # Ethanol MW ~46, ethylamine MW ~45, benzene MW ~78 — all in [30, 500]
        result = compute_target_satisfaction(
            ["CCO", "CCN", "c1ccccc1"],
            {"mw": (30.0, 500.0)},
        )
        assert result == pytest.approx(1.0)

    def test_target_satisfaction_none_met(self) -> None:
        """No molecule within an impossibly high MW window — 0%."""
        result = compute_target_satisfaction(
            ["CCO", "CCN"],
            {"mw": (10000.0, None)},
        )
        assert result == pytest.approx(0.0)

    def test_target_satisfaction_partial(self) -> None:
        """Only molecules within the window count as satisfied.

        Ethanol MW ~46 < 100 (fails), aspirin MW ~180 in [100, 500] (passes).
        """
        result = compute_target_satisfaction(
            ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
            {"mw": (100.0, 500.0)},
        )
        assert result == pytest.approx(0.5)

    def test_target_satisfaction_upper_bound_excludes(self) -> None:
        """Max bound excludes molecules above the ceiling.

        Benzene MW ~78 passes (78 <= 80), ethanol MW ~46 passes,
        aspirin MW ~180 fails (180 > 80).
        """
        result = compute_target_satisfaction(
            ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
            {"mw": (None, 80.0)},
        )
        assert result == pytest.approx(2.0 / 3.0)

    def test_target_satisfaction_empty_targets_returns_one(self) -> None:
        """Empty constraints dict is vacuously satisfied → 1.0."""
        result = compute_target_satisfaction(["CCO"], {})
        assert result == pytest.approx(1.0)

    def test_target_satisfaction_unknown_property_raises(self) -> None:
        """Requesting an unsupported property raises ValueError."""
        with pytest.raises(ValueError, match="Unknown property") as exc_info:
            compute_target_satisfaction(
                ["CCO"], {"bogus_property": (0.0, 1.0)},
            )
        assert "bogus_property" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Internal diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    """Tests for compute_internal_diversity — pairwise Tanimoto distance."""

    def test_diversity_identical_molecules(self) -> None:
        """Identical fingerprints produce zero diversity."""
        mean_d, std_d = compute_internal_diversity(["CCO", "CCO", "CCO"])
        assert mean_d == pytest.approx(0.0)
        assert std_d == pytest.approx(0.0)

    def test_diversity_diverse_molecules(self) -> None:
        """Structurally distinct molecules have high pairwise diversity.

        Ethanol, aspirin, and caffeine share almost no Morgan substructures.
        """
        mean_d, std_d = compute_internal_diversity([
            "CCO",
            "CC(=O)Oc1ccccc1C(=O)O",
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        ])
        assert mean_d > 0.5, f"Expected high diversity, got {mean_d}"
        assert isinstance(std_d, float)

    def test_diversity_single_molecule(self) -> None:
        """No pairwise comparisons possible → (0.0, 0.0)."""
        mean_d, std_d = compute_internal_diversity(["CCO"])
        assert mean_d == 0.0
        assert std_d == 0.0

    def test_diversity_empty_returns_zero(self) -> None:
        """Empty input gives zero diversity."""
        mean_d, std_d = compute_internal_diversity([])
        assert mean_d == 0.0
        assert std_d == 0.0


# ---------------------------------------------------------------------------
# Shortlist quality
# ---------------------------------------------------------------------------

class TestShortlistQuality:
    """Tests for compute_shortlist_quality — PAINS, QED, SA, clustering."""

    def test_shortlist_quality_pains_pass(self) -> None:
        """Simple clean molecules should all pass PAINS."""
        pains_pct, _, _, _ = compute_shortlist_quality(
            ["CCO", "CCN", "c1ccccc1"],
        )
        assert pains_pct == pytest.approx(100.0)

    def test_shortlist_quality_pains_with_alert(self) -> None:
        """Benzoquinone triggers PAINS, reducing pass rate.

        2 clean + 1 PAINS hit → 66.7% pass rate.
        """
        pains_pct, _, _, _ = compute_shortlist_quality([
            "CCO",
            "CCN",
            "O=C1C=CC(=O)C=C1",  # benzoquinone — PAINS A hit
        ])
        assert pains_pct == pytest.approx(66.666, rel=0.01)

    def test_shortlist_quality_median_qed(self) -> None:
        """Median QED must be in the valid range (0, 1]."""
        _, median_qed, _, _ = compute_shortlist_quality(
            ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
        )
        assert 0.0 < median_qed <= 1.0
        # Ethanol QED ~0.407, aspirin QED ~0.550; median ~0.478
        assert median_qed == pytest.approx(0.478, abs=0.05)

    def test_shortlist_quality_median_sa(self) -> None:
        """Median SA score must be in the valid range [1, 10]."""
        _, _, median_sa, _ = compute_shortlist_quality(
            ["CCO", "c1ccccc1"],
        )
        assert 1.0 <= median_sa <= 10.0
        # Small molecules are easy to synthesise → low SA scores
        assert median_sa < 4.0

    def test_shortlist_quality_cluster_count(self) -> None:
        """Structurally diverse molecules form multiple Butina clusters."""
        _, _, _, cluster_count = compute_shortlist_quality([
            "CCO",                                  # aliphatic alcohol
            "c1ccc2ccccc2c1",                       # naphthalene
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",          # caffeine
        ])
        # These share almost no fingerprint bits → each is its own cluster
        assert cluster_count >= 2

    def test_shortlist_quality_empty_returns_zeros(self) -> None:
        """Empty input gives all-zero quality metrics."""
        pains, qed, sa, clusters = compute_shortlist_quality([])
        assert pains == 0.0
        assert qed == 0.0
        assert sa == 0.0
        assert clusters == 0


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------

class TestBenchmarksEndToEnd:
    """Tests for compute_molecular_benchmarks — full pipeline."""

    def test_benchmarks_end_to_end(self) -> None:
        """Full pipeline with known inputs and value assertions.

        Input: 5 SMILES with 1 invalid and 1 duplicate.
        Reference set contains benzene's InChIKey.
        """
        smiles = ["CCO", "CCN", "c1ccccc1", "invalid", "CCO"]
        known = {INCHIKEY_BENZENE}
        result = compute_molecular_benchmarks(smiles, known, {})

        # Validity: 4 parse ok out of 5
        assert result.total_generated == 5
        assert result.valid_count == 4
        assert result.validity_pct == pytest.approx(80.0)

        # Uniqueness: CCO, CCN, c1ccccc1 — 3 unique out of 4 valid
        assert result.unique_count == 3
        assert result.uniqueness_pct == pytest.approx(75.0)

        # Novelty: benzene is known → CCO, CCN novel — 2 out of 3 unique
        assert result.novel_count == 2
        assert result.novelty_pct == pytest.approx(66.666, rel=0.01)

        # No targets → vacuously satisfied
        assert result.target_satisfaction_fraction == pytest.approx(1.0)

        # Diversity between CCO and CCN — both small aliphatics but distinct
        assert result.diversity_mean > 0.0
        assert isinstance(result.diversity_std, float)

        # Shortlist quality on 2 novel molecules
        assert result.shortlist_pains_pass_pct == pytest.approx(100.0)
        assert 0.0 < result.shortlist_median_qed <= 1.0
        assert 1.0 <= result.shortlist_median_sa <= 10.0
        assert result.shortlist_cluster_count >= 1

    def test_benchmarks_empty_smiles(self) -> None:
        """Empty input produces zero counts and zero percentages throughout."""
        result = compute_molecular_benchmarks([], set(), {})

        assert result.total_generated == 0
        assert result.valid_count == 0
        assert result.validity_pct == 0.0
        assert result.unique_count == 0
        assert result.uniqueness_pct == 0.0
        assert result.novel_count == 0
        assert result.novelty_pct == 0.0
        assert result.target_satisfaction_fraction == 1.0  # empty targets
        assert result.diversity_mean == 0.0
        assert result.diversity_std == 0.0
        assert result.shortlist_pains_pass_pct == 0.0
        assert result.shortlist_median_qed == 0.0
        assert result.shortlist_median_sa == 0.0
        assert result.shortlist_cluster_count == 0

    def test_benchmarks_all_invalid(self) -> None:
        """All-invalid input zeroes out downstream metrics."""
        result = compute_molecular_benchmarks(
            ["bad1", "bad2", "bad3"], set(), {},
        )
        assert result.total_generated == 3
        assert result.valid_count == 0
        assert result.validity_pct == 0.0
        assert result.unique_count == 0
        assert result.novel_count == 0

    def test_benchmarks_result_is_frozen(self) -> None:
        """MolecularBenchmarkResult is immutable."""
        result = compute_molecular_benchmarks(["CCO"], set(), {})
        assert isinstance(result, MolecularBenchmarkResult)
        assert result.total_generated == 1
        assert result.valid_count == 1
        with pytest.raises(AttributeError):
            result.total_generated = 999  # type: ignore[misc]
