"""Tests for energy-above-hull proxy classification.

Covers the three-band stability classification from estimate_energy_above_hull
(research-pack Q26).  All tests use synthetic energy dictionaries — no
external API calls, no pymatgen, no MatterSim.
"""

from agentic_materials_discovery.stability.hull import HullResult, estimate_energy_above_hull


class TestStableBelowThreshold:
    """Candidate below the lowest competing phase is stable."""

    def test_stable_below_threshold(self) -> None:
        # Candidate at -3.05 eV/atom vs competing phase at -3.0 eV/atom
        # gives energy_above_hull = -0.05, which is <= 0.0 → stable.
        result = estimate_energy_above_hull(
            energy_per_atom=-3.05,
            competing_phase_energies={"mp-123": -3.0},
        )

        assert isinstance(result, HullResult)
        assert result.energy_above_hull_ev < 0.0
        assert result.classification == "stable"
        assert result.evidence_level == "ml_predicted"
        assert result.caveat is None


class TestMetastableAtZeroBoundary:
    """Value just above 0.0 eV/atom falls in the metastable band."""

    def test_metastable_at_zero_boundary(self) -> None:
        # energy_above_hull = -3.0 - (-3.001) = 0.001 eV/atom
        # 0.0 < 0.001 <= 0.1 → metastable.
        result = estimate_energy_above_hull(
            energy_per_atom=-3.0,
            competing_phase_energies={"mp-456": -3.001},
        )

        assert result.energy_above_hull_ev > 0.0
        assert result.energy_above_hull_ev < 0.1
        assert result.classification == "metastable"
        assert result.evidence_level == "ml_predicted"
        assert result.caveat is None


class TestMetastableNearUpperBoundary:
    """Value near 0.1 eV/atom but still within metastable band."""

    def test_metastable_near_upper_boundary(self) -> None:
        # energy_above_hull = -2.901 - (-3.0) = 0.099 eV/atom
        # 0.0 < 0.099 <= 0.1 → metastable.
        result = estimate_energy_above_hull(
            energy_per_atom=-2.901,
            competing_phase_energies={"mp-789": -3.0},
        )

        assert abs(result.energy_above_hull_ev - 0.099) < 1e-9
        assert result.classification == "metastable"
        assert result.evidence_level == "ml_predicted"
        assert result.caveat is None


class TestUnstableAboveThreshold:
    """Value clearly above 0.1 eV/atom is unstable."""

    def test_unstable_above_threshold(self) -> None:
        # energy_above_hull = -2.85 - (-3.0) = 0.15 eV/atom
        # 0.15 > 0.1 → unstable.
        result = estimate_energy_above_hull(
            energy_per_atom=-2.85,
            competing_phase_energies={"mp-101": -3.0},
        )

        assert result.energy_above_hull_ev > 0.1
        assert result.classification == "unstable"
        assert result.evidence_level == "ml_predicted"
        assert result.caveat is None


class TestMissingPhaseCaveat:
    """Empty competing phases forces unstable with a caveat."""

    def test_missing_phase_caveat_sets_unstable(self) -> None:
        result = estimate_energy_above_hull(
            energy_per_atom=-3.0,
            competing_phase_energies={},
        )

        assert result.classification == "unstable"
        assert result.caveat is not None
        assert "no_competing_phases" in result.caveat
        assert result.evidence_level == "ml_predicted"
        # Hull distance is undefined — represented as infinity.
        assert result.energy_above_hull_ev == float("inf")


class TestExactThreshold:
    """Boundary precision: exactly 0.1 eV/atom is metastable, not unstable."""

    def test_exact_threshold_0_1_is_metastable(self) -> None:
        # energy_above_hull = -2.9 - (-3.0) = 0.1 eV/atom exactly.
        # 0.1 <= 0.1 → metastable (inclusive upper bound).
        result = estimate_energy_above_hull(
            energy_per_atom=-2.9,
            competing_phase_energies={"mp-200": -3.0},
        )

        assert abs(result.energy_above_hull_ev - 0.1) < 1e-9
        assert result.classification == "metastable"
        assert result.evidence_level == "ml_predicted"
        assert result.caveat is None
