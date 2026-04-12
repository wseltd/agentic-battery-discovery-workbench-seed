"""Tests for molecular_validity.stereocentre.flag_stereocentres."""

from __future__ import annotations

from rdkit import Chem

from molecular_validity.stereocentre import flag_stereocentres


# --- helpers -----------------------------------------------------------------

def _mol(smiles: str) -> Chem.Mol:
    """Parse SMILES, failing the test if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    return mol


# --- no stereocentres --------------------------------------------------------

class TestNoStereocentres:
    """Molecules without any chiral centres."""

    def test_simple_achiral_molecule(self):
        report = flag_stereocentres(_mol("CCO"))  # ethanol
        assert report.has_stereocentres is False
        assert report.specified == []
        assert report.unspecified == []
        assert report.total == 0

    def test_symmetric_molecule_has_no_centres(self):
        # Benzene — fully symmetric, no chiral centres
        report = flag_stereocentres(_mol("c1ccccc1"))
        assert report.has_stereocentres is False
        assert report.total == 0


# --- specified stereocentres -------------------------------------------------

class TestSpecifiedStereocentres:
    """Molecules where all chiral centres have assigned R/S configuration."""

    def test_single_specified_centre(self):
        # L-alanine with explicit S configuration at atom 1
        report = flag_stereocentres(_mol("N[C@@H](C)C(=O)O"))
        assert report.has_stereocentres is True
        assert len(report.specified) == 1
        assert report.unspecified == []
        assert report.total == 1
        # The chiral carbon is atom index 1
        assert 1 in report.specified

    def test_two_specified_centres(self):
        # L-threonine: two specified stereocentres
        report = flag_stereocentres(_mol("N[C@@H]([C@H](O)C)C(=O)O"))
        assert report.has_stereocentres is True
        assert len(report.specified) == 2
        assert report.unspecified == []
        assert report.total == 2


# --- unspecified stereocentres -----------------------------------------------

class TestUnspecifiedStereocentres:
    """Molecules with chiral centres but no assigned configuration."""

    def test_single_unspecified_centre(self):
        # Alanine without stereochemistry annotation — centre is unspecified
        report = flag_stereocentres(_mol("NC(C)C(=O)O"))
        assert report.has_stereocentres is True
        assert report.specified == []
        assert len(report.unspecified) == 1
        assert report.total == 1
        assert 1 in report.unspecified

    def test_multiple_unspecified_centres(self):
        # Threonine without any stereo annotation
        report = flag_stereocentres(_mol("NC(C(O)C)C(=O)O"))
        assert report.has_stereocentres is True
        assert report.specified == []
        assert len(report.unspecified) == 2
        assert report.total == 2


# --- mixed specified and unspecified -----------------------------------------

class TestMixedStereocentres:
    """Molecules with both specified and unspecified chiral centres."""

    def test_one_specified_one_unspecified(self):
        # Dot-disconnected: one fragment with specified, one without
        # [C@@H](F)(Cl)Br has specified R; C(F)(Cl)Br has unspecified
        report = flag_stereocentres(_mol("[C@@H](F)(Cl)Br.C(F)(Cl)Br"))
        assert report.has_stereocentres is True
        assert len(report.specified) == 1
        assert len(report.unspecified) == 1
        assert report.total == 2
        # Specified and unspecified lists must be disjoint
        assert set(report.specified).isdisjoint(set(report.unspecified))


# --- dataclass contract ------------------------------------------------------

class TestReportContract:
    """The StereocentreReport dataclass behaves as expected."""

    def test_report_is_frozen(self):
        report = flag_stereocentres(_mol("CCO"))
        try:
            report.total = 99  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "StereocentreReport should be frozen (immutable)"

    def test_total_equals_sum_of_lists(self):
        """total must always equal len(specified) + len(unspecified)."""
        report = flag_stereocentres(_mol("[C@@H](F)(Cl)Br.C(F)(Cl)Br"))
        assert report.total == len(report.specified) + len(report.unspecified)
