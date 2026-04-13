"""Tests for the molecular validity pipeline.

Integration tests against real RDKit parsing — no mocks.  Covers:
  - 3 valid drug-like SMILES (happy path)
  - 3 rejection cases asserting the specific rejection_stage
  - 3 edge cases asserting on warnings list contents
"""

from discovery_workbench.molecules.validity import validate_molecule


# ---------------------------------------------------------------------------
# Valid SMILES — happy path
# ---------------------------------------------------------------------------


def test_valid_smiles_aspirin_passes() -> None:
    """Aspirin (acetylsalicylic acid) passes all validation stages."""
    result = validate_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert result.is_valid is True
    assert result.rejection_stage is None
    assert result.warnings == []


def test_valid_smiles_ibuprofen_passes() -> None:
    """(S)-Ibuprofen with specified stereochemistry passes all stages."""
    result = validate_molecule("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    assert result.is_valid is True
    assert result.rejection_stage is None
    assert result.warnings == []


def test_valid_smiles_caffeine_passes() -> None:
    """Caffeine passes all validation stages."""
    result = validate_molecule("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert result.is_valid is True
    assert result.rejection_stage is None
    assert result.warnings == []


# ---------------------------------------------------------------------------
# Rejection cases — each must assert on the specific rejection_stage
# ---------------------------------------------------------------------------


def test_reject_unparseable_syntax() -> None:
    """Broken SMILES that RDKit cannot parse is rejected at the syntax stage."""
    result = validate_molecule("C(C)(")
    assert result.is_valid is False
    assert result.rejection_stage == "syntax"


def test_reject_pentavalent_carbon_valence() -> None:
    """Pentavalent carbon (5 bonds) is rejected at the valence stage.

    C(C)(C)(C)(C)C has valid SMILES syntax but impossible chemistry —
    carbon cannot form 5 single bonds.
    """
    result = validate_molecule("C(C)(C)(C)(C)C")
    assert result.is_valid is False
    assert result.rejection_stage == "valence"


def test_reject_excessive_formal_charge() -> None:
    """Mg2+ ion (formal charge +2) is rejected at the charge stage.

    The default allowed range is [-1, +1].  Mg2+ is a valid, sanitisable
    molecule but its +2 charge exceeds the ceiling.
    """
    result = validate_molecule("[Mg+2]")
    assert result.is_valid is False
    assert result.rejection_stage == "charge"


# ---------------------------------------------------------------------------
# Edge cases — must assert on warnings list contents, not just is_valid
# ---------------------------------------------------------------------------


def test_edge_undefined_stereocentre_warns() -> None:
    """Fluorochlorobromomethane has an unspecified chiral centre.

    FC(Cl)Br has four different substituents on carbon (F, Cl, Br, H)
    but no @ or @@ stereochemistry annotation — should pass with a
    warning about the undefined stereocentre.
    """
    result = validate_molecule("FC(Cl)Br")
    assert result.is_valid is True
    assert result.rejection_stage is None
    assert len(result.warnings) >= 1
    stereo_warnings = [w for w in result.warnings if "stereocentre" in w]
    assert len(stereo_warnings) == 1
    assert "undefined" in stereo_warnings[0]


def test_edge_salt_strips_counterion() -> None:
    """Sodium phenoxide ([Na+].[O-]c1ccccc1) triggers salt stripping.

    The Na+ counterion is inorganic (no carbon) and gets stripped.
    The phenoxide fragment is kept.  Both charges are within the
    default [-1, +1] range so the charge stage does not reject.
    """
    result = validate_molecule("[Na+].[O-]c1ccccc1")
    assert result.is_valid is True
    assert result.rejection_stage is None
    salt_warnings = [w for w in result.warnings if "salt" in w]
    assert len(salt_warnings) == 1
    assert "stripped" in salt_warnings[0] or "removed" in salt_warnings[0]


def test_edge_allowed_negative_charge_passes() -> None:
    """Phenoxide ion ([O-]c1ccccc1) with formal charge -1 passes.

    -1 is within the default allowed range [-1, +1].  Single fragment,
    no stereocentres — should pass with no warnings.
    """
    result = validate_molecule("[O-]c1ccccc1")
    assert result.is_valid is True
    assert result.rejection_stage is None
    assert result.warnings == []
