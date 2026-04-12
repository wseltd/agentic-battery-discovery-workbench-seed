"""Tests for discovery_workbench.molecule — CanonicalMolecule dataclass."""

import dataclasses

import pytest

from discovery_workbench.evidence import EvidenceLevel
from discovery_workbench.molecule import CanonicalMolecule


# ---------------------------------------------------------------------------
# from_smiles — happy path
# ---------------------------------------------------------------------------

def test_from_smiles_canonical_round_trip():
    """Parsing non-canonical SMILES produces canonical output that round-trips."""
    mol = CanonicalMolecule.from_smiles("C(C)O", evidence_level=EvidenceLevel.GENERATED)
    reparsed = CanonicalMolecule.from_smiles(mol.canonical_smiles, evidence_level=EvidenceLevel.GENERATED)
    assert reparsed.canonical_smiles == mol.canonical_smiles
    assert reparsed.inchikey == mol.inchikey


def test_from_smiles_computes_inchikey():
    """InChIKey is 27 characters and starts with the expected prefix for ethanol."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    assert len(mol.inchikey) == 27
    # Ethanol InChIKey is well-known.
    assert mol.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_from_smiles_aromatic_canonicalisation():
    """Kekulé and aromatic input both canonicalise to the same SMILES."""
    aromatic = CanonicalMolecule.from_smiles("c1ccccc1", evidence_level="generated")
    kekule = CanonicalMolecule.from_smiles("C1=CC=CC=C1", evidence_level="generated")
    assert aromatic.canonical_smiles == kekule.canonical_smiles
    assert aromatic.inchikey == kekule.inchikey


def test_from_smiles_charged_molecule():
    """Charged molecules parse correctly and retain charge in canonical SMILES."""
    # Trimethylammonium — a charged molecule that is not a common salt fragment
    mol = CanonicalMolecule.from_smiles("C[NH+](C)C", evidence_level=EvidenceLevel.GENERATED)
    assert "+" in mol.canonical_smiles
    assert len(mol.inchikey) == 27


def test_evidence_level_stored():
    """Evidence level is stored and accessible."""
    mol = CanonicalMolecule.from_smiles("C", evidence_level=EvidenceLevel.ML_PREDICTED)
    assert mol.evidence_level is EvidenceLevel.ML_PREDICTED


def test_evidence_level_from_string():
    """Evidence level can be passed as a label string."""
    mol = CanonicalMolecule.from_smiles("C", evidence_level="heuristic_estimated")
    assert mol.evidence_level is EvidenceLevel.HEURISTIC_ESTIMATED


# ---------------------------------------------------------------------------
# from_smiles — salt stripping
# ---------------------------------------------------------------------------

def test_from_smiles_strips_salts():
    """Sodium chloride salt in aspirin SMILES is stripped to aspirin only."""
    # Aspirin as acetylsalicylic acid with NaCl salt
    aspirin_salt = "CC(=O)Oc1ccccc1C(=O)O.[Na+].[Cl-]"
    aspirin_pure = "CC(=O)Oc1ccccc1C(=O)O"
    mol_salt = CanonicalMolecule.from_smiles(aspirin_salt, evidence_level="generated")
    mol_pure = CanonicalMolecule.from_smiles(aspirin_pure, evidence_level="generated")
    assert mol_salt.canonical_smiles == mol_pure.canonical_smiles


# ---------------------------------------------------------------------------
# from_smiles — error cases
# ---------------------------------------------------------------------------

def test_from_smiles_invalid_smiles_raises():
    """Garbage SMILES raises ValueError with informative message."""
    with pytest.raises(ValueError, match="Cannot parse SMILES") as exc_info:
        CanonicalMolecule.from_smiles("not_a_molecule!!!", evidence_level="generated")
    assert "not_a_molecule" in str(exc_info.value)


def test_from_smiles_empty_smiles_raises():
    """Empty string raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        CanonicalMolecule.from_smiles("", evidence_level="generated")
    assert exc_info.value is not None


def test_from_smiles_bad_evidence_level_raises():
    """Invalid evidence level string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown evidence level") as exc_info:
        CanonicalMolecule.from_smiles("C", evidence_level="bogus")
    assert "bogus" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_immutability_after_construction():
    """Frozen dataclass prevents attribute assignment after construction."""
    mol = CanonicalMolecule.from_smiles("C", evidence_level="generated")
    original_smiles = mol.canonical_smiles
    with pytest.raises(dataclasses.FrozenInstanceError):
        mol.canonical_smiles = "CC"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mol.inchikey = "FAKE"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mol.evidence_level = EvidenceLevel.DFT_VERIFIED  # type: ignore[misc]
    assert mol.canonical_smiles == original_smiles


# ---------------------------------------------------------------------------
# Export — molblock / SDF
# ---------------------------------------------------------------------------

def test_to_molblock_returns_valid_molblock():
    """Molblock contains RDKit identifier and atom/bond counts."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    block = mol.to_molblock()
    assert "RDKit" in block
    assert "V2000" in block


def test_to_sdf_returns_string_with_terminator():
    """SDF output ends with the $$$$ record terminator."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    sdf = mol.to_sdf()
    assert sdf.rstrip().endswith("$$$$")
    # SDF includes the molblock content too
    assert "V2000" in sdf


# ---------------------------------------------------------------------------
# Export — XYZ / conformer
# ---------------------------------------------------------------------------

def test_to_xyz_raises_without_conformer():
    """to_xyz raises ValueError when no conformer has been embedded."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    with pytest.raises(ValueError, match="No 3-D conformer") as exc_info:
        mol.to_xyz()
    assert "embed_conformer" in str(exc_info.value)


def test_embed_conformer_enables_xyz_export():
    """After embed_conformer, to_xyz returns valid XYZ-format output."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    mol_with_conf = mol.embed_conformer()
    xyz = mol_with_conf.to_xyz()
    lines = xyz.strip().splitlines()
    # First line of XYZ is atom count
    atom_count = int(lines[0].strip())
    assert atom_count > 0
    # Coordinate lines follow after the comment line
    assert len(lines) >= atom_count + 2


def test_embed_conformer_returns_new_instance():
    """embed_conformer returns a new instance; original is unchanged."""
    mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
    mol_with_conf = mol.embed_conformer()
    assert mol_with_conf is not mol
    assert mol._mol.GetNumConformers() == 0
    assert mol_with_conf._mol.GetNumConformers() > 0
    # Metadata preserved
    assert mol_with_conf.canonical_smiles == mol.canonical_smiles
    assert mol_with_conf.inchikey == mol.inchikey
    assert mol_with_conf.evidence_level == mol.evidence_level
