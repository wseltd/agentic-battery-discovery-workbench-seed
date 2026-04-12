"""Tests for xTB handoff bundle generation."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, inchi

from agentic_discovery.molecules.xtb_handoff import (
    ConformerGenerationError,
    HandoffBundle,
    XtbHandoffBuilder,
    build_bundle,
)
from agentic_discovery.shared.evidence import EvidenceLevel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ETHANOL_SMILES = "CCO"
_BENZENE_SMILES = "c1ccccc1"


@pytest.fixture()
def ethanol_mol() -> Chem.Mol:
    return Chem.MolFromSmiles(_ETHANOL_SMILES)


@pytest.fixture()
def benzene_mol() -> Chem.Mol:
    return Chem.MolFromSmiles(_BENZENE_SMILES)


@pytest.fixture()
def builder() -> XtbHandoffBuilder:
    return XtbHandoffBuilder()


@pytest.fixture()
def ethanol_bundle(builder: XtbHandoffBuilder, ethanol_mol: Chem.Mol) -> HandoffBundle:
    return builder.build_bundle(ethanol_mol)


@pytest.fixture()
def benzene_bundle(builder: XtbHandoffBuilder, benzene_mol: Chem.Mol) -> HandoffBundle:
    return builder.build_bundle(benzene_mol)


# ---------------------------------------------------------------------------
# XYZ format tests
# ---------------------------------------------------------------------------


class TestXyzFormat:
    """XYZ content must follow the standard: atom-count, blank, coord lines."""

    def test_build_bundle_returns_valid_xyz_format(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        lines = ethanol_bundle.xyz_content.strip().splitlines()
        # Line 1 must be a positive integer (atom count).
        assert lines[0].strip().isdigit()
        atom_count = int(lines[0])
        assert atom_count > 0
        # Line 2 is a comment/blank line.
        # Remaining lines are coordinate lines.
        assert len(lines) == atom_count + 2

    def test_xyz_first_line_is_atom_count(
        self, ethanol_bundle: HandoffBundle, ethanol_mol: Chem.Mol
    ) -> None:
        first_line = ethanol_bundle.xyz_content.splitlines()[0].strip()
        mol_h = Chem.AddHs(ethanol_mol)
        expected_count = mol_h.GetNumAtoms()
        assert int(first_line) == expected_count

    def test_xyz_coord_lines_match_atom_count(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        lines = ethanol_bundle.xyz_content.strip().splitlines()
        atom_count = int(lines[0])
        coord_lines = lines[2:]  # skip count + comment
        assert len(coord_lines) == atom_count
        for line in coord_lines:
            parts = line.split()
            assert len(parts) == 4, f"Expected 'symbol x y z', got: {line!r}"
            # Coordinates must be floats.
            for coord in parts[1:]:
                float(coord)  # raises ValueError on failure


# ---------------------------------------------------------------------------
# SDF tests
# ---------------------------------------------------------------------------


class TestSdfFormat:
    def test_build_bundle_returns_valid_sdf(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        # SDF mol-block must be parseable back into an RDKit Mol.
        mol = Chem.MolFromMolBlock(ethanol_bundle.sdf_content, removeHs=False)
        assert mol is not None, "SDF content could not be parsed back into a Mol"
        assert mol.GetNumAtoms() > 0


# ---------------------------------------------------------------------------
# Run-script tests
# ---------------------------------------------------------------------------


class TestRunScript:
    def test_run_script_contains_charge_placeholder(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        assert "CHARGE=" in ethanol_bundle.run_script

    def test_run_script_contains_multiplicity_placeholder(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        assert "MULTIPLICITY=" in ethanol_bundle.run_script

    def test_run_script_contains_user_confirm_comment(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        # The script must warn the user to verify charge/multiplicity.
        assert "confirm" in ethanol_bundle.run_script.lower()

    def test_run_script_is_valid_bash_syntax(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        # bash -n performs syntax checking without execution.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(ethanol_bundle.run_script)
            f.flush()
            result = subprocess.run(
                ["bash", "-n", f.name],
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"


# ---------------------------------------------------------------------------
# Evidence level
# ---------------------------------------------------------------------------


class TestEvidenceLevel:
    def test_evidence_level_is_generated(
        self, ethanol_bundle: HandoffBundle
    ) -> None:
        assert ethanol_bundle.evidence_level is EvidenceLevel.GENERATED


# ---------------------------------------------------------------------------
# Identity / SMILES tests
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_inchikey_matches_input_mol(
        self, ethanol_bundle: HandoffBundle, ethanol_mol: Chem.Mol
    ) -> None:
        expected = inchi.MolToInchiKey(ethanol_mol)
        assert ethanol_bundle.inchikey == expected

    def test_smiles_in_bundle_matches_input(
        self, ethanol_bundle: HandoffBundle, ethanol_mol: Chem.Mol
    ) -> None:
        expected = Chem.MolToSmiles(ethanol_mol)
        assert ethanol_bundle.smiles == expected


# ---------------------------------------------------------------------------
# Conformer fallback / error tests
# ---------------------------------------------------------------------------


class TestConformerFallback:
    """ETKDG failure should retry with random coords; both failing raises."""

    def test_etkdg_failure_retries_with_random_coords(
        self, builder: XtbHandoffBuilder, ethanol_mol: Chem.Mol
    ) -> None:
        call_count = 0
        original_embed = AllChem.EmbedMolecule

        def mock_embed(mol, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate ETKDG failure.
                return -1
            return original_embed(mol, *args, **kwargs)

        with patch(
            "agentic_discovery.molecules.xtb_handoff.AllChem.EmbedMolecule",
            side_effect=mock_embed,
        ):
            bundle = builder.build_bundle(ethanol_mol)

        # Must have called embed twice (ETKDG + random fallback).
        assert call_count == 2
        assert bundle.xyz_content  # got a valid result

    def test_both_embed_attempts_fail_raises_conformer_error(
        self, builder: XtbHandoffBuilder, ethanol_mol: Chem.Mol
    ) -> None:
        with patch(
            "agentic_discovery.molecules.xtb_handoff.AllChem.EmbedMolecule",
            return_value=-1,
        ):
            with pytest.raises(ConformerGenerationError):
                builder.build_bundle(ethanol_mol)

    def test_retry_adds_warning(
        self, builder: XtbHandoffBuilder, ethanol_mol: Chem.Mol
    ) -> None:
        call_count = 0
        original_embed = AllChem.EmbedMolecule

        def mock_embed(mol, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return -1
            return original_embed(mol, *args, **kwargs)

        with patch(
            "agentic_discovery.molecules.xtb_handoff.AllChem.EmbedMolecule",
            side_effect=mock_embed,
        ):
            bundle = builder.build_bundle(ethanol_mol)

        assert any("ETKDG" in w for w in bundle.warnings)


# ---------------------------------------------------------------------------
# Benzene (aromatic ring) smoke test
# ---------------------------------------------------------------------------


class TestBenzene:
    def test_simple_molecule_benzene(
        self, benzene_bundle: HandoffBundle
    ) -> None:
        lines = benzene_bundle.xyz_content.strip().splitlines()
        atom_count = int(lines[0])
        # Benzene + hydrogens = 12 atoms.
        assert atom_count == 12
        assert benzene_bundle.smiles  # non-empty
        assert benzene_bundle.inchikey  # non-empty
