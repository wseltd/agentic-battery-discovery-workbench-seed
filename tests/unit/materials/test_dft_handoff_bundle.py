"""Tests for build_dft_bundle orchestrator and DFTHandoffBundle dataclass.

Focuses on the bundle builder's orchestration logic: file creation,
field population, warning collection, and input validation.
"""

import json
from dataclasses import fields

import pytest
from pymatgen.core import Lattice, Structure

from discovery_workbench.materials.dft_handoff import (
    VASP_DEFAULTS,
    build_dft_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def nacl_structure():
    """Simple NaCl structure for testing."""
    lattice = Lattice.cubic(5.64)
    return Structure(
        lattice,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture()
def fe_structure():
    """BCC iron structure — magnetic, no magmom site property."""
    lattice = Lattice.cubic(2.87)
    return Structure(
        lattice,
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )


# ---------------------------------------------------------------------------
# DFTHandoffBundle dataclass
# ---------------------------------------------------------------------------

def test_bundle_is_dataclass_with_correct_fields(nacl_structure, tmp_path):
    """DFTHandoffBundle must be a proper dataclass with expected field values."""
    bundle = build_dft_bundle("nacl_dc", nacl_structure, tmp_path)
    field_names = {f.name for f in fields(bundle)}
    assert "candidate_id" in field_names
    assert "evidence_level" in field_names
    # Value assertions — not just shape
    assert bundle.candidate_id == "nacl_dc"
    assert bundle.evidence_level == "ml_relaxed"
    assert bundle.vasp_parameters["ENCUT"] == 520


def test_bundle_candidate_id_propagated(nacl_structure, tmp_path):
    """candidate_id must be stored verbatim on the bundle."""
    bundle = build_dft_bundle("my-cand-42", nacl_structure, tmp_path)
    assert bundle.candidate_id == "my-cand-42"


def test_bundle_vasp_parameters_match_defaults(nacl_structure, tmp_path):
    """Bundle VASP parameters for non-magnetic structure must match defaults."""
    bundle = build_dft_bundle("vp_check", nacl_structure, tmp_path)
    for key, expected in VASP_DEFAULTS.items():
        assert bundle.vasp_parameters[key] == expected, (
            f"VASP param {key}: expected {expected}, got {bundle.vasp_parameters[key]}"
        )


def test_bundle_vasp_params_no_internal_keys(nacl_structure, tmp_path):
    """Bundle vasp_parameters must not contain internal underscore keys."""
    bundle = build_dft_bundle("clean_keys", nacl_structure, tmp_path)
    internal = [k for k in bundle.vasp_parameters if k.startswith("_")]
    assert internal == []


# ---------------------------------------------------------------------------
# build_dft_bundle — file creation
# ---------------------------------------------------------------------------

def test_build_creates_four_files(nacl_structure, tmp_path):
    """All four output files (CIF, POSCAR, params, stub) must exist."""
    bundle = build_dft_bundle("quad", nacl_structure, tmp_path)
    assert bundle.cif_path.exists()
    assert bundle.poscar_path.exists()
    assert bundle.vasp_params_path.exists()
    assert bundle.atomate2_stub_path.exists()


def test_build_cif_content_matches_structure(nacl_structure, tmp_path):
    """CIF written by bundle must be parseable and preserve composition."""
    bundle = build_dft_bundle("cif_val", nacl_structure, tmp_path)
    parsed = Structure.from_file(str(bundle.cif_path))
    assert parsed.composition.reduced_formula == "NaCl"


def test_build_poscar_content_matches_structure(nacl_structure, tmp_path):
    """POSCAR written by bundle must preserve lattice parameter a."""
    bundle = build_dft_bundle("pos_val", nacl_structure, tmp_path)
    parsed = Structure.from_file(str(bundle.poscar_path))
    assert abs(parsed.lattice.a - 5.64) < 0.01


def test_build_vasp_params_file_valid_json(nacl_structure, tmp_path):
    """VASP params file must contain valid JSON with ENCUT=520."""
    bundle = build_dft_bundle("json_val", nacl_structure, tmp_path)
    loaded = json.loads(bundle.vasp_params_path.read_text())
    assert loaded["ENCUT"] == 520
    assert loaded["ISPIN"] == 2


def test_build_atomate2_stub_file_valid(nacl_structure, tmp_path):
    """Atomate2 stub file must contain expected workflow structure."""
    bundle = build_dft_bundle("stub_val", nacl_structure, tmp_path)
    loaded = json.loads(bundle.atomate2_stub_path.read_text())
    assert loaded["workflow"] == "DFT_verification"
    makers = [s["maker"] for s in loaded["steps"]]
    assert "RelaxMaker" in makers
    assert "StaticMaker" in makers


# ---------------------------------------------------------------------------
# build_dft_bundle — warnings
# ---------------------------------------------------------------------------

def test_bundle_magnetic_element_warning_content(fe_structure, tmp_path):
    """Bundle for Fe without magmom must warn mentioning Fe and MAGMOM."""
    bundle = build_dft_bundle("fe_warn", fe_structure, tmp_path)
    assert len(bundle.warnings) >= 1
    warning_text = " ".join(bundle.warnings)
    assert "Fe" in warning_text
    assert "MAGMOM" in warning_text


def test_bundle_nonmagnetic_no_warnings(nacl_structure, tmp_path):
    """Non-magnetic structure must produce zero warnings."""
    bundle = build_dft_bundle("nacl_nw", nacl_structure, tmp_path)
    assert bundle.warnings == []


# ---------------------------------------------------------------------------
# build_dft_bundle — input validation
# ---------------------------------------------------------------------------

def test_nonexistent_dir_raises(nacl_structure, tmp_path):
    """Non-existent output_dir must raise ValueError naming the directory."""
    bad_dir = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="does not exist") as exc_info:
        build_dft_bundle("x", nacl_structure, bad_dir)
    assert "does_not_exist" in str(exc_info.value)


def test_empty_candidate_id_raises(nacl_structure, tmp_path):
    """Empty candidate_id must raise ValueError with actionable message."""
    with pytest.raises(ValueError, match="non-empty string") as exc_info:
        build_dft_bundle("", nacl_structure, tmp_path)
    assert "candidate_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------

def test_module_all_exports():
    """__all__ must export the three public names."""
    from discovery_workbench.materials import dft_handoff
    assert "build_dft_bundle" in dft_handoff.__all__
    assert "DFTHandoffBundle" in dft_handoff.__all__
    assert "default_vasp_parameters" in dft_handoff.__all__
