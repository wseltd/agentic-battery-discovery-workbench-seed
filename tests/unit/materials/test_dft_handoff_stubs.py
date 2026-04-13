"""Tests for VASP parameter export and atomate2 workflow stub generation.

Covers export_vasp_params (dict-to-JSON serialisation) and the
BandStructureMaker addition to generate_atomate2_stub.
"""

import json

import pytest
from pymatgen.core import Lattice, Structure

from discovery_workbench.materials.dft_handoff import (
    VASP_DEFAULTS,
    export_vasp_params,
    generate_atomate2_stub,
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
def sample_vasp_params():
    """Pre-built VASP parameter dict for export tests."""
    return dict(VASP_DEFAULTS)


# ---------------------------------------------------------------------------
# export_vasp_params — happy path
# ---------------------------------------------------------------------------

def test_export_vasp_params_writes_json(sample_vasp_params, tmp_path):
    """Exported file must contain valid JSON."""
    path = export_vasp_params(sample_vasp_params, tmp_path, "cand_100")
    loaded = json.loads(path.read_text())
    assert isinstance(loaded, dict)


def test_export_vasp_params_returns_correct_path(sample_vasp_params, tmp_path):
    """Returned path must follow the {candidate_id}_vasp_params.json pattern."""
    path = export_vasp_params(sample_vasp_params, tmp_path, "cand_100")
    assert path == tmp_path / "cand_100_vasp_params.json"
    assert path.exists()


def test_export_vasp_params_preserves_all_keys(sample_vasp_params, tmp_path):
    """Every key in the input dict must appear in the output JSON."""
    path = export_vasp_params(sample_vasp_params, tmp_path, "cand_101")
    loaded = json.loads(path.read_text())
    for key in sample_vasp_params:
        assert key in loaded, f"Missing key: {key}"


def test_export_vasp_params_preserves_values(sample_vasp_params, tmp_path):
    """Values must survive the JSON roundtrip exactly."""
    path = export_vasp_params(sample_vasp_params, tmp_path, "cand_102")
    loaded = json.loads(path.read_text())
    assert loaded["ENCUT"] == 520
    assert loaded["ISPIN"] == 2
    assert loaded["KSPACING"] == pytest.approx(0.22)
    assert loaded["ALGO"] == "Normal"


def test_export_vasp_params_custom_dict(tmp_path):
    """Arbitrary dicts (not just VASP_DEFAULTS) must serialise correctly."""
    custom = {"ENCUT": 600, "NSW": 200, "IBRION": 2}
    path = export_vasp_params(custom, tmp_path, "custom_001")
    loaded = json.loads(path.read_text())
    assert loaded == custom


def test_export_vasp_params_with_magmom(tmp_path):
    """MAGMOM list values must survive JSON export."""
    params = {"ENCUT": 520, "MAGMOM": [5.0, 5.0, 0.0]}
    path = export_vasp_params(params, tmp_path, "mag_001")
    loaded = json.loads(path.read_text())
    assert loaded["MAGMOM"] == [5.0, 5.0, 0.0]


# ---------------------------------------------------------------------------
# export_vasp_params — error paths
# ---------------------------------------------------------------------------

def test_export_vasp_params_empty_candidate_id_raises(
    sample_vasp_params, tmp_path
):
    """Empty candidate_id must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        export_vasp_params(sample_vasp_params, tmp_path, "")


def test_export_vasp_params_nonexistent_dir_raises(sample_vasp_params, tmp_path):
    """Non-existent output_dir must raise ValueError."""
    bad_dir = tmp_path / "no_such_dir"
    with pytest.raises(ValueError, match="does not exist"):
        export_vasp_params(sample_vasp_params, bad_dir, "cand_999")


def test_export_vasp_params_empty_dict(tmp_path):
    """Empty dict must serialise to an empty JSON object — not an error."""
    path = export_vasp_params({}, tmp_path, "empty_001")
    loaded = json.loads(path.read_text())
    assert loaded == {}


# ---------------------------------------------------------------------------
# export_vasp_params — filename uniqueness
# ---------------------------------------------------------------------------

def test_export_vasp_params_distinct_candidates(sample_vasp_params, tmp_path):
    """Different candidate_ids must produce different files."""
    path_a = export_vasp_params(sample_vasp_params, tmp_path, "alpha")
    path_b = export_vasp_params(sample_vasp_params, tmp_path, "beta")
    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()


# ---------------------------------------------------------------------------
# generate_atomate2_stub — BandStructureMaker
# ---------------------------------------------------------------------------

def test_atomate2_stub_contains_band_structure_maker(nacl_structure, tmp_path):
    """Atomate2 stub must include a BandStructureMaker step."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    maker_names = [step["maker"] for step in stub["steps"]]
    assert "BandStructureMaker" in maker_names


def test_atomate2_stub_step_order(nacl_structure, tmp_path):
    """Steps must follow RelaxMaker -> StaticMaker -> BandStructureMaker."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    maker_names = [step["maker"] for step in stub["steps"]]
    assert maker_names == ["RelaxMaker", "StaticMaker", "BandStructureMaker"]


def test_atomate2_stub_band_structure_has_description(nacl_structure, tmp_path):
    """BandStructureMaker step must have a non-empty description."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    band_steps = [s for s in stub["steps"] if s["maker"] == "BandStructureMaker"]
    assert len(band_steps) == 1
    assert band_steps[0]["description"]


def test_atomate2_stub_composition_matches(nacl_structure, tmp_path):
    """Stub composition must match the input structure."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    assert stub["composition"] == nacl_structure.composition.reduced_formula


def test_atomate2_stub_file_matches_returned_dict(nacl_structure, tmp_path):
    """Written JSON file must match the returned dict exactly."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    loaded = json.loads(path.read_text())
    assert loaded == stub
