"""Tests for DFT handoff bundle generation."""

import json

import pytest
from pymatgen.core import Lattice, Structure

from agentic_materials_discovery.handoff.dft_handoff import (
    VASP_DEFAULTS,
    build_dft_bundle,
    default_vasp_parameters,
    export_cif,
    export_poscar,
    generate_atomate2_stub,
    generate_vasp_params,
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
    """BCC iron structure — magnetic, for MAGMOM tests."""
    lattice = Lattice.cubic(2.87)
    return Structure(
        lattice,
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )


@pytest.fixture()
def fe_structure_with_magmom():
    """BCC iron with explicit magmom site property."""
    lattice = Lattice.cubic(2.87)
    struct = Structure(
        lattice,
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    struct.add_site_property("magmom", [5.0])
    return struct


# ---------------------------------------------------------------------------
# build_dft_bundle
# ---------------------------------------------------------------------------

def test_build_dft_bundle_creates_all_files(nacl_structure, tmp_path):
    """All four output files must exist after build."""
    bundle = build_dft_bundle("cand_001", nacl_structure, tmp_path)
    assert bundle.cif_path.exists()
    assert bundle.poscar_path.exists()
    assert bundle.vasp_params_path.exists()
    assert bundle.atomate2_stub_path.exists()


def test_build_dft_bundle_nonexistent_dir_raises(nacl_structure, tmp_path):
    """build_dft_bundle must reject a directory that does not exist."""
    bad_dir = tmp_path / "nonexistent"
    with pytest.raises(ValueError, match="does not exist"):
        build_dft_bundle("cand_001", nacl_structure, bad_dir)
    assert not bad_dir.exists(), "Directory should not have been created"


def test_build_dft_bundle_empty_candidate_id_raises(nacl_structure, tmp_path):
    """Empty candidate_id must be rejected with a clear error."""
    with pytest.raises(ValueError, match="non-empty string"):
        build_dft_bundle("", nacl_structure, tmp_path)
    # Verify no files were written before the error
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# export_cif
# ---------------------------------------------------------------------------

def test_export_cif_valid_structure(nacl_structure, tmp_path):
    """CIF file must be written and non-empty."""
    path = tmp_path / "test.cif"
    result = export_cif(nacl_structure, path)
    assert result == path
    assert path.stat().st_size > 0


def test_export_cif_preserves_composition(nacl_structure, tmp_path):
    """CIF content must mention the elements present."""
    path = tmp_path / "test.cif"
    export_cif(nacl_structure, path)
    content = path.read_text()
    assert "Na" in content
    assert "Cl" in content


def test_export_cif_roundtrip_parseable(nacl_structure, tmp_path):
    """Exported CIF must be parseable back into a Structure."""
    path = tmp_path / "roundtrip.cif"
    export_cif(nacl_structure, path)
    parsed = Structure.from_file(str(path))
    assert len(parsed) == len(nacl_structure)
    original_formula = nacl_structure.composition.reduced_formula
    parsed_formula = parsed.composition.reduced_formula
    assert parsed_formula == original_formula


# ---------------------------------------------------------------------------
# export_poscar
# ---------------------------------------------------------------------------

def test_export_poscar_valid_structure(nacl_structure, tmp_path):
    """POSCAR file must be written and non-empty."""
    path = tmp_path / "POSCAR"
    result = export_poscar(nacl_structure, path)
    assert result == path
    assert path.stat().st_size > 0


def test_export_poscar_preserves_lattice(nacl_structure, tmp_path):
    """POSCAR roundtrip must preserve lattice parameters within tolerance."""
    path = tmp_path / "POSCAR"
    export_poscar(nacl_structure, path)
    parsed = Structure.from_file(str(path))
    for orig, roundtripped in zip(
        nacl_structure.lattice.abc, parsed.lattice.abc
    ):
        assert abs(orig - roundtripped) < 0.01, (
            f"Lattice parameter drift: {orig} vs {roundtripped}"
        )


# ---------------------------------------------------------------------------
# generate_vasp_params
# ---------------------------------------------------------------------------

def test_vasp_params_contains_required_keys(nacl_structure, tmp_path):
    """VASP param file must contain all default INCAR keys."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    for key in VASP_DEFAULTS:
        assert key in params, f"Missing VASP key: {key}"


def test_vasp_params_encut_value(nacl_structure, tmp_path):
    """ENCUT must be 520 eV."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    assert params["ENCUT"] == 520


def test_vasp_params_ispin_enabled(nacl_structure, tmp_path):
    """ISPIN must be 2 (spin-polarised)."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    assert params["ISPIN"] == 2


def test_vasp_params_kspacing_value(nacl_structure, tmp_path):
    """KSPACING must be 0.22."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    assert params["KSPACING"] == pytest.approx(0.22)


def test_vasp_params_magmom_from_site_property(fe_structure_with_magmom, tmp_path):
    """MAGMOM must be picked up from site_properties when present."""
    path = tmp_path / "params.json"
    generate_vasp_params(fe_structure_with_magmom, path)
    params = json.loads(path.read_text())
    assert "MAGMOM" in params
    assert params["MAGMOM"] == [5.0]


def test_vasp_params_no_magmom_for_nonmagnetic(nacl_structure, tmp_path):
    """Non-magnetic structures should not get a MAGMOM key."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    assert "MAGMOM" not in params


def test_vasp_params_json_no_internal_keys(nacl_structure, tmp_path):
    """Internal keys like _warnings must not leak into the JSON file."""
    path = tmp_path / "params.json"
    generate_vasp_params(nacl_structure, path)
    params = json.loads(path.read_text())
    internal_keys = [k for k in params if k.startswith("_")]
    assert internal_keys == [], f"Internal keys in JSON: {internal_keys}"


# ---------------------------------------------------------------------------
# generate_atomate2_stub
# ---------------------------------------------------------------------------

def test_atomate2_stub_contains_relax_maker(nacl_structure, tmp_path):
    """Atomate2 stub must include a RelaxMaker step."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    maker_names = [step["maker"] for step in stub["steps"]]
    assert "RelaxMaker" in maker_names


def test_atomate2_stub_contains_static_maker(nacl_structure, tmp_path):
    """Atomate2 stub must include a StaticMaker step."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    maker_names = [step["maker"] for step in stub["steps"]]
    assert "StaticMaker" in maker_names


def test_atomate2_stub_json_roundtrip(nacl_structure, tmp_path):
    """Stub file must be valid JSON matching the returned dict."""
    path = tmp_path / "stub.json"
    stub = generate_atomate2_stub(nacl_structure, path)
    loaded = json.loads(path.read_text())
    assert loaded == stub


# ---------------------------------------------------------------------------
# Bundle-level properties
# ---------------------------------------------------------------------------

def test_bundle_evidence_level_ml_relaxed(nacl_structure, tmp_path):
    """Bundle evidence_level must be 'ml_relaxed' for ML-relaxed inputs."""
    bundle = build_dft_bundle("cand_002", nacl_structure, tmp_path)
    assert bundle.evidence_level == "ml_relaxed"


def test_bundle_warnings_for_magnetic_elements(fe_structure, tmp_path):
    """Bundle must warn when magnetic elements lack MAGMOM site property."""
    bundle = build_dft_bundle("fe_001", fe_structure, tmp_path)
    assert len(bundle.warnings) > 0
    joined = " ".join(bundle.warnings)
    assert "Fe" in joined
    assert "MAGMOM" in joined


def test_bundle_no_warnings_for_nonmagnetic(nacl_structure, tmp_path):
    """Non-magnetic structures should produce no warnings."""
    bundle = build_dft_bundle("nacl_001", nacl_structure, tmp_path)
    assert bundle.warnings == []


def test_bundle_warnings_with_magmom_present(fe_structure_with_magmom, tmp_path):
    """Magnetic elements WITH magmom should warn to verify, not complain about absence."""
    bundle = build_dft_bundle("fe_002", fe_structure_with_magmom, tmp_path)
    joined = " ".join(bundle.warnings)
    assert "verify" in joined.lower() or "MAGMOM" in joined


# ---------------------------------------------------------------------------
# default_vasp_parameters isolation
# ---------------------------------------------------------------------------

def test_default_vasp_parameters_returns_copy():
    """Mutating the returned dict must not change the module defaults."""
    params = default_vasp_parameters()
    params["ENCUT"] = 9999
    fresh = default_vasp_parameters()
    assert fresh["ENCUT"] == 520, "Module defaults were mutated"


def test_default_vasp_parameters_matches_constants():
    """Returned dict must match VASP_DEFAULTS exactly."""
    params = default_vasp_parameters()
    assert params == VASP_DEFAULTS
