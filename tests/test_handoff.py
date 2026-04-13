"""Tests for xTB and DFT handoff bundle generation.

Verifies that handoff functions produce correct bundle structures
and assign the right evidence levels -- the core scientific-honesty
gate from research-pack Q31.
"""

from amdw.materials.dft_handoff import DftHandoff, prepare_dft_handoff
from amdw.molecules.xtb_handoff import XtbHandoff, prepare_xtb_handoff
from amdw.shared.evidence import EvidenceLevel

# ---------------------------------------------------------------------------
# Synthetic test data -- no real generator output needed.
# ---------------------------------------------------------------------------

_FAKE_MOLECULE = {
    "smiles": "CCO",
    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    "conformer_xyz": (
        "3\n"
        "ethanol\n"
        "C   0.000000   0.000000   0.000000\n"
        "C   1.540000   0.000000   0.000000\n"
        "O   2.100000   1.200000   0.000000\n"
    ),
    "charge": 0,
    "multiplicity": 1,
}

_FAKE_CRYSTAL = {
    "composition": "NaCl",
    "lattice_params": {"a": 5.64, "b": 5.64, "c": 5.64},
    "species": ["Na", "Cl"],
    "cart_coords": [
        [0.0, 0.0, 0.0],
        [2.82, 2.82, 2.82],
    ],
}


# ---------------------------------------------------------------------------
# Test 1: xTB handoff bundle content
# ---------------------------------------------------------------------------


def test_xtb_handoff_generates_xyz_and_script() -> None:
    """xTB handoff must produce non-empty XYZ content and a run script invoking xtb."""
    result = prepare_xtb_handoff(_FAKE_MOLECULE)

    assert isinstance(result, XtbHandoff)

    # XYZ content must be the conformer geometry, non-empty.
    assert isinstance(result.xyz_content, str)
    assert len(result.xyz_content) > 0
    assert result.xyz_content == _FAKE_MOLECULE["conformer_xyz"]

    # Run script must mention the xtb binary.
    assert isinstance(result.run_script, str)
    assert "xtb" in result.run_script

    # Charge and multiplicity must be integers matching input.
    assert isinstance(result.charge, int)
    assert result.charge == 0
    assert isinstance(result.multiplicity, int)
    assert result.multiplicity == 1

    # Evidence level must be semiempirical QC -- not generated, not DFT.
    assert result.evidence_level is EvidenceLevel.SEMIEMPIRICAL_QC


# ---------------------------------------------------------------------------
# Test 2: DFT handoff bundle content
# ---------------------------------------------------------------------------


def test_dft_handoff_generates_structure_and_incar() -> None:
    """DFT handoff must produce structure content, INCAR with ENCUT/ISPIN, and KPOINTS."""
    result = prepare_dft_handoff(_FAKE_CRYSTAL)

    assert isinstance(result, DftHandoff)

    # Structure file must be non-empty POSCAR-like content.
    assert isinstance(result.structure_file, str)
    assert len(result.structure_file) > 0
    assert "NaCl" in result.structure_file

    # INCAR must contain the two key parameters.
    assert isinstance(result.incar_content, str)
    assert "ENCUT" in result.incar_content
    assert "ISPIN" in result.incar_content

    # KPOINTS must be non-empty.
    assert isinstance(result.kpoints_content, str)
    assert len(result.kpoints_content) > 0

    # Evidence level must be DFT-verified.
    assert result.evidence_level is EvidenceLevel.DFT_VERIFIED


# ---------------------------------------------------------------------------
# Test 3: Evidence level scientific-honesty gate
# ---------------------------------------------------------------------------


def test_evidence_level_labels_correct() -> None:
    """Scientific-honesty gate: xTB claims semiempirical_qc, DFT claims dft_verified.

    xTB handoff must NOT claim dft_verified -- that would overstate
    the credibility of a semiempirical method.  DFT handoff must NOT
    claim ml_relaxed -- the handoff marks intent to run DFT, which is
    the highest computational evidence tier before experimental.
    """
    xtb_result = prepare_xtb_handoff(_FAKE_MOLECULE)
    dft_result = prepare_dft_handoff(_FAKE_CRYSTAL)

    # Positive: correct evidence levels assigned.
    assert xtb_result.evidence_level is EvidenceLevel.SEMIEMPIRICAL_QC
    assert dft_result.evidence_level is EvidenceLevel.DFT_VERIFIED

    # Negative: evidence levels must not be confused across domains.
    assert xtb_result.evidence_level is not EvidenceLevel.DFT_VERIFIED
    assert dft_result.evidence_level is not EvidenceLevel.ML_RELAXED
    assert xtb_result.evidence_level is not EvidenceLevel.ML_RELAXED

    # Ordering: DFT evidence is strictly more credible than semiempirical.
    assert xtb_result.evidence_level < dft_result.evidence_level
