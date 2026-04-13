"""DFT handoff bundle for inorganic materials candidates.

Packages a relaxed structure into CIF, POSCAR, VASP parameter, and
atomate2 workflow stub files for downstream DFT verification.  All
VASP parameters are heuristic defaults suitable for a first-pass
relaxation; production runs should be tuned per-system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VASP parameter constants
# ---------------------------------------------------------------------------

# Electronic minimisation algorithm.
VASP_ALGO = "Normal"

# Total energy convergence criterion in eV.
VASP_EDIFF = 1e-6

# Plane-wave cutoff in eV — 520 covers most PAW potentials safely.
VASP_ENCUT = 520

# Smearing method: Gaussian (0) is safest for unknown band character.
VASP_ISMEAR = 0

# Smearing width in eV — 0.05 is conservative for semiconductors.
VASP_SIGMA = 0.05

# Spin-polarised calculation — always on for generality.
VASP_ISPIN = 2

# K-point spacing in reciprocal angstroms — 0.22 balances cost and accuracy.
VASP_KSPACING = 0.22

# Real-space projection: 'Auto' lets VASP choose based on system size.
VASP_LREAL = "Auto"

VASP_DEFAULTS: dict[str, object] = {
    "ALGO": VASP_ALGO,
    "EDIFF": VASP_EDIFF,
    "ENCUT": VASP_ENCUT,
    "ISMEAR": VASP_ISMEAR,
    "SIGMA": VASP_SIGMA,
    "ISPIN": VASP_ISPIN,
    "KSPACING": VASP_KSPACING,
    "LREAL": VASP_LREAL,
}

# Elements with commonly non-zero magnetic moments (3d, 4f series).
# Used to warn when MAGMOM should be set but site properties are missing.
MAGNETIC_ELEMENTS = frozenset({
    "Mn", "Fe", "Co", "Ni", "Cr", "V", "Ti", "Cu",
    "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DFTHandoffBundle:
    """Artifacts needed to launch a DFT verification calculation.

    Args:
        candidate_id: Unique identifier for this candidate.
        cif_path: Path to the exported CIF file.
        poscar_path: Path to the exported POSCAR file.
        vasp_params_path: Path to the VASP parameter JSON file.
        atomate2_stub_path: Path to the atomate2 workflow stub JSON.
        vasp_parameters: VASP INCAR parameters used for this candidate.
        evidence_level: Credibility tier of the input structure.
        warnings: Diagnostic messages for the user.
    """

    candidate_id: str
    cif_path: Path
    poscar_path: Path
    vasp_params_path: Path
    atomate2_stub_path: Path
    vasp_parameters: dict
    evidence_level: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VASP parameter helpers
# ---------------------------------------------------------------------------

def default_vasp_parameters() -> dict:
    """Return a fresh copy of the default VASP parameter dict.

    Returns:
        Dictionary of VASP INCAR parameters with conservative defaults.
    """
    return dict(VASP_DEFAULTS)


def vasp_parameters_with_magmom(structure: Structure) -> dict:
    """Return VASP parameters augmented with MAGMOM from site properties.

    If the structure's site_properties contain 'magmom', those values are
    included as the MAGMOM list.  If magmom is absent but the structure
    contains magnetic elements, a warning string is appended to the
    returned dict under the '_warnings' key.

    Chose to store warnings in the return dict under a private key rather
    than a side channel because callers always consume the dict immediately
    and need both parameters and warnings together.

    Args:
        structure: pymatgen Structure with optional 'magmom' site property.

    Returns:
        VASP parameter dict, possibly augmented with MAGMOM and _warnings.
    """
    params = default_vasp_parameters()
    warnings: list[str] = []

    if "magmom" in structure.site_properties:
        params["MAGMOM"] = list(structure.site_properties["magmom"])
    else:
        elements_present = {str(site.specie) for site in structure}
        magnetic_present = elements_present & MAGNETIC_ELEMENTS
        if magnetic_present:
            warnings.append(
                f"Structure contains magnetic elements "
                f"{sorted(magnetic_present)} but no MAGMOM site property — "
                f"VASP will use default initial moments"
            )

    params["_warnings"] = warnings
    return params


# ---------------------------------------------------------------------------
# File export functions
# ---------------------------------------------------------------------------

def export_cif(structure: Structure, path: Path) -> Path:
    """Write a CIF file for the given structure.

    Args:
        structure: pymatgen Structure to export.
        path: Destination file path.

    Returns:
        The path written to.

    Raises:
        ValueError: If structure has no sites.
    """
    if len(structure) == 0:
        raise ValueError("Cannot export CIF for structure with no sites")
    logger.info("Exporting CIF to %s", path)
    writer = CifWriter(structure)
    writer.write_file(str(path))
    return path


def export_poscar(structure: Structure, path: Path) -> Path:
    """Write a POSCAR file for the given structure.

    Args:
        structure: pymatgen Structure to export.
        path: Destination file path.

    Returns:
        The path written to.

    Raises:
        ValueError: If structure has no sites.
    """
    if len(structure) == 0:
        raise ValueError("Cannot export POSCAR for structure with no sites")
    logger.info("Exporting POSCAR to %s", path)
    poscar = Poscar(structure)
    poscar.write_file(str(path))
    return path


def generate_vasp_params(structure: Structure, path: Path) -> dict:
    """Generate VASP parameters and write them to a JSON file.

    Args:
        structure: pymatgen Structure (used for MAGMOM detection).
        path: Destination JSON file path.

    Returns:
        The VASP parameter dict (including any MAGMOM).
    """
    logger.info("Generating VASP parameters to %s", path)
    params = vasp_parameters_with_magmom(structure)
    warnings = params.pop("_warnings", [])

    # Write the clean parameter dict (no _warnings key in the JSON)
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)

    # Re-attach warnings for the caller
    params["_warnings"] = warnings
    return params


def generate_atomate2_stub(structure: Structure, path: Path) -> dict:
    """Generate an atomate2 workflow stub and write it to a JSON file.

    The stub defines a two-step workflow: RelaxMaker followed by
    StaticMaker.  This is a serialised recipe, not executable code —
    actual execution requires atomate2 and jobflow installed.

    Args:
        structure: pymatgen Structure for the workflow.
        path: Destination JSON file path.

    Returns:
        The workflow stub dict.
    """
    logger.info("Generating atomate2 stub to %s", path)
    stub = {
        "workflow": "DFT_verification",
        "steps": [
            {
                "maker": "RelaxMaker",
                "description": "Full ionic relaxation with ISIF=3",
                "vasp_input_set": "MPRelaxSet",
            },
            {
                "maker": "StaticMaker",
                "description": "Single-point energy on relaxed geometry",
                "vasp_input_set": "MPStaticSet",
            },
        ],
        "composition": structure.composition.reduced_formula,
        "num_sites": len(structure),
    }
    with open(path, "w") as fh:
        json.dump(stub, fh, indent=2)
    return stub


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

def build_dft_bundle(
    candidate_id: str,
    structure: Structure,
    output_dir: Path,
) -> DFTHandoffBundle:
    """Build a complete DFT handoff bundle for a candidate structure.

    Creates CIF, POSCAR, VASP parameter, and atomate2 stub files in
    output_dir and returns a DFTHandoffBundle referencing them.

    Args:
        candidate_id: Unique identifier for this candidate.
        structure: Relaxed pymatgen Structure.
        output_dir: Directory to write output files into (must exist).

    Returns:
        DFTHandoffBundle with paths to all generated files.

    Raises:
        ValueError: If candidate_id is empty or output_dir does not exist.
    """
    if not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    if not output_dir.is_dir():
        raise ValueError(
            f"output_dir does not exist or is not a directory: {output_dir}"
        )

    logger.info(
        "Building DFT bundle candidate_id=%s output_dir=%s",
        candidate_id, output_dir,
    )

    cif_path = output_dir / f"{candidate_id}.cif"
    poscar_path = output_dir / f"{candidate_id}_POSCAR"
    vasp_params_path = output_dir / f"{candidate_id}_vasp_params.json"
    atomate2_stub_path = output_dir / f"{candidate_id}_atomate2_stub.json"

    export_cif(structure, cif_path)
    export_poscar(structure, poscar_path)

    params_result = generate_vasp_params(structure, vasp_params_path)
    warnings = list(params_result.pop("_warnings", []))

    generate_atomate2_stub(structure, atomate2_stub_path)

    # Check for magnetic elements to add handoff warnings
    elements_present = {str(site.specie) for site in structure}
    magnetic_present = elements_present & MAGNETIC_ELEMENTS
    if magnetic_present:
        if "magmom" not in structure.site_properties:
            # Warning already captured in params_result warnings above
            pass
        else:
            warnings.append(
                f"Structure contains magnetic elements "
                f"{sorted(magnetic_present)} — verify MAGMOM values "
                f"are physical before launching DFT"
            )

    return DFTHandoffBundle(
        candidate_id=candidate_id,
        cif_path=cif_path,
        poscar_path=poscar_path,
        vasp_params_path=vasp_params_path,
        atomate2_stub_path=atomate2_stub_path,
        vasp_parameters={
            k: v for k, v in params_result.items() if not k.startswith("_")
        },
        evidence_level="ml_relaxed",
        warnings=warnings,
    )
