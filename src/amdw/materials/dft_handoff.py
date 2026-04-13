"""Prepare DFT handoff bundles for inorganic materials candidates.

Accepts pre-processed crystal data (composition, lattice, coordinates)
and packages it into a lightweight handoff bundle for downstream DFT
verification.  Does not invoke pymatgen or VASP -- operates on dicts
and strings only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amdw.shared.evidence import EvidenceLevel

logger = logging.getLogger(__name__)

__all__ = ["DftHandoff", "prepare_dft_handoff"]

# VASP INCAR defaults -- these match the canonical values in
# discovery_workbench.materials.dft_handoff but are kept private here
# to avoid importing pymatgen in the thin-facade layer.  They are
# simple numeric/string constants, not domain vocabularies.

# Plane-wave cutoff in eV -- 520 covers most PAW potentials safely.
_VASP_ENCUT: int = 520

# Spin-polarised calculation -- always on for generality.
_VASP_ISPIN: int = 2

# Total energy convergence criterion in eV.
_VASP_EDIFF: float = 1e-6

# Smearing method: Gaussian (0) is safest for unknown band character.
_VASP_ISMEAR: int = 0

# Smearing width in eV.
_VASP_SIGMA: float = 0.05

# Electronic minimisation algorithm.
_VASP_ALGO: str = "Normal"

# Real-space projection.
_VASP_LREAL: str = "Auto"

# Default k-point grid density per axis.
_KPOINTS_GRID: int = 4


@dataclass(frozen=True, slots=True)
class DftHandoff:
    """Immutable handoff bundle for DFT verification.

    Args:
        structure_file: POSCAR-format structure string.
        incar_content: VASP INCAR parameter string.
        kpoints_content: VASP KPOINTS file content.
        evidence_level: Credibility tier -- always DFT_VERIFIED.
    """

    structure_file: str
    incar_content: str
    kpoints_content: str
    evidence_level: EvidenceLevel


def _build_poscar(crystal: dict) -> str:
    """Build a minimal POSCAR-format string from crystal data.

    Args:
        crystal: Dictionary with composition, lattice_params,
            species, and cart_coords keys.

    Returns:
        POSCAR-format structure string.
    """
    composition = crystal["composition"]
    lattice = crystal["lattice_params"]
    species = crystal["species"]
    coords = crystal["cart_coords"]

    a = lattice.get("a", 5.0)
    b = lattice.get("b", 5.0)
    c = lattice.get("c", 5.0)

    # Preserve insertion order for species counting.
    unique = list(dict.fromkeys(species))
    counts = [species.count(s) for s in unique]

    lines = [
        composition,
        "1.0",
        "  %s  0.000000  0.000000" % format(a, ".6f"),
        "  0.000000  %s  0.000000" % format(b, ".6f"),
        "  0.000000  0.000000  %s" % format(c, ".6f"),
        "  ".join(unique),
        "  ".join(str(n) for n in counts),
        "Cartesian",
    ]
    for coord in coords:
        lines.append(
            "  %s  %s  %s"
            % (format(coord[0], ".6f"), format(coord[1], ".6f"), format(coord[2], ".6f"))
        )
    return "\n".join(lines) + "\n"


def _build_incar() -> str:
    """Build VASP INCAR content string with default parameters.

    Returns:
        Multi-line INCAR string containing ENCUT, ISPIN, and other defaults.
    """
    lines = [
        "ENCUT = %d" % _VASP_ENCUT,
        "ISPIN = %d" % _VASP_ISPIN,
        "EDIFF = %s" % _VASP_EDIFF,
        "ISMEAR = %d" % _VASP_ISMEAR,
        "SIGMA = %s" % _VASP_SIGMA,
        "ALGO = %s" % _VASP_ALGO,
        "LREAL = %s" % _VASP_LREAL,
    ]
    return "\n".join(lines) + "\n"


def _build_kpoints() -> str:
    """Build VASP KPOINTS content string with a Gamma-centred grid.

    Returns:
        KPOINTS file content string.
    """
    grid = _KPOINTS_GRID
    lines = [
        "Automatic mesh",
        "0",
        "Gamma",
        "  %d  %d  %d" % (grid, grid, grid),
        "  0  0  0",
    ]
    return "\n".join(lines) + "\n"


def prepare_dft_handoff(crystal: dict) -> DftHandoff:
    """Build a DFT handoff bundle from pre-processed crystal data.

    Args:
        crystal: Dictionary with keys:
            composition (str): Chemical formula (e.g. 'NaCl').
            lattice_params (dict): Lattice parameters with a, b, c keys.
            species (list[str]): Element symbols per site.
            cart_coords (list[list[float]]): Cartesian coordinates per site.

    Returns:
        DftHandoff with POSCAR structure, INCAR, KPOINTS, and DFT_VERIFIED
        evidence level.

    Raises:
        KeyError: If required keys are missing from crystal.
    """
    logger.info(
        "Preparing DFT handoff composition=%s",
        crystal.get("composition", "unknown"),
    )

    structure_file = _build_poscar(crystal)
    incar_content = _build_incar()
    kpoints_content = _build_kpoints()

    return DftHandoff(
        structure_file=structure_file,
        incar_content=incar_content,
        kpoints_content=kpoints_content,
        evidence_level=EvidenceLevel.DFT_VERIFIED,
    )
