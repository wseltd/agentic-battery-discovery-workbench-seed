"""Prepare xTB semiempirical quantum-chemistry handoff bundles.

Accepts pre-processed molecule data (SMILES, InChIKey, XYZ geometry)
and packages it into a lightweight handoff bundle for downstream xTB
execution.  Does not invoke RDKit or xTB -- operates on strings only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amdw.shared.evidence import EvidenceLevel

logger = logging.getLogger(__name__)

__all__ = ["XtbHandoff", "prepare_xtb_handoff"]

# xTB run-script template.  Charge and multiplicity are filled at
# bundle-generation time; the user should still verify them before
# launching a real calculation.
_RUN_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
CHARGE={charge}
MULTIPLICITY={multiplicity}
xtb structure.xyz --chrg "$CHARGE" --uhf $(( MULTIPLICITY - 1 )) --opt tight
"""


@dataclass(frozen=True, slots=True)
class XtbHandoff:
    """Immutable handoff bundle for xTB semiempirical QC.

    Args:
        xyz_content: XYZ-format geometry string.
        run_script: Bash script invoking xTB.
        charge: Molecular formal charge.
        multiplicity: Spin multiplicity.
        evidence_level: Credibility tier -- always SEMIEMPIRICAL_QC.
    """

    xyz_content: str
    run_script: str
    charge: int
    multiplicity: int
    evidence_level: EvidenceLevel


def prepare_xtb_handoff(molecule: dict) -> XtbHandoff:
    """Build an xTB handoff bundle from pre-processed molecule data.

    Args:
        molecule: Dictionary with keys:
            conformer_xyz (str): XYZ-format geometry.
            charge (int): Molecular formal charge.
            multiplicity (int): Spin multiplicity.
            Optional keys (smiles, inchikey) are accepted but not
            required for bundle generation.

    Returns:
        XtbHandoff with populated fields and SEMIEMPIRICAL_QC evidence level.

    Raises:
        ValueError: If conformer_xyz is missing or empty.
    """
    logger.info("Preparing xTB handoff")

    xyz_content = molecule.get("conformer_xyz", "")
    if not xyz_content or not isinstance(xyz_content, str):
        raise ValueError("molecule must contain a non-empty 'conformer_xyz' string")

    charge = molecule.get("charge", 0)
    multiplicity = molecule.get("multiplicity", 1)

    run_script = _RUN_SCRIPT_TEMPLATE.format(
        charge=charge,
        multiplicity=multiplicity,
    )

    return XtbHandoff(
        xyz_content=xyz_content,
        run_script=run_script,
        charge=charge,
        multiplicity=multiplicity,
        evidence_level=EvidenceLevel.SEMIEMPIRICAL_QC,
    )
