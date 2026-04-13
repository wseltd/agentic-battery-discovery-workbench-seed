"""Prepare molecules for xTB semiempirical quantum-chemistry handoff.

Generates 3-D conformers via RDKit ETKDG, builds XYZ and SDF content,
and packages everything into a HandoffBundle for downstream xTB execution.
"""

from __future__ import annotations

import dataclasses
import logging

from rdkit import Chem  # provided by rdkit-pypi in pyproject.toml
from rdkit.Chem import AllChem, inchi  # provided by rdkit-pypi in pyproject.toml

from agentic_molecule_discovery.evidence import EvidenceLevel

logger = logging.getLogger(__name__)

# xTB run-script template.  Charge and multiplicity are placeholders
# because the correct values depend on protonation state, which the
# user must confirm before launching a real calculation.
_RUN_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# --- xTB run script (auto-generated) ---
# USER: confirm charge and multiplicity before running.
set -euo pipefail

CHARGE={charge}
MULTIPLICITY={multiplicity}

xtb structure.xyz --chrg "$CHARGE" --uhf $(( MULTIPLICITY - 1 )) --opt tight
"""

class ConformerGenerationError(ValueError):
    """Raised when RDKit cannot generate a 3-D conformer for a molecule."""

    def __repr__(self) -> str:
        msg = str(self)
        return f"ConformerGenerationError({msg!r})"


@dataclasses.dataclass(frozen=True)
class HandoffBundle:
    """Immutable bundle of artifacts needed to launch an xTB calculation.

    Parameters
    ----------
    smiles : str
        Canonical SMILES of the input molecule.
    inchikey : str
        InChIKey for identity verification.
    xyz_content : str
        XYZ-format geometry (atom count on line 1, blank/comment line 2,
        then ``symbol  x  y  z`` lines).
    run_script : str
        Bash script that invokes xTB with charge/multiplicity placeholders.
    sdf_content : str
        SDF/V3000 mol-block for interchange with other tools.
    evidence_level : EvidenceLevel
        Credibility tier — always GENERATED for freshly embedded conformers.
    warnings : list[str]
        Diagnostic messages (e.g. ETKDG fallback).
    """

    smiles: str
    inchikey: str
    xyz_content: str
    run_script: str
    sdf_content: str
    evidence_level: EvidenceLevel
    warnings: list[str]


class XtbHandoffBuilder:
    """Builds a HandoffBundle from an RDKit Mol or SMILES string.

    Conformer generation uses ETKDG first; if that fails, retries once
    with random coordinates. Both failures raise ConformerGenerationError.
    """

    def build_bundle(
        self,
        mol: Chem.Mol,
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> HandoffBundle:
        """Generate a 3-D conformer and package it for xTB.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Input molecule (2-D or 3-D). Hydrogens are added automatically.
        charge : int
            Formal charge placeholder for the run script (default 0).
        multiplicity : int
            Spin multiplicity placeholder for the run script (default 1).

        Returns
        -------
        HandoffBundle

        Raises
        ------
        ConformerGenerationError
            If neither ETKDG nor random-coordinate embedding succeeds.
        """
        if not isinstance(mol, Chem.Mol):
            raise TypeError(
                f"mol must be an rdkit.Chem.Mol, got {type(mol).__name__}"
            )
        if mol.GetNumAtoms() == 0:
            raise ValueError("mol has no atoms — cannot generate a conformer")

        warnings: list[str] = []

        mol_h = Chem.AddHs(mol)
        mol_h, warnings = self._embed(mol_h, warnings)

        smiles = Chem.MolToSmiles(mol)
        inchikey = inchi.MolToInchiKey(mol)

        xyz_content = self._export_xyz(mol_h)
        sdf_content = self._export_sdf(mol_h)
        run_script = self._generate_run_script(charge=charge, multiplicity=multiplicity)

        return HandoffBundle(
            smiles=smiles,
            inchikey=inchikey,
            xyz_content=xyz_content,
            run_script=run_script,
            sdf_content=sdf_content,
            evidence_level=EvidenceLevel.GENERATED,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_script(charge: int = 0, multiplicity: int = 1) -> str:
        """Return a bash script string for xTB execution."""
        return _RUN_SCRIPT_TEMPLATE.format(charge=charge, multiplicity=multiplicity)

    def _embed(
        self,
        mol_h: Chem.Mol,
        warnings: list[str],
    ) -> tuple[Chem.Mol, list[str]]:
        """Try ETKDG, then fall back to random coordinates."""
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        status = AllChem.EmbedMolecule(mol_h, params)  # type: ignore[attr-defined]
        if status == 0:
            return mol_h, warnings

        # ETKDG failed — retry with random coordinates.
        logger.warning("ETKDG embedding failed; retrying with random coordinates")
        warnings.append("ETKDG failed — used random coordinate embedding")
        status = AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)  # type: ignore[attr-defined]
        if status == 0:
            return mol_h, warnings

        raise ConformerGenerationError(
            "Could not generate a 3-D conformer after ETKDG and random-coordinate attempts"
        )

    @staticmethod
    def _export_xyz(mol: Chem.Mol) -> str:
        """Convert a molecule with an embedded conformer to XYZ-format string."""
        return Chem.MolToXYZBlock(mol)

    @staticmethod
    def _export_sdf(mol: Chem.Mol) -> str:
        """Convert a molecule with an embedded conformer to SDF mol-block."""
        return Chem.MolToMolBlock(mol)


def build_bundle(
    mol: Chem.Mol,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> HandoffBundle:
    """Module-level convenience — delegates to XtbHandoffBuilder.build_bundle."""
    return XtbHandoffBuilder().build_bundle(mol, charge=charge, multiplicity=multiplicity)
