"""Prepare molecules for xTB semiempirical quantum-chemistry handoff.

Generates 3-D conformers via RDKit ETKDG, builds XYZ and SDF content,
and packages everything into a HandoffBundle for downstream xTB execution.
"""

from __future__ import annotations

import dataclasses
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, inchi

from agentic_discovery.shared.evidence import EvidenceLevel

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
        warnings: list[str] = []

        mol_h = Chem.AddHs(mol)
        mol_h, warnings = self._embed(mol_h, warnings)

        smiles = Chem.MolToSmiles(mol)
        inchikey = inchi.MolToInchiKey(mol)

        xyz_content = self._mol_to_xyz(mol_h)
        sdf_content = Chem.MolToMolBlock(mol_h)
        run_script = _RUN_SCRIPT_TEMPLATE.format(
            charge=charge,
            multiplicity=multiplicity,
        )

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

    def _embed(
        self,
        mol_h: Chem.Mol,
        warnings: list[str],
    ) -> tuple[Chem.Mol, list[str]]:
        """Try ETKDG, then fall back to random coordinates."""
        params = AllChem.ETKDGv3()
        status = AllChem.EmbedMolecule(mol_h, params)
        if status == 0:
            return mol_h, warnings

        # ETKDG failed — retry with random coordinates.
        logger.warning("ETKDG embedding failed; retrying with random coordinates")
        warnings.append("ETKDG failed — used random coordinate embedding")
        status = AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
        if status == 0:
            return mol_h, warnings

        raise ConformerGenerationError(
            "Could not generate a 3-D conformer after ETKDG and random-coordinate attempts"
        )

    @staticmethod
    def _mol_to_xyz(mol_h: Chem.Mol) -> str:
        """Convert a molecule with a conformer to XYZ-format string."""
        conf = mol_h.GetConformer()
        atoms = mol_h.GetAtoms()
        lines: list[str] = [str(mol_h.GetNumAtoms()), ""]
        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            lines.append(f"{symbol}  {pos.x: .6f}  {pos.y: .6f}  {pos.z: .6f}")
        return "\n".join(lines) + "\n"


def build_bundle(
    mol: Chem.Mol,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> HandoffBundle:
    """Module-level convenience — delegates to XtbHandoffBuilder.build_bundle."""
    return XtbHandoffBuilder().build_bundle(mol, charge=charge, multiplicity=multiplicity)
