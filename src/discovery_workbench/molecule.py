"""Canonical molecule representation with RDKit-backed validation and export.

Provides a frozen dataclass wrapping an RDKit Mol object with canonical SMILES,
InChIKey, and evidence-level tracking.  All construction goes through the
``from_smiles`` classmethod which sanitises, canonicalises, strips salts, and
computes the InChIKey.
"""

from __future__ import annotations

__version__ = "0.1.0"

import dataclasses

from rdkit import Chem  # provided by rdkit-pypi in pyproject.toml
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.SaltRemover import SaltRemover

from discovery_workbench.evidence import EvidenceLevel

# Module-level salt remover — built once, reused across calls.
_SALT_REMOVER = SaltRemover()


def _resolve_evidence_level(value: str | EvidenceLevel) -> EvidenceLevel:
    """Accept an EvidenceLevel member or its label string, return the enum member.

    Raises
    ------
    ValueError
        If *value* is a string that does not match any EvidenceLevel label.
    """
    if isinstance(value, EvidenceLevel):
        return value
    # Match against the label (first element of the value tuple).
    for member in EvidenceLevel:
        if member.value[0] == value:
            return member
    valid = [m.value[0] for m in EvidenceLevel]
    raise ValueError(
        f"Unknown evidence level {value!r}. Valid labels: {valid}"
    )


@dataclasses.dataclass(frozen=True)
class CanonicalMolecule:
    """Immutable canonical representation of a small molecule.

    Do not construct directly — use :meth:`from_smiles`.

    Parameters
    ----------
    canonical_smiles:
        RDKit-canonical SMILES string.
    inchikey:
        27-character InChIKey.
    evidence_level:
        Credibility tier for this molecule's provenance.
    """

    canonical_smiles: str
    inchikey: str
    evidence_level: EvidenceLevel
    _mol: Chem.Mol = dataclasses.field(repr=False, compare=False)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        evidence_level: str | EvidenceLevel,
    ) -> CanonicalMolecule:
        """Parse, sanitise, canonicalise, and validate a SMILES string.

        Performs salt/fragment stripping (keeps the largest fragment),
        canonical SMILES generation, and InChIKey computation.

        Parameters
        ----------
        smiles:
            Input SMILES (may be non-canonical, contain salts, etc.).
        evidence_level:
            An ``EvidenceLevel`` member or its label string (e.g. ``"generated"``).

        Returns
        -------
        CanonicalMolecule

        Raises
        ------
        ValueError
            If *smiles* cannot be parsed, sanitised, or an InChIKey cannot
            be generated.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")

        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(f"Sanitisation failed for SMILES {smiles!r}: {exc}") from exc

        # Strip salts — keeps the largest fragment.
        mol = _SALT_REMOVER.StripMol(mol)

        canonical = Chem.MolToSmiles(mol, canonical=True)

        inchi = MolToInchi(mol)
        if inchi is None:
            raise ValueError(f"InChI generation failed for SMILES: {smiles!r}")

        inchikey = InchiToInchiKey(inchi)
        if inchikey is None:
            raise ValueError(f"InChIKey generation failed for SMILES: {smiles!r}")

        level = _resolve_evidence_level(evidence_level)

        return cls(
            canonical_smiles=canonical,
            inchikey=inchikey,
            evidence_level=level,
            _mol=mol,
        )

    # -- export methods --------------------------------------------------------

    def to_molblock(self) -> str:
        """Return a V2000 molblock string (MDL MOL format).

        The molblock is generated from a copy of the internal molecule to
        avoid mutating frozen state.
        """
        return rdmolfiles.MolToMolBlock(Chem.RWMol(self._mol))

    def to_sdf(self) -> str:
        """Return an SDF string (molblock followed by ``$$$$`` terminator)."""
        return self.to_molblock() + "\n$$$$\n"

    def to_xyz(self, conformer_id: int = 0) -> str:
        """Return an XYZ-format string for the given conformer.

        Parameters
        ----------
        conformer_id:
            Index of the conformer to export (default ``0``).

        Raises
        ------
        ValueError
            If the molecule has no conformers or *conformer_id* is out of range.
            Call :meth:`embed_conformer` first.
        """
        num_confs = self._mol.GetNumConformers()
        if num_confs == 0:
            raise ValueError(
                "No conformer embedded. Call embed_conformer() first."
            )
        if conformer_id < 0 or conformer_id >= num_confs:
            raise ValueError(
                f"conformer_id {conformer_id} out of range "
                f"(molecule has {num_confs} conformer(s))"
            )
        return Chem.MolToXYZBlock(self._mol, confId=conformer_id)

    def embed_conformer(self, random_seed: int = 42) -> CanonicalMolecule:
        """Return a **new** CanonicalMolecule with an embedded 3-D conformer.

        Uses ETKDG (v3) for conformer generation.  The current instance is
        unchanged (frozen dataclass).

        Parameters
        ----------
        random_seed:
            Seed for the conformer embedder, for reproducibility.

        Raises
        ------
        ValueError
            If conformer embedding fails (e.g. molecule too constrained).
        """
        mol_copy = Chem.RWMol(self._mol)
        mol_copy = Chem.AddHs(mol_copy)
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = random_seed
        status = AllChem.EmbedMolecule(mol_copy, params)  # type: ignore[attr-defined]
        if status == -1:
            raise ValueError(
                f"Conformer embedding failed for {self.canonical_smiles!r}. "
                "The molecule may be too constrained for 3-D coordinate generation."
            )
        mol_copy = Chem.RemoveHs(mol_copy)
        return CanonicalMolecule(
            canonical_smiles=self.canonical_smiles,
            inchikey=self.inchikey,
            evidence_level=self.evidence_level,
            _mol=mol_copy,
        )
