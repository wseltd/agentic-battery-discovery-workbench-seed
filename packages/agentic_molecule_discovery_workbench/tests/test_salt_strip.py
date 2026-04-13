"""Tests for molecular_validity.salt_strip.strip_salts."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

from agentic_molecule_discovery.validation.salt_strip import strip_salts


# --- helpers -----------------------------------------------------------------

def _mol(smiles: str) -> Chem.Mol:
    """Parse SMILES, failing the test if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    return mol


def _smiles(mol: Chem.Mol) -> str:
    """Canonical SMILES for comparison."""
    return Chem.MolToSmiles(mol)


# --- single-fragment molecules -----------------------------------------------

class TestSingleFragment:
    """A molecule with one fragment should be returned unchanged."""

    def test_single_organic_fragment_unchanged(self):
        mol = _mol("c1ccccc1")  # benzene
        result = strip_salts(mol)
        assert result.mol is not None
        assert _smiles(result.mol) == _smiles(mol)
        assert result.original_fragment_count == 1
        assert result.fragments_removed == 0

    def test_single_fragment_counts(self):
        result = strip_salts(_mol("CCO"))
        assert result.original_fragment_count == 1
        assert result.fragments_removed == 0


# --- organic + inorganic salt ------------------------------------------------

class TestOrganicWithSalt:
    """Organic fragment should be kept, inorganic counterion discarded."""

    def test_sodium_phenoxide_keeps_phenol(self):
        # Sodium phenoxide: [Na+].[O-]c1ccccc1
        result = strip_salts(_mol("[Na+].[O-]c1ccccc1"))
        assert result.mol is not None
        assert result.original_fragment_count == 2
        assert result.fragments_removed == 1
        # The organic fragment should contain the aromatic ring
        assert result.mol.GetNumHeavyAtoms() == 7  # phenoxide: 6C + 1O

    def test_hydrochloride_salt_keeps_amine(self):
        # Amine hydrochloride: CCN.[Cl-]
        result = strip_salts(_mol("CCN.[Cl-]"))
        assert result.mol is not None
        assert result.fragments_removed == 1
        assert _smiles(result.mol) == _smiles(_mol("CCN"))

    def test_calcium_salt_with_two_organic_anions(self):
        # Calcium acetate: [Ca+2].[O-]C(=O)C.[O-]C(=O)C
        # Three fragments total, two organic (identical acetate), one inorganic
        result = strip_salts(_mol("[Ca+2].[O-]C(=O)C.[O-]C(=O)C"))
        assert result.mol is not None
        assert result.original_fragment_count == 3
        assert result.fragments_removed == 2


# --- multi-salt mixture ------------------------------------------------------

class TestMultiSaltMixture:
    """Multiple salts: the largest organic fragment wins."""

    def test_largest_organic_fragment_selected(self):
        # Aspirin sodium: large organic + Na + small acetic acid fragment
        # c1ccc(OC(=O)C)c(C(=O)[O-])c1.[Na+].CC(=O)O
        result = strip_salts(_mol("c1ccc(OC(=O)C)c(C(=O)[O-])c1.[Na+].CC(=O)O"))
        assert result.mol is not None
        assert result.original_fragment_count == 3
        # The largest organic fragment is the aspirin core (13 heavy atoms)
        assert result.mol.GetNumHeavyAtoms() == 13

    def test_three_organic_fragments_picks_largest(self):
        # Methane.Ethane.Propane — three organics, propane is largest
        result = strip_salts(_mol("C.CC.CCC"))
        assert result.mol is not None
        assert result.mol.GetNumHeavyAtoms() == 3
        assert result.fragments_removed == 2


# --- purely inorganic input --------------------------------------------------

class TestPurelyInorganic:
    """No organic fragment exists — result mol should be None."""

    def test_sodium_chloride_returns_none(self):
        result = strip_salts(_mol("[Na+].[Cl-]"))
        assert result.mol is None
        assert result.original_fragment_count == 2
        assert result.fragments_removed == 2

    def test_single_metal_ion_returns_none(self):
        result = strip_salts(_mol("[Fe+3]"))
        assert result.mol is None
        assert result.original_fragment_count == 1
        assert result.fragments_removed == 1

    def test_multiple_inorganic_fragments_returns_none(self):
        # Calcium chloride: [Ca+2].[Cl-].[Cl-]
        result = strip_salts(_mol("[Ca+2].[Cl-].[Cl-]"))
        assert result.mol is None
        assert result.original_fragment_count == 3
        assert result.fragments_removed == 3


# --- tie-breaking by molecular weight ----------------------------------------

class TestMolecularWeightTieBreaker:
    """Two organic fragments with the same heavy atom count — heavier wins."""

    def test_equal_heavy_atoms_picks_heavier(self):
        # Chlorobenzene (6C + 1Cl = 7 heavy) vs phenol (6C + 1O = 7 heavy)
        # Chlorobenzene MW ~112.6, phenol MW ~94.1 — chlorobenzene should win
        chlorobenzene = _mol("Clc1ccccc1")
        phenol = _mol("Oc1ccccc1")
        assert chlorobenzene.GetNumHeavyAtoms() == phenol.GetNumHeavyAtoms()

        # Dot-separated: phenol first, chlorobenzene second
        result = strip_salts(_mol("Oc1ccccc1.Clc1ccccc1"))
        assert result.mol is not None
        # Should pick chlorobenzene (heavier)
        result_mw = Descriptors.ExactMolWt(result.mol)
        chloro_mw = Descriptors.ExactMolWt(chlorobenzene)
        assert abs(result_mw - chloro_mw) < 0.1, (
            f"Expected chlorobenzene (MW={chloro_mw:.1f}), "
            f"got MW={result_mw:.1f}"
        )

    def test_reversed_input_order_same_result(self):
        # Ensure tie-breaking is stable regardless of fragment order
        result_a = strip_salts(_mol("Oc1ccccc1.Clc1ccccc1"))
        result_b = strip_salts(_mol("Clc1ccccc1.Oc1ccccc1"))
        assert result_a.mol is not None
        assert result_b.mol is not None
        assert _smiles(result_a.mol) == _smiles(result_b.mol)


# --- dataclass contract ------------------------------------------------------

class TestResultContract:
    """The SaltStripResult dataclass behaves as expected."""

    def test_result_is_frozen(self):
        result = strip_salts(_mol("CCO"))
        try:
            result.fragments_removed = 99  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "SaltStripResult should be frozen (immutable)"

    def test_fragments_removed_plus_one_equals_original_when_organic_found(self):
        """When an organic fragment is found, fragments_removed = original - 1."""
        result = strip_salts(_mol("[Na+].[Cl-].CCO"))
        assert result.mol is not None
        assert result.fragments_removed == result.original_fragment_count - 1
