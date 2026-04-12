"""Tests for the PAINS A/B/C filter module."""

from __future__ import annotations

import pytest
from rdkit.Chem import MolFromSmiles

from workbench.molecules.pains_filter import run_pains_filter


# ---------------------------------------------------------------------------
# Clean molecule — no PAINS alerts
# ---------------------------------------------------------------------------

def test_clean_molecule_passes_pains() -> None:
    """Caffeine contains no PAINS substructures."""
    mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    result = run_pains_filter(mol)
    assert result.passed is True
    assert result.matched_filters == []


# ---------------------------------------------------------------------------
# Category-specific hits
# ---------------------------------------------------------------------------

def test_known_pains_a_hit_fails() -> None:
    """1,4-benzoquinone matches PAINS_A (quinone_A)."""
    mol = MolFromSmiles("O=C1C=CC(=O)C=C1")
    result = run_pains_filter(mol)
    assert result.passed is False
    assert len(result.matched_filters) >= 1
    # At least one hit must come from the PAINS_A quinone family
    assert any("quinone" in name for name in result.matched_filters)


def test_known_pains_b_hit_fails() -> None:
    """Catechol (1,2-benzenediol) matches PAINS_B (catechol_A)."""
    mol = MolFromSmiles("Oc1ccccc1O")
    result = run_pains_filter(mol)
    assert result.passed is False
    assert any("catechol" in name for name in result.matched_filters)


def test_known_pains_c_hit_fails() -> None:
    """An ene-rhodanine variant matches PAINS_C (ene_rhod_E)."""
    mol = MolFromSmiles("O=C1NC(=S)/C(=C/c2cccc([N+](=O)[O-])c2)S1")
    result = run_pains_filter(mol)
    assert result.passed is False
    assert any("ene_rhod" in name for name in result.matched_filters)


# ---------------------------------------------------------------------------
# Canonical RDKit names
# ---------------------------------------------------------------------------

def test_matched_filter_names_are_rdkit_canonical() -> None:
    """Matched filter names must be the RDKit canonical names, not invented.

    Canonical names follow the pattern 'family_suffix(count)' — e.g.
    'quinone_A(370)'.  We verify the format rather than hardcoding exact
    names, since RDKit may update catalogs across versions.
    """
    mol = MolFromSmiles("O=C1C=CC(=O)C=C1")  # quinone
    result = run_pains_filter(mol)
    for name in result.matched_filters:
        # Canonical names contain parenthesised integers at the end
        assert "(" in name and name.endswith(")"), (
            f"Filter name {name!r} does not look like an RDKit canonical name"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_none_mol_raises_typeerror() -> None:
    """Passing None instead of a Mol must raise TypeError."""
    with pytest.raises(TypeError, match="Expected rdkit.Chem.Mol") as exc_info:
        run_pains_filter(None)
    assert "Expected rdkit.Chem.Mol" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Multiple matches
# ---------------------------------------------------------------------------

def test_multiple_pains_matches_returns_all() -> None:
    """A molecule with both ene-rhodanine and catechol substructures
    should return multiple matched filter names."""
    # ene-rhodanine linked to catechol
    mol = MolFromSmiles("O=C1/C(=C/c2ccc(O)c(O)c2)SC(=S)N1")
    result = run_pains_filter(mol)
    assert result.passed is False
    assert len(result.matched_filters) >= 2, (
        f"Expected >=2 matches, got {result.matched_filters}"
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_catalog_is_module_level_singleton() -> None:
    """The PAINS_CATALOG must be a module-level object, not rebuilt per call.

    We verify identity: importing the catalog twice yields the same object,
    and calling run_pains_filter does not replace it.
    """
    from workbench.molecules import pains_filter

    catalog_before = pains_filter.PAINS_CATALOG
    # Run a filter to confirm the catalog isn't replaced during use
    mol = MolFromSmiles("c1ccccc1")
    run_pains_filter(mol)
    catalog_after = pains_filter.PAINS_CATALOG
    assert catalog_before is catalog_after
