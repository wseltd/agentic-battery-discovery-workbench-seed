"""Tests for the domain keyword registry."""

from agentic_discovery_core.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    INORGANIC_MATERIALS_KEYWORDS,
    SMALL_MOLECULE_KEYWORDS,
    STRUCTURED_CONSTRAINT_CUES,
    UNSUPPORTED_KEYWORDS,
    classify_token,
)


_EXPECTED_SMALL_MOLECULE = frozenset({
    "smiles", "inchi", "ligand", "docking", "admet", "clogp", "logp",
    "tpsa", "hbd", "hba", "qed", "pains", "scaffold", "linker", "r-group",
    "lead optimisation", "hit generation", "sar", "qsar", "ic50", "ec50",
    "ki", "lipinski", "fragment", "safe", "fragment remasking",
    "pmo benchmark",
})

_EXPECTED_INORGANIC = frozenset({
    "crystal", "unit cell", "lattice", "fractional coordinates",
    "space group", "cif", "poscar", "primitive cell", "niggli reduction",
    "convex hull", "energy above hull", "formation energy", "phonon",
    "k-points", "vasp", "bulk modulus", "band gap", "magnetic density",
    "structure relaxation", "mattergen", "mattersim", "s.u.n. structures",
})

_EXPECTED_UNSUPPORTED = frozenset({
    "polymer", "protein", "biologics", "mixture", "solution",
    "device", "process design", "synthesis planning", "mof", "cof",
})

_EXPECTED_AMBIGUITY = frozenset({
    "catalyst", "battery electrolyte", "semiconductor",
    "molecular magnet", "crystalline magnet", "perovskite", "nanoparticle",
})


class TestKeywordCompleteness:
    def test_small_molecule_keywords_complete(self) -> None:
        missing = _EXPECTED_SMALL_MOLECULE - SMALL_MOLECULE_KEYWORDS
        extra = SMALL_MOLECULE_KEYWORDS - _EXPECTED_SMALL_MOLECULE
        assert not missing
        assert not extra

    def test_inorganic_keywords_complete(self) -> None:
        missing = _EXPECTED_INORGANIC - INORGANIC_MATERIALS_KEYWORDS
        extra = INORGANIC_MATERIALS_KEYWORDS - _EXPECTED_INORGANIC
        assert not missing
        assert not extra

    def test_unsupported_keywords_complete(self) -> None:
        missing = _EXPECTED_UNSUPPORTED - UNSUPPORTED_KEYWORDS
        extra = UNSUPPORTED_KEYWORDS - _EXPECTED_UNSUPPORTED
        assert not missing
        assert not extra

    def test_ambiguity_keywords_complete(self) -> None:
        missing = _EXPECTED_AMBIGUITY - AMBIGUITY_KEYWORDS
        extra = AMBIGUITY_KEYWORDS - _EXPECTED_AMBIGUITY
        assert not missing
        assert not extra


class TestDisjointness:
    def test_domain_sets_mutually_disjoint(self) -> None:
        sm = SMALL_MOLECULE_KEYWORDS
        im = INORGANIC_MATERIALS_KEYWORDS
        us = UNSUPPORTED_KEYWORDS
        assert sm.isdisjoint(im)
        assert sm.isdisjoint(us)
        assert im.isdisjoint(us)

    def test_ambiguity_keywords_disjoint_from_domains(self) -> None:
        all_domain = (
            SMALL_MOLECULE_KEYWORDS
            | INORGANIC_MATERIALS_KEYWORDS
            | UNSUPPORTED_KEYWORDS
        )
        overlap = AMBIGUITY_KEYWORDS & all_domain
        assert not overlap


class TestTypeInvariants:
    def test_all_sets_are_frozenset(self) -> None:
        assert isinstance(SMALL_MOLECULE_KEYWORDS, frozenset)
        assert isinstance(INORGANIC_MATERIALS_KEYWORDS, frozenset)
        assert isinstance(UNSUPPORTED_KEYWORDS, frozenset)
        assert isinstance(AMBIGUITY_KEYWORDS, frozenset)
        for domain, cues in STRUCTURED_CONSTRAINT_CUES.items():
            assert isinstance(cues, frozenset)
        assert len(SMALL_MOLECULE_KEYWORDS) >= 20
        assert len(INORGANIC_MATERIALS_KEYWORDS) >= 15
        assert len(STRUCTURED_CONSTRAINT_CUES) == 2

    def test_all_keywords_lowercase(self) -> None:
        all_terms = (
            SMALL_MOLECULE_KEYWORDS
            | INORGANIC_MATERIALS_KEYWORDS
            | UNSUPPORTED_KEYWORDS
            | AMBIGUITY_KEYWORDS
        )
        for cues in STRUCTURED_CONSTRAINT_CUES.values():
            all_terms = all_terms | cues

        not_lower = {t for t in all_terms if t != t.lower()}
        assert not not_lower


class TestClassifyToken:
    def test_classify_token_small_molecule(self) -> None:
        assert classify_token("smiles") == "small_molecule"
        assert classify_token("qed") == "small_molecule"
        assert classify_token("pmo benchmark") == "small_molecule"

    def test_classify_token_inorganic(self) -> None:
        assert classify_token("crystal") == "inorganic_materials"
        assert classify_token("poscar") == "inorganic_materials"
        assert classify_token("s.u.n. structures") == "inorganic_materials"

    def test_classify_token_unsupported(self) -> None:
        assert classify_token("polymer") == "unsupported"
        assert classify_token("mof") == "unsupported"
        assert classify_token("biologics") == "unsupported"

    def test_classify_token_unknown_returns_none(self) -> None:
        assert classify_token("banana") is None
        assert classify_token("") is None
        assert classify_token("  ") is None
        assert classify_token("smiles") == "small_molecule"

    def test_classify_token_case_insensitive(self) -> None:
        assert classify_token("SMILES") == "small_molecule"
        assert classify_token("Crystal") == "inorganic_materials"
        assert classify_token("POLYMER") == "unsupported"
        assert classify_token("Catalyst") is None

    def test_classify_token_ambiguity_returns_none(self) -> None:
        for term in AMBIGUITY_KEYWORDS:
            result = classify_token(term)
            assert result is None
        assert len(AMBIGUITY_KEYWORDS) == len(_EXPECTED_AMBIGUITY)

    def test_classify_token_ambiguity_case_variants_return_none(self) -> None:
        assert classify_token("CATALYST") is None
        assert classify_token("Battery Electrolyte") is None
        assert classify_token("PEROVSKITE") is None
        assert classify_token("CRYSTAL") == "inorganic_materials"

    def test_classify_token_every_small_molecule_keyword(self) -> None:
        for kw in SMALL_MOLECULE_KEYWORDS:
            assert classify_token(kw) == "small_molecule"

    def test_classify_token_every_inorganic_keyword(self) -> None:
        for kw in INORGANIC_MATERIALS_KEYWORDS:
            assert classify_token(kw) == "inorganic_materials"

    def test_classify_token_every_unsupported_keyword(self) -> None:
        for kw in UNSUPPORTED_KEYWORDS:
            assert classify_token(kw) == "unsupported"

    def test_structured_constraint_cues_domains_valid(self) -> None:
        valid_domains = {"small_molecule", "inorganic_materials"}
        assert set(STRUCTURED_CONSTRAINT_CUES.keys()) == valid_domains
