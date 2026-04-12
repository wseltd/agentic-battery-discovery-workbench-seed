"""Tests for the domain keyword registry.

Focus: completeness of keyword lists against Q1 of research-pack.md,
disjointness invariants, and classify_token correctness including
case-insensitivity and ambiguity handling.
"""

from discovery_workbench.routing.keywords import (
    AMBIGUITY_KEYWORDS,
    INORGANIC_MATERIALS_KEYWORDS,
    SMALL_MOLECULE_KEYWORDS,
    STRUCTURED_CONSTRAINT_CUES,
    UNSUPPORTED_KEYWORDS,
    classify_token,
)


# ── Completeness checks against Q1 ──────────────────────────────────────

# These expected sets are transcribed directly from research-pack.md Q1.
# If a governance update changes Q1, update these sets and the module.

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
    """Every Q1 term must be present; no silent omissions."""

    def test_small_molecule_keywords_complete(self) -> None:
        missing = _EXPECTED_SMALL_MOLECULE - SMALL_MOLECULE_KEYWORDS
        extra = SMALL_MOLECULE_KEYWORDS - _EXPECTED_SMALL_MOLECULE
        assert not missing, f"missing from SMALL_MOLECULE_KEYWORDS: {missing}"
        assert not extra, f"unexpected in SMALL_MOLECULE_KEYWORDS: {extra}"

    def test_inorganic_keywords_complete(self) -> None:
        missing = _EXPECTED_INORGANIC - INORGANIC_MATERIALS_KEYWORDS
        extra = INORGANIC_MATERIALS_KEYWORDS - _EXPECTED_INORGANIC
        assert not missing, f"missing from INORGANIC_MATERIALS_KEYWORDS: {missing}"
        assert not extra, f"unexpected in INORGANIC_MATERIALS_KEYWORDS: {extra}"

    def test_unsupported_keywords_complete(self) -> None:
        missing = _EXPECTED_UNSUPPORTED - UNSUPPORTED_KEYWORDS
        extra = UNSUPPORTED_KEYWORDS - _EXPECTED_UNSUPPORTED
        assert not missing, f"missing from UNSUPPORTED_KEYWORDS: {missing}"
        assert not extra, f"unexpected in UNSUPPORTED_KEYWORDS: {extra}"

    def test_ambiguity_keywords_complete(self) -> None:
        missing = _EXPECTED_AMBIGUITY - AMBIGUITY_KEYWORDS
        extra = AMBIGUITY_KEYWORDS - _EXPECTED_AMBIGUITY
        assert not missing, f"missing from AMBIGUITY_KEYWORDS: {missing}"
        assert not extra, f"unexpected in AMBIGUITY_KEYWORDS: {extra}"


# ── Disjointness invariants ──────────────────────────────────────────────

class TestDisjointness:
    """Domain sets must never overlap -- overlapping keywords would make
    routing non-deterministic."""

    def test_domain_sets_mutually_disjoint(self) -> None:
        sm = SMALL_MOLECULE_KEYWORDS
        im = INORGANIC_MATERIALS_KEYWORDS
        us = UNSUPPORTED_KEYWORDS
        assert sm.isdisjoint(im), f"overlap sm/im: {sm & im}"
        assert sm.isdisjoint(us), f"overlap sm/us: {sm & us}"
        assert im.isdisjoint(us), f"overlap im/us: {im & us}"

    def test_ambiguity_keywords_disjoint_from_domains(self) -> None:
        all_domain = (
            SMALL_MOLECULE_KEYWORDS
            | INORGANIC_MATERIALS_KEYWORDS
            | UNSUPPORTED_KEYWORDS
        )
        overlap = AMBIGUITY_KEYWORDS & all_domain
        assert not overlap, (
            f"ambiguity terms found in domain sets (would auto-route): {overlap}"
        )


# ── Type invariants ──────────────────────────────────────────────────────

class TestTypeInvariants:
    """All public sets must be frozenset -- mutable sets would allow
    accidental runtime modification of routing behaviour."""

    def test_all_sets_are_frozenset(self) -> None:
        assert isinstance(SMALL_MOLECULE_KEYWORDS, frozenset)
        assert isinstance(INORGANIC_MATERIALS_KEYWORDS, frozenset)
        assert isinstance(UNSUPPORTED_KEYWORDS, frozenset)
        assert isinstance(AMBIGUITY_KEYWORDS, frozenset)
        for domain, cues in STRUCTURED_CONSTRAINT_CUES.items():
            assert isinstance(cues, frozenset), (
                f"STRUCTURED_CONSTRAINT_CUES[{domain!r}] is {type(cues).__name__}, "
                "expected frozenset"
            )

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
        assert not not_lower, f"keywords not lowercase-normalised: {not_lower}"


# ── classify_token behaviour ─────────────────────────────────────────────

class TestClassifyToken:
    """classify_token is the single public lookup function.  It must be
    case-insensitive, return the correct domain string, and return None
    for ambiguous or unknown tokens."""

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

    def test_classify_token_case_insensitive(self) -> None:
        assert classify_token("SMILES") == "small_molecule"
        assert classify_token("Crystal") == "inorganic_materials"
        assert classify_token("POLYMER") == "unsupported"
        assert classify_token("Catalyst") is None

    def test_classify_token_ambiguity_returns_none(self) -> None:
        for term in AMBIGUITY_KEYWORDS:
            result = classify_token(term)
            assert result is None, (
                f"ambiguity term {term!r} returned {result!r} instead of None"
            )

    def test_classify_token_ambiguity_case_variants_return_none(self) -> None:
        """Ambiguity guard must hold regardless of casing."""
        assert classify_token("CATALYST") is None
        assert classify_token("Battery Electrolyte") is None
        assert classify_token("PEROVSKITE") is None

    def test_classify_token_every_small_molecule_keyword(self) -> None:
        """Exhaustive: every keyword in the set must classify correctly."""
        for kw in SMALL_MOLECULE_KEYWORDS:
            assert classify_token(kw) == "small_molecule", f"{kw!r} misclassified"

    def test_classify_token_every_inorganic_keyword(self) -> None:
        for kw in INORGANIC_MATERIALS_KEYWORDS:
            assert classify_token(kw) == "inorganic_materials", f"{kw!r} misclassified"

    def test_classify_token_every_unsupported_keyword(self) -> None:
        for kw in UNSUPPORTED_KEYWORDS:
            assert classify_token(kw) == "unsupported", f"{kw!r} misclassified"

    def test_structured_constraint_cues_domains_valid(self) -> None:
        """STRUCTURED_CONSTRAINT_CUES keys must be known domain strings."""
        valid_domains = {"small_molecule", "inorganic_materials"}
        assert set(STRUCTURED_CONSTRAINT_CUES.keys()) == valid_domains
