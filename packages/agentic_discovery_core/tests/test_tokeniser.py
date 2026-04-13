"""Tests for the routing tokeniser."""

from __future__ import annotations

from agentic_discovery_core.routing.tokeniser import tokenise


class TestWhitespaceSplitting:
    def test_simple_whitespace_split(self) -> None:
        result = tokenise("hello world")
        assert "hello" in result
        assert "world" in result
        assert "hello world" in result

    def test_multiple_spaces_collapsed(self) -> None:
        result = tokenise("foo   bar")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["foo", "bar"]
        assert "foo bar" in result

    def test_tabs_and_newlines(self) -> None:
        result = tokenise("alpha\tbeta\ngamma")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["alpha", "beta", "gamma"]


class TestDelimiterSplitting:
    def test_hyphen_split(self) -> None:
        result = tokenise("Li-Fe-P-O")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["li", "fe", "p", "o"]

    def test_slash_split(self) -> None:
        result = tokenise("eV/atom")
        assert "ev" in result
        assert "atom" in result

    def test_comma_split(self) -> None:
        result = tokenise("crystal,lattice,phonon")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["crystal", "lattice", "phonon"]

    def test_semicolon_split(self) -> None:
        result = tokenise("clogp;tpsa")
        assert "clogp" in result
        assert "tpsa" in result
        assert "clogp tpsa" in result

    def test_parentheses_split(self) -> None:
        result = tokenise("molecule(smiles)")
        assert "molecule" in result
        assert "smiles" in result

    def test_mixed_delimiters(self) -> None:
        result = tokenise("crystal/lattice-phonon,band;gap(cif)")
        unigrams = [t for t in result if " " not in t]
        assert set(unigrams) == {"crystal", "lattice", "phonon", "band", "gap", "cif"}


class TestBigramGeneration:
    def test_bigram_space_group(self) -> None:
        result = tokenise("the space group is P21")
        assert "space group" in result

    def test_bigram_lead_optimisation(self) -> None:
        result = tokenise("apply lead optimisation")
        assert "lead optimisation" in result

    def test_bigram_convex_hull(self) -> None:
        result = tokenise("check convex hull stability")
        assert "convex hull" in result

    def test_bigram_count(self) -> None:
        result = tokenise("a b c d")
        unigrams = [t for t in result if " " not in t]
        bigrams = [t for t in result if " " in t]
        assert len(unigrams) == 4
        assert len(bigrams) == 3

    def test_single_word_no_bigrams(self) -> None:
        result = tokenise("crystal")
        assert result == ["crystal"]

    def test_bigrams_respect_delimiter_boundaries(self) -> None:
        result = tokenise("Li-Fe")
        assert "li fe" in result
        assert "li-fe" not in result


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert tokenise("") == []

    def test_only_delimiters(self) -> None:
        assert tokenise("---///,,,;;;()") == []

    def test_leading_trailing_delimiters(self) -> None:
        result = tokenise("-crystal-")
        assert result == ["crystal"]

    def test_lowercasing(self) -> None:
        result = tokenise("CRYSTAL CIF SMILES")
        unigrams = [t for t in result if " " not in t]
        assert all(t == t.lower() for t in unigrams)
        assert "crystal" in result
        assert "cif" in result
        assert "smiles" in result

    def test_unicode_passthrough(self) -> None:
        result = tokenise("\u00c5ngstr\u00f6m caf\u00e9")
        assert "\u00e5ngstr\u00f6m" in result
        assert "caf\u00e9" in result
