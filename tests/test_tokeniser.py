"""Tests for the routing tokeniser.

The tokeniser feeds the deterministic router — wrong tokens mean wrong
routing verdicts.  Tests focus on delimiter handling, bigram generation,
and edge cases that would silently corrupt keyword matching.
"""

from __future__ import annotations

from discovery_workbench.routing.tokeniser import tokenise


# ---------------------------------------------------------------------------
# Whitespace splitting
# ---------------------------------------------------------------------------


class TestWhitespaceSplitting:
    """Verify basic whitespace-only input is split correctly."""

    def test_simple_whitespace_split(self) -> None:
        """Plain space-separated words produce correct unigrams and bigrams."""
        result = tokenise("hello world")
        assert "hello" in result
        assert "world" in result
        assert "hello world" in result

    def test_multiple_spaces_collapsed(self) -> None:
        """Consecutive spaces do not produce empty tokens."""
        result = tokenise("foo   bar")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["foo", "bar"]
        assert "foo bar" in result

    def test_tabs_and_newlines(self) -> None:
        """Tabs and newlines are treated as delimiters."""
        result = tokenise("alpha\tbeta\ngamma")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Delimiter splitting
# ---------------------------------------------------------------------------


class TestDelimiterSplitting:
    """Verify splitting on hyphens, slashes, commas, semicolons, parentheses."""

    def test_hyphen_split(self) -> None:
        """Hyphens split into separate tokens — needed for 'Li-Fe-P-O'."""
        result = tokenise("Li-Fe-P-O")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["li", "fe", "p", "o"]

    def test_slash_split(self) -> None:
        """Slashes split tokens — common in 'eV/atom' style units."""
        result = tokenise("eV/atom")
        assert "ev" in result
        assert "atom" in result

    def test_comma_split(self) -> None:
        """Comma-separated items become individual tokens."""
        result = tokenise("crystal,lattice,phonon")
        unigrams = [t for t in result if " " not in t]
        assert unigrams == ["crystal", "lattice", "phonon"]

    def test_semicolon_split(self) -> None:
        """Semicolons split tokens."""
        result = tokenise("clogp;tpsa")
        assert "clogp" in result
        assert "tpsa" in result
        assert "clogp tpsa" in result

    def test_parentheses_split(self) -> None:
        """Parenthesised content becomes separate tokens."""
        result = tokenise("molecule(smiles)")
        assert "molecule" in result
        assert "smiles" in result

    def test_mixed_delimiters(self) -> None:
        """Multiple different delimiters in one input all split correctly."""
        result = tokenise("crystal/lattice-phonon,band;gap(cif)")
        unigrams = [t for t in result if " " not in t]
        assert set(unigrams) == {"crystal", "lattice", "phonon", "band", "gap", "cif"}


# ---------------------------------------------------------------------------
# Bigram generation
# ---------------------------------------------------------------------------


class TestBigramGeneration:
    """Verify consecutive-pair bigrams from the unigram sequence."""

    def test_bigram_space_group(self) -> None:
        """'space group' bigram from adjacent words."""
        result = tokenise("the space group is P21")
        assert "space group" in result

    def test_bigram_lead_optimisation(self) -> None:
        """Multi-word keyword 'lead optimisation' detected as bigram."""
        result = tokenise("apply lead optimisation")
        assert "lead optimisation" in result

    def test_bigram_convex_hull(self) -> None:
        """'convex hull' bigram from domain text."""
        result = tokenise("check convex hull stability")
        assert "convex hull" in result

    def test_bigram_count(self) -> None:
        """n unigrams produce exactly n-1 bigrams."""
        result = tokenise("a b c d")
        unigrams = [t for t in result if " " not in t]
        bigrams = [t for t in result if " " in t]
        assert len(unigrams) == 4
        assert len(bigrams) == 3

    def test_single_word_no_bigrams(self) -> None:
        """Single unigram produces zero bigrams."""
        result = tokenise("crystal")
        assert result == ["crystal"]

    def test_bigrams_respect_delimiter_boundaries(self) -> None:
        """Bigrams form across delimiter boundaries, not raw text adjacency.

        'Li-Fe' splits into ['li', 'fe'] → bigram 'li fe', not 'li-fe'.
        """
        result = tokenise("Li-Fe")
        assert "li fe" in result
        # The hyphenated form must NOT appear
        assert "li-fe" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Adversarial and boundary inputs."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list — no phantom tokens."""
        assert tokenise("") == []

    def test_only_delimiters(self) -> None:
        """String of pure delimiters returns empty list."""
        assert tokenise("---///,,,;;;()") == []

    def test_leading_trailing_delimiters(self) -> None:
        """Leading/trailing delimiters do not produce empty tokens."""
        result = tokenise("-crystal-")
        assert result == ["crystal"]

    def test_lowercasing(self) -> None:
        """All tokens are lowercased regardless of input casing."""
        result = tokenise("CRYSTAL CIF SMILES")
        unigrams = [t for t in result if " " not in t]
        assert all(t == t.lower() for t in unigrams)
        assert "crystal" in result
        assert "cif" in result
        assert "smiles" in result

    def test_unicode_passthrough(self) -> None:
        """Non-ASCII characters pass through lowercased, not dropped."""
        result = tokenise("Ångström café")
        assert "ångström" in result
        assert "café" in result
