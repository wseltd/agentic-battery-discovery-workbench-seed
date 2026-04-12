"""Tests for molecular_constraints.smarts_parser — parse_smarts_constraint."""

import pytest

from molecular_constraints.smarts_parser import parse_smarts_constraint


# --- Valid tokens: positive prefix ---


class TestPositivePrefix:
    """Tokens starting with 'has:' produce must_match=True."""

    def test_simple_smarts(self):
        result = parse_smarts_constraint("has:[#6]")
        assert result.smarts == "[#6]"
        assert result.must_match is True

    def test_aromatic_ring(self):
        result = parse_smarts_constraint("has:c1ccccc1")
        assert result.smarts == "c1ccccc1"
        assert result.must_match is True

    def test_complex_pattern(self):
        result = parse_smarts_constraint("has:[#6]1[#6][#6][#6][#6][#6]1")
        assert result.smarts == "[#6]1[#6][#6][#6][#6][#6]1"
        assert result.must_match is True

    def test_smiles_like_smarts(self):
        """Simple SMILES strings are valid SMARTS — RDKit accepts them."""
        result = parse_smarts_constraint("has:CCO")
        assert result.smarts == "CCO"
        assert result.must_match is True


# --- Valid tokens: negated prefix ---


class TestNegatedPrefix:
    """Tokens starting with '!has:' produce must_match=False."""

    def test_negated_simple(self):
        result = parse_smarts_constraint("!has:[#7]")
        assert result.smarts == "[#7]"
        assert result.must_match is False

    def test_negated_aromatic(self):
        result = parse_smarts_constraint("!has:c1ccccc1")
        assert result.smarts == "c1ccccc1"
        assert result.must_match is False


# --- Whitespace handling ---


class TestWhitespaceHandling:
    """Leading/trailing whitespace on token and SMARTS is stripped."""

    def test_leading_whitespace_on_token(self):
        result = parse_smarts_constraint("  has:[#6]")
        assert result.smarts == "[#6]"
        assert result.must_match is True

    def test_trailing_whitespace_on_token(self):
        result = parse_smarts_constraint("has:[#6]  ")
        assert result.smarts == "[#6]"
        assert result.must_match is True

    def test_whitespace_around_smarts(self):
        result = parse_smarts_constraint("has:  [#6]  ")
        assert result.smarts == "[#6]"
        assert result.must_match is True

    def test_negated_with_whitespace(self):
        result = parse_smarts_constraint("  !has:  [#7]  ")
        assert result.smarts == "[#7]"
        assert result.must_match is False


# --- Missing or wrong prefix ---


class TestBadPrefix:
    """Tokens without 'has:' or '!has:' prefix raise ValueError."""

    def test_no_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("[#6]")

    def test_wrong_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("contains:[#6]")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("")

    def test_only_whitespace(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("   ")

    def test_case_sensitive_prefix(self):
        """Prefix is case-sensitive — 'HAS:' is not valid."""
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("HAS:[#6]")

    def test_has_without_colon(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_smarts_constraint("has[#6]")


# --- Empty SMARTS after prefix ---


class TestEmptySmarts:
    """SMARTS portion that is empty or whitespace-only raises ValueError."""

    def test_positive_empty(self):
        with pytest.raises(ValueError, match="empty after prefix"):
            parse_smarts_constraint("has:")

    def test_negated_empty(self):
        with pytest.raises(ValueError, match="empty after prefix"):
            parse_smarts_constraint("!has:")

    def test_positive_whitespace_only(self):
        with pytest.raises(ValueError, match="empty after prefix"):
            parse_smarts_constraint("has:   ")

    def test_negated_whitespace_only(self):
        with pytest.raises(ValueError, match="empty after prefix"):
            parse_smarts_constraint("!has:   ")


# --- Invalid SMARTS patterns ---


class TestInvalidSmarts:
    """SMARTS strings that RDKit cannot parse raise ValueError."""

    def test_garbage_smarts(self):
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            parse_smarts_constraint("has:[[[invalid")

    def test_unbalanced_ring(self):
        """Unclosed ring digit — RDKit rejects this."""
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            parse_smarts_constraint("has:C1CC")

    def test_negated_invalid_smarts(self):
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            parse_smarts_constraint("!has:[[[invalid")
