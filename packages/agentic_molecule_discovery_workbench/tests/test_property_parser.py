"""Tests for parse_property_constraint — the property window token parser.

Organised by risk surface: most effort on the tricky regex parsing and
error paths, less on trivial happy-path variants.
"""

from __future__ import annotations

import pytest

from agentic_molecule_discovery.constraints.models import ALLOWED_PROPERTIES, PropertyConstraint
from agentic_molecule_discovery.constraints.property_parser import parse_property_constraint


# ---------------------------------------------------------------------------
# Happy path: single operators
# ---------------------------------------------------------------------------


class TestLessThan:
    """Tokens with < and <= operators produce max_val-only constraints."""

    def test_less_than_integer(self) -> None:
        result = parse_property_constraint("MW<500")
        assert result == PropertyConstraint("MW", min_val=None, max_val=500.0)

    def test_less_than_or_equal_float(self) -> None:
        result = parse_property_constraint("logP<=3.5")
        assert result == PropertyConstraint("logP", min_val=None, max_val=3.5)

    def test_less_than_with_spaces(self) -> None:
        result = parse_property_constraint("  TPSA  <  140  ")
        assert result == PropertyConstraint("TPSA", min_val=None, max_val=140.0)


class TestGreaterThan:
    """Tokens with > and >= operators produce min_val-only constraints."""

    def test_greater_than_integer(self) -> None:
        result = parse_property_constraint("HBD>0")
        assert result == PropertyConstraint("HBD", min_val=0.0, max_val=None)

    def test_greater_than_or_equal_float(self) -> None:
        result = parse_property_constraint("logP>=1.5")
        assert result == PropertyConstraint("logP", min_val=1.5, max_val=None)


class TestEquals:
    """Tokens with = produce constraints where min_val == max_val."""

    def test_equals_integer(self) -> None:
        result = parse_property_constraint("HBD=2")
        assert result == PropertyConstraint("HBD", min_val=2.0, max_val=2.0)

    def test_equals_float(self) -> None:
        result = parse_property_constraint("logP=0.0")
        assert result == PropertyConstraint("logP", min_val=0.0, max_val=0.0)


# ---------------------------------------------------------------------------
# Happy path: range form
# ---------------------------------------------------------------------------


class TestRangeForm:
    """Tokens like '100<=TPSA<=140' produce two-sided constraints."""

    def test_range_integers(self) -> None:
        result = parse_property_constraint("100<=TPSA<=140")
        assert result == PropertyConstraint("TPSA", min_val=100.0, max_val=140.0)

    def test_range_floats(self) -> None:
        result = parse_property_constraint("1.0<=logP<=5.0")
        assert result == PropertyConstraint("logP", min_val=1.0, max_val=5.0)

    def test_range_with_spaces(self) -> None:
        result = parse_property_constraint(" 200 <= MW <= 500 ")
        assert result == PropertyConstraint("MW", min_val=200.0, max_val=500.0)

    def test_range_same_bounds(self) -> None:
        """Degenerate range where lo == hi is valid (equivalent to =)."""
        result = parse_property_constraint("3<=HBD<=3")
        assert result == PropertyConstraint("HBD", min_val=3.0, max_val=3.0)


# ---------------------------------------------------------------------------
# All allowed property names work
# ---------------------------------------------------------------------------


class TestAllPropertyNames:
    """Every name in ALLOWED_PROPERTIES is accepted by the parser."""

    @pytest.mark.parametrize("name", sorted(ALLOWED_PROPERTIES))
    def test_each_allowed_property(self, name: str) -> None:
        result = parse_property_constraint(f"{name}<100")
        assert result.property_name == name


# ---------------------------------------------------------------------------
# Error paths — where the risk surface is
# ---------------------------------------------------------------------------


class TestUnknownProperty:
    """Unknown or misspelled property names raise ValueError."""

    def test_completely_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown property 'foo'") as exc_info:
            parse_property_constraint("foo<5")
        assert "foo" in str(exc_info.value)

    def test_wrong_case(self) -> None:
        """Property names are case-sensitive: 'mw' != 'MW'."""
        with pytest.raises(ValueError, match="Unknown property 'mw'") as exc_info:
            parse_property_constraint("mw<500")
        assert "mw" in str(exc_info.value)

    def test_unknown_in_range(self) -> None:
        with pytest.raises(ValueError, match="Unknown property 'XYZ'") as exc_info:
            parse_property_constraint("1<=XYZ<=10")
        assert "XYZ" in str(exc_info.value)


class TestMalformedTokens:
    """Tokens that don't match any expected form raise ValueError."""

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="empty") as exc_info:
            parse_property_constraint("")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="empty") as exc_info:
            parse_property_constraint("   ")
        assert "empty" in str(exc_info.value).lower()

    def test_no_operator(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("MW500")
        assert "MW500" in str(exc_info.value)

    def test_double_operator(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("MW<<500")
        assert "MW<<500" in str(exc_info.value)

    def test_missing_value(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("MW<")
        assert "MW<" in str(exc_info.value)

    def test_missing_property(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("<500")
        assert "<500" in str(exc_info.value)

    def test_range_missing_lower(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("<=MW<=500")
        assert "<=MW<=500" in str(exc_info.value)

    def test_range_with_gt_operator(self) -> None:
        """Range form only supports <=, not >= or mixed operators."""
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("100>=MW>=50")
        assert "100>=MW>=50" in str(exc_info.value)

    def test_just_a_number(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("500")
        assert "500" in str(exc_info.value)

    def test_just_a_name(self) -> None:
        with pytest.raises(ValueError, match="Malformed") as exc_info:
            parse_property_constraint("MW")
        assert "MW" in str(exc_info.value)


class TestRangeInversion:
    """Range where lo > hi is rejected by PropertyConstraint validation."""

    def test_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="min_val.*max_val") as exc_info:
            parse_property_constraint("500<=MW<=200")
        assert "min_val" in str(exc_info.value)


class TestNegativeValues:
    """Negative numbers are valid in property constraints."""

    def test_negative_lower_bound(self) -> None:
        result = parse_property_constraint("-2.0<=logP<=5.0")
        assert result == PropertyConstraint("logP", min_val=-2.0, max_val=5.0)

    def test_negative_single_value(self) -> None:
        result = parse_property_constraint("logP>=-1.5")
        assert result == PropertyConstraint("logP", min_val=-1.5, max_val=None)
