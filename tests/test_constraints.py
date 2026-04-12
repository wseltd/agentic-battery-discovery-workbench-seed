"""Tests for discovery_workbench.constraints — parse_range and normalise_unit."""

from discovery_workbench.constraints import ParsedRange, normalise_unit, parse_range

import pytest


def test_parse_plain_range():
    """Plain numeric range without unit."""
    result = parse_range("300-450")
    assert result == ParsedRange(min_val=300.0, max_val=450.0, unit=None)


def test_parse_less_equal():
    """Less-than-or-equal operator."""
    result = parse_range("<=0.1")
    assert result == ParsedRange(min_val=None, max_val=0.1, unit=None)


def test_parse_greater_equal():
    """Greater-than-or-equal operator."""
    result = parse_range(">=5")
    assert result == ParsedRange(min_val=5.0, max_val=None, unit=None)


def test_parse_less_than():
    """Strict less-than operator."""
    result = parse_range("<10")
    assert result == ParsedRange(min_val=None, max_val=10.0, unit=None)


def test_parse_greater_than():
    """Strict greater-than operator."""
    result = parse_range(">2")
    assert result == ParsedRange(min_val=2.0, max_val=None, unit=None)


def test_parse_range_with_unit():
    """Range with a unit that is already canonical."""
    result = parse_range("5-6 eV")
    assert result == ParsedRange(min_val=5.0, max_val=6.0, unit="eV")


def test_parse_range_with_unit_normalised():
    """Range with a unit variant that should be normalised."""
    result = parse_range("300-450 daltons")
    assert result == ParsedRange(min_val=300.0, max_val=450.0, unit="Da")


def test_parse_decimal_range():
    """Range with decimal values to exercise float parsing."""
    result = parse_range("0.5-1.5 GPa")
    assert result == ParsedRange(min_val=0.5, max_val=1.5, unit="GPa")


def test_parse_invalid_raises_valueerror():
    """Unparseable input must raise ValueError with helpful message."""
    with pytest.raises(ValueError, match="Cannot parse range") as exc_info:
        parse_range("not a range")
    assert "not a range" in str(exc_info.value)

    with pytest.raises(ValueError, match="Cannot parse range"):
        parse_range("")


def test_normalise_known_alias():
    """Known alias maps to canonical form."""
    assert normalise_unit("daltons") == "Da"
    assert normalise_unit("g/mol") == "Da"
    assert normalise_unit("angstrom^2") == "Å²"


def test_normalise_case_insensitive():
    """Case-insensitive lookup for unit aliases."""
    assert normalise_unit("gpa") == "GPa"
    assert normalise_unit("GPA") == "GPa"
    assert normalise_unit("ev") == "eV"


def test_normalise_unknown_passthrough():
    """Unknown units are returned unchanged."""
    assert normalise_unit("kg") == "kg"
    assert normalise_unit("m/s") == "m/s"
