"""Tests for parse_constraints — numeric ranges, SMARTS, elements, and rejection."""

import pytest

from discovery_workbench.shared.constraints import (
    ConstraintParseError,
    NumericRange,
    parse_constraints,
)


# ---------------------------------------------------------------------------
# Numeric range parsing (3 tests)
# ---------------------------------------------------------------------------


def test_parse_mw_range_min_and_max():
    """MW constraint with both min and max produces a closed NumericRange."""
    result = parse_constraints({"MW": {"min": 200, "max": 500}})
    mw = result.numeric["MW"]
    assert isinstance(mw, NumericRange)
    assert mw.min_val == 200.0
    assert mw.max_val == 500.0


def test_parse_clogp_range_min_only():
    """cLogP with min-only leaves max_val as None (open-ended above)."""
    result = parse_constraints({"cLogP": {"min": 1.5}})
    clogp = result.numeric["cLogP"]
    assert clogp.min_val == 1.5
    assert clogp.max_val is None


def test_parse_tpsa_range_max_only():
    """TPSA with max-only leaves min_val as None (open-ended below)."""
    result = parse_constraints({"TPSA": {"max": 140}})
    tpsa = result.numeric["TPSA"]
    assert tpsa.min_val is None
    assert tpsa.max_val == 140.0


# ---------------------------------------------------------------------------
# SMARTS / element-list parsing (3 tests)
# ---------------------------------------------------------------------------


def test_parse_smarts_required():
    """A single required SMARTS pattern is stored in smarts_required."""
    result = parse_constraints({"smarts_required": ["c1ccccc1"]})
    assert result.smarts_required == ["c1ccccc1"]
    assert result.smarts_forbidden == []


def test_parse_smarts_forbidden():
    """A forbidden SMARTS pattern is stored in smarts_forbidden."""
    result = parse_constraints({"smarts_forbidden": ["[#7]"]})
    assert result.smarts_forbidden == ["[#7]"]
    assert result.smarts_required == []


def test_parse_element_whitelist():
    """Element whitelist is stored verbatim in elements_allowed."""
    elements = ["Li", "Fe", "P", "O"]
    result = parse_constraints({"elements_allowed": elements})
    assert result.elements_allowed == ["Li", "Fe", "P", "O"]
    assert result.elements_excluded is None


# ---------------------------------------------------------------------------
# Malformed-input rejection (3 tests)
# ---------------------------------------------------------------------------


def test_reject_numeric_range_min_exceeds_max():
    """min > max is an impossible range — must raise with descriptive message."""
    with pytest.raises(ConstraintParseError, match="min.*>.*max") as exc_info:
        parse_constraints({"MW": {"min": 500, "max": 200}})
    msg = str(exc_info.value)
    assert "500" in msg
    assert "200" in msg


def test_reject_invalid_smarts_pattern():
    """Unbalanced brackets are not valid SMARTS — must raise, not propagate RDKit error."""
    with pytest.raises(ConstraintParseError, match="Invalid SMARTS") as exc_info:
        parse_constraints({"smarts_required": ["[[[invalid"]})
    msg = str(exc_info.value)
    assert "[[[invalid" in msg


def test_reject_wrong_type_for_numeric_field():
    """A string where a float is expected must raise with the offending value."""
    with pytest.raises(ConstraintParseError, match="non-numeric") as exc_info:
        parse_constraints({"MW": {"min": "not_a_number", "max": 500}})
    msg = str(exc_info.value)
    assert "not_a_number" in msg
