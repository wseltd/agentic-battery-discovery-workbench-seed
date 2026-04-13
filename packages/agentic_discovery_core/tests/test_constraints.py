"""Tests for parse_constraints -- numeric ranges, SMARTS, elements, and rejection."""

import pytest

from agentic_discovery_core.shared.constraints import (
    ConstraintParseError,
    NumericRange,
    parse_constraints,
)


def test_parse_mw_range_min_and_max():
    result = parse_constraints({"MW": {"min": 200, "max": 500}})
    mw = result.numeric["MW"]
    assert isinstance(mw, NumericRange)
    assert mw.min_val == 200.0
    assert mw.max_val == 500.0


def test_parse_clogp_range_min_only():
    result = parse_constraints({"cLogP": {"min": 1.5}})
    clogp = result.numeric["cLogP"]
    assert clogp.min_val == 1.5
    assert clogp.max_val is None


def test_parse_tpsa_range_max_only():
    result = parse_constraints({"TPSA": {"max": 140}})
    tpsa = result.numeric["TPSA"]
    assert tpsa.min_val is None
    assert tpsa.max_val == 140.0


def test_parse_element_whitelist():
    elements = ["Li", "Fe", "P", "O"]
    result = parse_constraints({"elements_allowed": elements})
    assert result.elements_allowed == ["Li", "Fe", "P", "O"]
    assert result.elements_excluded is None


def test_reject_numeric_range_min_exceeds_max():
    with pytest.raises(ConstraintParseError, match="min.*>.*max") as exc_info:
        parse_constraints({"MW": {"min": 500, "max": 200}})
    msg = str(exc_info.value)
    assert "500" in msg
    assert "200" in msg


def test_reject_wrong_type_for_numeric_field():
    with pytest.raises(ConstraintParseError, match="non-numeric") as exc_info:
        parse_constraints({"MW": {"min": "not_a_number", "max": 500}})
    msg = str(exc_info.value)
    assert "not_a_number" in msg
