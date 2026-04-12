"""Tests for parse_max_atoms and parse_stability_threshold.

The numeric parsers are regex-based and heuristic, so the test effort
is concentrated on boundary patterns, ambiguous inputs, and the default
fallback path — not trivial happy-path confirmations.
"""

from __future__ import annotations

from ammd.materials.parse_numeric import parse_max_atoms, parse_stability_threshold


# ---------------------------------------------------------------------------
# parse_max_atoms — pattern coverage
# ---------------------------------------------------------------------------


def test_max_atoms_leq_operator():
    assert parse_max_atoms("<=20 atoms") == 20


def test_max_atoms_unicode_leq():
    assert parse_max_atoms("\u226415 atoms") == 15


def test_max_atoms_lt_operator():
    """'<30 atoms' — less-than without equals is still accepted."""
    assert parse_max_atoms("<30 atoms") == 30


def test_max_atoms_up_to():
    assert parse_max_atoms("up to 25 atoms") == 25


def test_max_atoms_at_most():
    assert parse_max_atoms("at most 10 atoms") == 10


def test_max_atoms_max_keyword():
    assert parse_max_atoms("max 50 atoms") == 50


def test_max_atoms_maximum_keyword():
    assert parse_max_atoms("maximum 12 atoms") == 12


def test_max_atoms_per_cell_suffix():
    """/cell suffix is tolerated — units are always atoms per cell."""
    assert parse_max_atoms("up to 15 atoms/cell") == 15


def test_max_atoms_singular():
    """'atom' (singular) is accepted alongside 'atoms'."""
    assert parse_max_atoms("<=1 atom") == 1


# ---------------------------------------------------------------------------
# parse_max_atoms — defaults
# ---------------------------------------------------------------------------


def test_max_atoms_empty_string():
    assert parse_max_atoms("") == 20


def test_max_atoms_whitespace_only():
    assert parse_max_atoms("   ") == 20


def test_max_atoms_no_match():
    assert parse_max_atoms("cubic perovskites ABO3") == 20


def test_max_atoms_custom_default():
    assert parse_max_atoms("no match here", default=50) == 50


# ---------------------------------------------------------------------------
# parse_max_atoms — edge cases
# ---------------------------------------------------------------------------


def test_max_atoms_embedded_in_longer_text():
    text = "Generate oxides in Li-Fe-P-O, <=20 atoms, space group 62"
    assert parse_max_atoms(text) == 20


def test_max_atoms_case_insensitive():
    assert parse_max_atoms("Up To 40 Atoms") == 40


def test_max_atoms_bare_number_ignored():
    """A bare number like '20' should not match without 'atoms'."""
    assert parse_max_atoms("space group 20") == 20  # default, not SG 20


# ---------------------------------------------------------------------------
# parse_stability_threshold — pattern coverage
# ---------------------------------------------------------------------------


def test_threshold_stable_within():
    assert abs(parse_stability_threshold("stable within 0.05 eV/atom") - 0.05) < 1e-9


def test_threshold_stability_keyword():
    assert abs(parse_stability_threshold("stability threshold 0.1 eV") - 0.1) < 1e-9


def test_threshold_stability_lt():
    assert abs(parse_stability_threshold("stability < 0.03 eV") - 0.03) < 1e-9


def test_threshold_keyword_alone():
    assert abs(parse_stability_threshold("threshold 0.2 eV") - 0.2) < 1e-9


def test_threshold_standalone_ev_per_atom():
    """'0.05 eV/atom' without a keyword prefix still matches."""
    assert abs(parse_stability_threshold("0.05 eV/atom") - 0.05) < 1e-9


def test_threshold_integer_value():
    """Integer values like '1 eV/atom' are accepted."""
    assert abs(parse_stability_threshold("stable within 1 eV/atom") - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# parse_stability_threshold — defaults
# ---------------------------------------------------------------------------


def test_threshold_empty_string():
    assert abs(parse_stability_threshold("") - 0.1) < 1e-9


def test_threshold_whitespace_only():
    assert abs(parse_stability_threshold("   ") - 0.1) < 1e-9


def test_threshold_no_match():
    assert abs(parse_stability_threshold("cubic perovskites ABO3") - 0.1) < 1e-9


def test_threshold_custom_default():
    assert abs(parse_stability_threshold("no match", default=0.5) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# parse_stability_threshold — edge cases
# ---------------------------------------------------------------------------


def test_threshold_embedded_in_longer_text():
    text = "Generate oxides, stable within 0.05 eV/atom, <=20 atoms"
    assert abs(parse_stability_threshold(text) - 0.05) < 1e-9


def test_threshold_case_insensitive():
    assert abs(parse_stability_threshold("Stable Within 0.05 EV/atom") - 0.05) < 1e-9


def test_threshold_ev_without_number_ignored():
    """'eV' alone without a preceding number should not match."""
    assert abs(parse_stability_threshold("energy in eV units") - 0.1) < 1e-9
