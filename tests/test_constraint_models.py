"""Tests for molecular_constraints.models — PropertyConstraint, SubstructureConstraint."""

import pytest

from molecular_constraints.models import (
    ALLOWED_PROPERTIES,
    ConstraintList,
    PropertyConstraint,
    SubstructureConstraint,
)


# ---------------------------------------------------------------------------
# ALLOWED_PROPERTIES constant
# ---------------------------------------------------------------------------


class TestAllowedProperties:
    """Verify the canonical property set is correct and stable."""

    def test_contains_all_expected_names(self):
        expected = {
            "MW", "logP", "TPSA", "HBD", "HBA",
            "rotatable_bonds", "ring_count", "aromatic_rings",
        }
        assert ALLOWED_PROPERTIES == expected

    def test_is_frozenset(self):
        assert isinstance(ALLOWED_PROPERTIES, frozenset)


# ---------------------------------------------------------------------------
# PropertyConstraint — valid construction
# ---------------------------------------------------------------------------


class TestPropertyConstraintValid:
    """PropertyConstraint accepts well-formed inputs."""

    def test_both_bounds(self):
        pc = PropertyConstraint("MW", min_val=100.0, max_val=500.0)
        assert pc.property_name == "MW"
        assert pc.min_val == 100.0
        assert pc.max_val == 500.0

    def test_min_only(self):
        pc = PropertyConstraint("logP", min_val=-2.0, max_val=None)
        assert pc.min_val == -2.0
        assert pc.max_val is None

    def test_max_only(self):
        pc = PropertyConstraint("TPSA", min_val=None, max_val=140.0)
        assert pc.min_val is None
        assert pc.max_val == 140.0

    def test_equal_bounds(self):
        """min_val == max_val is a valid point constraint."""
        pc = PropertyConstraint("HBD", min_val=3.0, max_val=3.0)
        assert pc.min_val == pc.max_val == 3.0

    @pytest.mark.parametrize("prop", sorted(ALLOWED_PROPERTIES))
    def test_every_allowed_property_accepted(self, prop):
        pc = PropertyConstraint(prop, min_val=0.0, max_val=None)
        assert pc.property_name == prop

    def test_slots_prevent_arbitrary_attrs(self):
        """__slots__ prevents adding attributes not in the schema."""
        pc = PropertyConstraint("MW", min_val=100.0, max_val=None)
        with pytest.raises(AttributeError):
            pc.extra = "nope"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# PropertyConstraint — validation failures
# ---------------------------------------------------------------------------


class TestPropertyConstraintRejection:
    """PropertyConstraint rejects malformed inputs eagerly."""

    def test_unknown_property_name(self):
        with pytest.raises(ValueError, match="Unknown property") as exc_info:
            PropertyConstraint("solubility", min_val=0.0, max_val=None)
        assert "solubility" in str(exc_info.value)

    def test_empty_property_name(self):
        with pytest.raises(ValueError, match="Unknown property"):
            PropertyConstraint("", min_val=0.0, max_val=None)

    def test_case_sensitive_property_name(self):
        """Property names are case-sensitive — 'mw' is not 'MW'."""
        with pytest.raises(ValueError, match="Unknown property"):
            PropertyConstraint("mw", min_val=100.0, max_val=None)

    def test_both_bounds_none(self):
        with pytest.raises(ValueError, match="at least one of min_val or max_val"):
            PropertyConstraint("MW", min_val=None, max_val=None)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_val.*>.*max_val") as exc_info:
            PropertyConstraint("logP", min_val=5.0, max_val=1.0)
        assert "5.0" in str(exc_info.value)
        assert "1.0" in str(exc_info.value)

    def test_error_message_lists_allowed(self):
        """Rejection message should tell the caller what IS allowed."""
        with pytest.raises(ValueError, match="Allowed") as exc_info:
            PropertyConstraint("bogus", min_val=0.0, max_val=None)
        assert "MW" in str(exc_info.value)


# ---------------------------------------------------------------------------
# SubstructureConstraint — valid construction
# ---------------------------------------------------------------------------


class TestSmartsValid:
    """SubstructureConstraint accepts valid SMARTS."""

    def test_simple_ring_smarts(self):
        sc = SubstructureConstraint("[#6]1[#6][#6][#6][#6][#6]1")
        assert sc.must_match is True

    def test_must_match_false(self):
        sc = SubstructureConstraint("[#7]", must_match=False)
        assert sc.must_match is False
        assert sc.smarts == "[#7]"

    def test_aromatic_ring(self):
        sc = SubstructureConstraint("c1ccccc1")
        assert sc.smarts == "c1ccccc1"

    def test_complex_smarts(self):
        """A real-world pharmacophore-style SMARTS pattern."""
        smarts = "[#6](=O)[OX2H1]"  # carboxylic acid
        sc = SubstructureConstraint(smarts)
        assert sc.smarts == smarts

    def test_frozen(self):
        sc = SubstructureConstraint("[#6]")
        with pytest.raises(AttributeError):
            sc.smarts = "[#7]"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SubstructureConstraint — validation failures
# ---------------------------------------------------------------------------


class TestSmartsInvalid:
    """SubstructureConstraint rejects bad SMARTS eagerly."""

    def test_empty_smarts(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SubstructureConstraint("")

    def test_unparseable_smarts(self):
        with pytest.raises(ValueError, match="Invalid SMARTS") as exc_info:
            SubstructureConstraint("[[[invalid")
        assert "[[[invalid" in str(exc_info.value)

    def test_smiles_not_smarts_still_valid(self):
        """Plain SMILES are valid SMARTS (subset) — should not reject."""
        sc = SubstructureConstraint("CCO")
        assert sc.smarts == "CCO"

    def test_whitespace_only_smarts(self):
        """Whitespace-only is effectively empty."""
        # RDKit treats whitespace SMARTS as unparseable, so this should fail
        # on either the emptiness check or the parse check.
        with pytest.raises(ValueError):
            SubstructureConstraint("   ")


# ---------------------------------------------------------------------------
# ConstraintList type alias
# ---------------------------------------------------------------------------


class TestConstraintList:
    """ConstraintList is a plain list accepting both constraint types."""

    def test_mixed_list(self):
        constraints: ConstraintList = [
            PropertyConstraint("MW", min_val=200.0, max_val=500.0),
            SubstructureConstraint("c1ccccc1"),
            PropertyConstraint("logP", min_val=None, max_val=5.0),
            SubstructureConstraint("[#7]", must_match=False),
        ]
        assert len(constraints) == 4
        assert isinstance(constraints[0], PropertyConstraint)
        assert isinstance(constraints[1], SubstructureConstraint)

    def test_empty_list(self):
        constraints: ConstraintList = []
        assert constraints == []

    def test_property_only_list(self):
        constraints: ConstraintList = [
            PropertyConstraint("HBA", min_val=1.0, max_val=None),
            PropertyConstraint("HBD", min_val=None, max_val=5.0),
        ]
        assert all(isinstance(c, PropertyConstraint) for c in constraints)

    def test_substructure_only_list(self):
        constraints: ConstraintList = [
            SubstructureConstraint("c1ccccc1"),
            SubstructureConstraint("[#8]", must_match=False),
        ]
        assert all(isinstance(c, SubstructureConstraint) for c in constraints)
