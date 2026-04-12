"""Tests for discovery_workbench.schemas — construction paths and validation edges."""

import pytest
from dataclasses import FrozenInstanceError

from discovery_workbench.schemas import (
    MaterialsRequest,
    MoleculeRequest,
    RoutingResult,
)


def test_molecule_request_mandatory_fields() -> None:
    """Missing or invalid mandatory fields must raise, not silently default."""
    # Empty task_type
    with pytest.raises(ValueError, match="task_type is required"):
        MoleculeRequest(task_type="", objective="x", constraints={}, output_count=5)

    # Invalid task_type
    with pytest.raises(ValueError, match="task_type must be one of"):
        MoleculeRequest(
            task_type="invalid", objective="test", constraints={}, output_count=5
        )

    # Empty objective
    with pytest.raises(ValueError, match="objective is required"):
        MoleculeRequest(
            task_type="de_novo", objective="", constraints={}, output_count=5
        )

    # Non-positive output_count
    with pytest.raises(ValueError, match="output_count must be a positive integer"):
        MoleculeRequest(
            task_type="de_novo", objective="test", constraints={}, output_count=0
        )

    # constraints wrong type
    with pytest.raises(TypeError, match="constraints must be a dict"):
        MoleculeRequest(
            task_type="de_novo", objective="test",
            constraints="not a dict",  # type: ignore[arg-type]
            output_count=5,
        )

    # Valid construction succeeds and stores the right values
    valid = MoleculeRequest(
        task_type="de_novo", objective="inhibit CDK2",
        constraints={"MW": "300-450"}, output_count=10,
    )
    assert valid.task_type == "de_novo"
    assert valid.objective == "inhibit CDK2"
    assert valid.constraints == {"MW": "300-450"}
    assert valid.output_count == 10


def test_molecule_request_defaults() -> None:
    """Default values match the Q2 spec exactly."""
    req = MoleculeRequest(
        task_type="de_novo", objective="find potent inhibitor",
        constraints={"MW": "300-450"}, output_count=10,
    )
    assert req.reference_set == "ChEMBL_36"
    assert req.starting_scaffold is None
    assert req.starting_molecules is None
    assert req.property_priorities is None
    # Mutable — downstream enrichment must be possible
    req.reference_set = "ChEMBL_35"
    assert req.reference_set == "ChEMBL_35"


def test_materials_request_mandatory_fields() -> None:
    """Missing or invalid mandatory fields must raise."""
    # Empty chemistry_scope
    with pytest.raises(ValueError, match="chemistry_scope is required"):
        MaterialsRequest(chemistry_scope="", output_count=5)

    # Non-positive output_count
    with pytest.raises(ValueError, match="output_count must be a positive integer"):
        MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=-1)

    # Invalid structure_size_limit
    with pytest.raises(ValueError, match="structure_size_limit must be a positive integer"):
        MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=5, structure_size_limit=0)

    # Negative stability_target
    with pytest.raises(ValueError, match="stability_target must be a non-negative number"):
        MaterialsRequest(
            chemistry_scope="Li-Fe-O", output_count=5, stability_target=-0.5
        )

    # Valid construction succeeds and stores the right values
    valid = MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=5)
    assert valid.chemistry_scope == "Li-Fe-O"
    assert valid.output_count == 5


def test_materials_request_defaults() -> None:
    """Default values match the Q2 spec exactly."""
    req = MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=10)
    assert req.structure_size_limit == 20
    assert req.symmetry_request == "any"
    assert req.stability_target == 0.1
    assert req.band_gap_eV is None
    assert req.bulk_modulus_GPa is None
    assert req.magnetic_density is None
    assert req.exclude_elements is None
    assert req.allow_P1 is False


def test_routing_result_frozen() -> None:
    """RoutingResult is frozen — field assignment must raise."""
    mol = MoleculeRequest(
        task_type="de_novo", objective="test", constraints={}, output_count=5,
    )
    result = RoutingResult(
        domain="small_molecule_design", confidence=0.95,
        clarification_question=None, parsed_request=mol,
    )
    assert result.domain == "small_molecule_design"
    assert result.confidence == 0.95

    with pytest.raises(FrozenInstanceError):
        result.domain = "unsupported"  # type: ignore[misc]

    # Invalid domain rejected
    with pytest.raises(ValueError, match="domain must be one of"):
        RoutingResult(
            domain="bad_domain", confidence=0.5,
            clarification_question=None, parsed_request=None,
        )

    # Confidence out of range
    with pytest.raises(ValueError, match="confidence must be in"):
        RoutingResult(
            domain="unsupported", confidence=1.5,
            clarification_question=None, parsed_request=None,
        )


def test_routing_result_unsupported_has_no_parsed_request() -> None:
    """An 'unsupported' routing result should carry no parsed request.

    This is a convention test — the dataclass does not enforce it, but
    callers constructing unsupported results should follow this pattern.
    """
    result = RoutingResult(
        domain="unsupported", confidence=0.3,
        clarification_question="Could you clarify the target domain?",
        parsed_request=None,
    )
    assert result.domain == "unsupported"
    assert result.parsed_request is None
    assert result.clarification_question is not None
