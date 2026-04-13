"""Tests for agentic_discovery_core.schemas -- construction paths and validation edges."""

import pytest
from dataclasses import FrozenInstanceError

from agentic_discovery_core.schemas import (
    MaterialsRequest,
    MoleculeRequest,
    RoutingResult,
)


def test_molecule_request_mandatory_fields() -> None:
    with pytest.raises(ValueError, match="task_type is required"):
        MoleculeRequest(task_type="", objective="x", constraints={}, output_count=5)

    with pytest.raises(ValueError, match="task_type must be one of"):
        MoleculeRequest(
            task_type="invalid", objective="test", constraints={}, output_count=5
        )

    with pytest.raises(ValueError, match="objective is required"):
        MoleculeRequest(
            task_type="de_novo", objective="", constraints={}, output_count=5
        )

    with pytest.raises(ValueError, match="output_count must be a positive integer"):
        MoleculeRequest(
            task_type="de_novo", objective="test", constraints={}, output_count=0
        )

    with pytest.raises(TypeError, match="constraints must be a dict"):
        MoleculeRequest(
            task_type="de_novo", objective="test",
            constraints="not a dict",
            output_count=5,
        )

    valid = MoleculeRequest(
        task_type="de_novo", objective="inhibit CDK2",
        constraints={"MW": "300-450"}, output_count=10,
    )
    assert valid.task_type == "de_novo"
    assert valid.objective == "inhibit CDK2"
    assert valid.constraints == {"MW": "300-450"}
    assert valid.output_count == 10


def test_molecule_request_defaults() -> None:
    req = MoleculeRequest(
        task_type="de_novo", objective="find potent inhibitor",
        constraints={"MW": "300-450"}, output_count=10,
    )
    assert req.reference_set == "ChEMBL_36"
    assert req.starting_scaffold is None
    assert req.starting_molecules is None
    assert req.property_priorities is None
    req.reference_set = "ChEMBL_35"
    assert req.reference_set == "ChEMBL_35"


def test_materials_request_mandatory_fields() -> None:
    with pytest.raises(ValueError, match="chemistry_scope is required"):
        MaterialsRequest(chemistry_scope="", output_count=5)

    with pytest.raises(ValueError, match="output_count must be a positive integer"):
        MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=-1)

    with pytest.raises(ValueError, match="structure_size_limit must be a positive integer"):
        MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=5, structure_size_limit=0)

    with pytest.raises(ValueError, match="stability_target must be a non-negative number"):
        MaterialsRequest(
            chemistry_scope="Li-Fe-O", output_count=5, stability_target=-0.5
        )

    valid = MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=5)
    assert valid.chemistry_scope == "Li-Fe-O"
    assert valid.output_count == 5


def test_materials_request_defaults() -> None:
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
        result.domain = "unsupported"

    with pytest.raises(ValueError, match="domain must be one of"):
        RoutingResult(
            domain="bad_domain", confidence=0.5,
            clarification_question=None, parsed_request=None,
        )

    with pytest.raises(ValueError, match="confidence must be in"):
        RoutingResult(
            domain="unsupported", confidence=1.5,
            clarification_question=None, parsed_request=None,
        )


def test_routing_result_unsupported_has_no_parsed_request() -> None:
    result = RoutingResult(
        domain="unsupported", confidence=0.3,
        clarification_question="Could you clarify the target domain?",
        parsed_request=None,
    )
    assert result.domain == "unsupported"
    assert result.parsed_request is None
    assert result.clarification_question is not None


def test_routing_result_unsupported_rejects_parsed_request() -> None:
    mol = MoleculeRequest(
        task_type="de_novo", objective="test", constraints={}, output_count=5,
    )
    with pytest.raises(ValueError, match="parsed_request must be None"):
        RoutingResult(
            domain="unsupported", confidence=0.3,
            clarification_question=None, parsed_request=mol,
        )


def test_routing_result_domain_request_type_mismatch() -> None:
    mat = MaterialsRequest(chemistry_scope="Li-Fe-O", output_count=5)
    mol = MoleculeRequest(
        task_type="de_novo", objective="test", constraints={}, output_count=5,
    )

    with pytest.raises(TypeError, match="must be a MoleculeRequest"):
        RoutingResult(
            domain="small_molecule_design", confidence=0.75,
            clarification_question=None, parsed_request=mat,
        )

    with pytest.raises(TypeError, match="must be a MaterialsRequest"):
        RoutingResult(
            domain="inorganic_materials_design", confidence=0.75,
            clarification_question=None, parsed_request=mol,
        )

    with pytest.raises(TypeError, match="must be a MoleculeRequest"):
        RoutingResult(
            domain="small_molecule_design", confidence=0.75,
            clarification_question=None, parsed_request=None,
        )


def test_routing_result_clarification_rejected_at_high_confidence() -> None:
    mol = MoleculeRequest(
        task_type="de_novo", objective="test", constraints={}, output_count=5,
    )
    with pytest.raises(ValueError, match="clarification_question must be None"):
        RoutingResult(
            domain="small_molecule_design", confidence=0.95,
            clarification_question="Are you sure?", parsed_request=mol,
        )


def test_routing_result_clarification_allowed_at_low_confidence() -> None:
    mol = MoleculeRequest(
        task_type="de_novo", objective="test", constraints={}, output_count=5,
    )
    result = RoutingResult(
        domain="small_molecule_design", confidence=0.6,
        clarification_question="Did you mean de novo?", parsed_request=mol,
    )
    assert result.clarification_question == "Did you mean de novo?"


def test_routing_result_clarification_allowed_when_no_parsed_request() -> None:
    result = RoutingResult(
        domain="unsupported", confidence=0.3,
        clarification_question="Could you clarify?", parsed_request=None,
    )
    assert result.clarification_question == "Could you clarify?"
