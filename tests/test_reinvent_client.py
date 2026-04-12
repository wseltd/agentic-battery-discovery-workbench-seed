"""Tests for REINVENT 4 client: config building, output parsing, invocation.

Heavy coverage on parse_reinvent_output (the parser is where malformed
data creates risk) and build_config validation (wrong task_type is a
boundary error). Invocation tests use mocked subprocess since we cannot
run REINVENT in CI.
"""

from __future__ import annotations

import subprocess  # nosec B404 — testing subprocess-based REINVENT client
from unittest.mock import patch

import pytest

from agentic_discovery.molecules.reinvent_client import (
    GeneratedMolecule,
    Reinvent4Client,
    invoke_reinvent,
    parse_reinvent_output,
)
from agentic_discovery.shared.evidence import EvidenceLevel


# ---------------------------------------------------------------------------
# build_config tests
# ---------------------------------------------------------------------------


class TestBuildConfigDeNovo:
    def test_build_config_de_novo_contains_run_type(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        cfg = client.build_config("de_novo", {"num_molecules": 100})
        assert cfg["run_type"] == "sampling"
        assert cfg["task_type"] == "de_novo"
        assert cfg["parameters"]["num_molecules"] == 100


class TestBuildConfigScaffold:
    def test_build_config_scaffold_constrained_includes_scaffold(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        cfg = client.build_config(
            "scaffold_constrained", {"scaffold": "c1ccccc1"}
        )
        assert cfg["task_type"] == "scaffold_constrained"
        assert cfg["parameters"]["scaffold"] == "c1ccccc1"


class TestBuildConfigOptimise:
    def test_build_config_optimise_includes_starting_molecules(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        cfg = client.build_config(
            "optimise", {"starting_molecules": ["CCO", "CCN"]}
        )
        assert cfg["task_type"] == "optimise"
        assert cfg["parameters"]["starting_molecules"] == ["CCO", "CCN"]


class TestBuildConfigRejectsUnknown:
    def test_build_config_rejects_unknown_task_type(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        with pytest.raises(ValueError, match="Unknown task_type 'bogus'"):
            client.build_config("bogus", {})
        # Verify the error message names the bad value and the valid options
        try:
            client.build_config("bogus", {})
        except ValueError as exc:
            assert "bogus" in str(exc)
            assert "de_novo" in str(exc)


class TestBuildConfigConstraints:
    def test_build_config_injects_constraints_from_parsed_constraints(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        constraints = {"max_weight": 500, "min_logp": -1.0}
        cfg = client.build_config("de_novo", {}, constraints=constraints)
        assert cfg["constraints"] == {"max_weight": 500, "min_logp": -1.0}
        # Constraints dict should be a copy, not the same object
        assert cfg["constraints"] is not constraints


# ---------------------------------------------------------------------------
# parse_reinvent_output tests
# ---------------------------------------------------------------------------


class TestParseOutputValid:
    def test_parse_output_valid_csv_returns_generated_molecules(self) -> None:
        csv_text = "SMILES,Score\nCCO,0.85\nCCN,0.72\n"
        molecules = parse_reinvent_output(csv_text, "de_novo")
        assert len(molecules) == 2
        assert molecules[0].smiles == "CCO"
        assert molecules[0].score == pytest.approx(0.85)
        assert molecules[0].task_type == "de_novo"
        assert molecules[1].smiles == "CCN"
        assert molecules[1].score == pytest.approx(0.72)


class TestParseOutputEmpty:
    def test_parse_output_empty_returns_empty_list(self) -> None:
        result = parse_reinvent_output("", "de_novo")
        assert result == []
        # Whitespace-only should also be empty
        assert parse_reinvent_output("   \n  ", "de_novo") == []


class TestParseOutputMalformed:
    def test_parse_output_malformed_lines_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        csv_text = "SMILES,Score\nCCO,0.85\n,not_a_float\nCCN,0.72\n"
        with caplog.at_level("WARNING"):
            molecules = parse_reinvent_output(csv_text, "de_novo")
        assert len(molecules) == 2
        assert molecules[0].smiles == "CCO"
        assert molecules[1].smiles == "CCN"
        # The malformed line should have triggered a warning
        assert any("malformed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# invoke_reinvent tests
# ---------------------------------------------------------------------------


class TestInvokeReinventNonzeroExit:
    def test_invoke_reinvent_nonzero_exit_raises(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        config = {"task_type": "de_novo", "run_type": "sampling", "parameters": {}}
        with patch("agentic_discovery.molecules.reinvent_client.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["/fake/reinvent"], stderr="segfault"
            )
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                invoke_reinvent(client, config)
            assert exc_info.value.returncode == 1
            assert "segfault" in exc_info.value.stderr


class TestInvokeReinventSuccess:
    def test_invoke_reinvent_success_returns_molecules(self) -> None:
        client = Reinvent4Client("/fake/reinvent")
        config = {"task_type": "de_novo", "run_type": "sampling", "parameters": {}}
        fake_stdout = "SMILES,Score\nCCO,0.95\n"
        with patch("agentic_discovery.molecules.reinvent_client.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["/fake/reinvent"], returncode=0, stdout=fake_stdout, stderr=""
            )
            molecules = invoke_reinvent(client, config)
        assert len(molecules) == 1
        assert molecules[0].smiles == "CCO"
        assert molecules[0].score == pytest.approx(0.95)
        assert molecules[0].evidence_level is EvidenceLevel.GENERATED


# ---------------------------------------------------------------------------
# evidence_level tests
# ---------------------------------------------------------------------------


class TestEvidenceLevel:
    def test_evidence_level_always_generated(self) -> None:
        mol = GeneratedMolecule(smiles="CCO", score=0.5, task_type="de_novo")
        assert mol.evidence_level is EvidenceLevel.GENERATED
        # Frozen — cannot be overridden
        with pytest.raises(AttributeError):
            mol.evidence_level = EvidenceLevel.DFT_VERIFIED  # type: ignore[misc]
