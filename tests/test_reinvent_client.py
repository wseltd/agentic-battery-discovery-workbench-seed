"""Tests for REINVENT 4 client: config building, output parsing, invocation.

Heavy coverage on parse_reinvent_output (the parser is where malformed
data creates risk) and build_config validation (wrong task_type is a
boundary error). Invocation tests use mocked subprocess since we cannot
run REINVENT in CI.
"""

from __future__ import annotations

import subprocess  # nosec B404 — testing subprocess-based REINVENT client
from pathlib import Path
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
        # Constraints are now expressed as scoring components
        assert "scoring" in cfg
        component_names = [c["name"] for c in cfg["scoring"]["components"]]
        assert "max_weight" in component_names
        assert "min_logp" in component_names


# ---------------------------------------------------------------------------
# Private config builder tests
# ---------------------------------------------------------------------------


class TestBuildDeNovoConfig:
    def test_de_novo_config_structure(self, tmp_path: Path) -> None:
        """De-novo config must contain run_type, task_type, output_csv."""
        client = Reinvent4Client("/fake/reinvent")
        output_csv = tmp_path / "output.csv"
        cfg = client._build_de_novo_config({}, output_csv)
        assert cfg["run_type"] == "sampling"
        assert cfg["task_type"] == "de_novo"
        assert cfg["output_csv"] == str(output_csv)

    def test_de_novo_config_with_constraints(self, tmp_path: Path) -> None:
        """Constraints should populate the scoring section."""
        client = Reinvent4Client("/fake/reinvent")
        constraints = {"molecular_weight": {"max": 500}}
        cfg = client._build_de_novo_config(constraints, tmp_path / "out.csv")
        assert "scoring" in cfg
        assert len(cfg["scoring"]["components"]) == 1
        assert cfg["scoring"]["components"][0]["name"] == "molecular_weight"

    def test_de_novo_config_no_constraints_omits_scoring(self, tmp_path: Path) -> None:
        """Empty constraints should not produce a scoring section."""
        client = Reinvent4Client("/fake/reinvent")
        cfg = client._build_de_novo_config({}, tmp_path / "out.csv")
        assert "scoring" not in cfg


class TestScaffoldBuilder:
    def test_scaffold_config_includes_scaffold(self, tmp_path: Path) -> None:
        client = Reinvent4Client("/fake/reinvent")
        output_csv = tmp_path / "output.csv"
        cfg = client._build_scaffold_constrained_config(
            {}, scaffold="c1ccccc1", output_csv=output_csv
        )
        assert cfg["task_type"] == "scaffold_constrained"
        assert cfg["parameters"]["scaffold"] == "c1ccccc1"
        assert cfg["output_csv"] == str(output_csv)

    def test_scaffold_config_with_constraints(self, tmp_path: Path) -> None:
        client = Reinvent4Client("/fake/reinvent")
        constraints = {"logp": {"min": 1.0, "max": 5.0}}
        cfg = client._build_scaffold_constrained_config(
            constraints, scaffold="c1ccccc1", output_csv=tmp_path / "out.csv"
        )
        assert cfg["scoring"]["components"][0]["name"] == "logp"
        assert cfg["scoring"]["components"][0]["params"] == {"min": 1.0, "max": 5.0}


class TestBuildOptimiseConfig:
    def test_optimise_config_includes_starting_molecules(self, tmp_path: Path) -> None:
        client = Reinvent4Client("/fake/reinvent")
        output_csv = tmp_path / "output.csv"
        cfg = client._build_optimise_config(
            {}, smiles=["CCO", "CCN"], output_csv=output_csv
        )
        assert cfg["task_type"] == "optimise"
        assert cfg["parameters"]["starting_molecules"] == ["CCO", "CCN"]
        assert cfg["output_csv"] == str(output_csv)

    def test_optimise_config_with_constraints(self, tmp_path: Path) -> None:
        client = Reinvent4Client("/fake/reinvent")
        constraints = {"tpsa": {"max": 140}}
        cfg = client._build_optimise_config(
            constraints, smiles=["CCO"], output_csv=tmp_path / "out.csv"
        )
        assert "scoring" in cfg
        assert cfg["scoring"]["components"][0]["name"] == "tpsa"

    def test_optimise_config_no_smiles(self, tmp_path: Path) -> None:
        """Omitting smiles should not add starting_molecules key."""
        client = Reinvent4Client("/fake/reinvent")
        cfg = client._build_optimise_config({}, output_csv=tmp_path / "out.csv")
        assert "starting_molecules" not in cfg["parameters"]


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
# _parse_output (file-based) tests
# ---------------------------------------------------------------------------


class TestParseOutputFile:
    """Tests for Reinvent4Client._parse_output which reads CSV from disk."""

    def test_parse_output_file_valid_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("SMILES,Score\nCCO,0.85\nCCN,0.72\n")
        client = Reinvent4Client("/fake/reinvent")
        molecules = client._parse_output(csv_file, "de_novo")
        assert len(molecules) == 2
        assert molecules[0].smiles == "CCO"
        assert molecules[0].score == pytest.approx(0.85)
        assert molecules[1].smiles == "CCN"

    def test_parse_output_file_empty_returns_empty_list(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        client = Reinvent4Client("/fake/reinvent")
        assert client._parse_output(csv_file, "de_novo") == []

    def test_parse_output_file_header_only_returns_empty_list(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("SMILES,Score\n")
        client = Reinvent4Client("/fake/reinvent")
        assert client._parse_output(csv_file, "de_novo") == []

    def test_parse_output_file_missing_returns_empty_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing file (partial/aborted run) should return [] and warn."""
        missing = tmp_path / "no_such_file.csv"
        client = Reinvent4Client("/fake/reinvent")
        with caplog.at_level("WARNING"):
            result = client._parse_output(missing, "de_novo")
        assert result == []
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_parse_output_file_malformed_lines_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed rows in the CSV are skipped; valid rows are returned."""
        csv_file = tmp_path / "partial.csv"
        csv_file.write_text("SMILES,Score\nCCO,0.9\nbad_row,not_float\nCCN,0.8\n")
        client = Reinvent4Client("/fake/reinvent")
        with caplog.at_level("WARNING"):
            molecules = client._parse_output(csv_file, "optimise")
        assert len(molecules) == 2
        assert molecules[0].smiles == "CCO"
        assert molecules[1].smiles == "CCN"
        assert molecules[0].task_type == "optimise"
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_parse_output_file_stamps_task_type(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("SMILES,Score\nCCO,0.5\n")
        client = Reinvent4Client("/fake/reinvent")
        molecules = client._parse_output(csv_file, "scaffold_constrained")
        assert molecules[0].task_type == "scaffold_constrained"
        assert molecules[0].evidence_level is EvidenceLevel.GENERATED


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
