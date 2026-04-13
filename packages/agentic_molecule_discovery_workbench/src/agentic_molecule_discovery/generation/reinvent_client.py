"""REINVENT 4 client for molecular generation.

Wraps REINVENT 4 as a subprocess, providing config construction,
invocation, and output parsing for de-novo, scaffold-constrained,
and optimisation task types.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import subprocess  # nosec B404 — subprocess is the intended interface to REINVENT 4
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_molecule_discovery.evidence import EvidenceLevel

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES = frozenset({"de_novo", "scaffold_constrained", "optimise"})


class Reinvent4Error(Exception):
    """Raised when the REINVENT 4 subprocess exits with a non-zero return code.

    Parameters
    ----------
    stderr:
        Standard-error output from the failed process.
    returncode:
        The non-zero exit code.
    """

    def __init__(self, stderr: str, returncode: int) -> None:
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(
            f"REINVENT 4 failed with exit code {returncode}: {stderr}"
        )

    def __repr__(self) -> str:
        return f"Reinvent4Error(returncode={self.returncode!r}, stderr={self.stderr!r})"

# Type alias — ParsedConstraints is a plain dict until a richer type is warranted.
ParsedConstraints = dict[str, Any]


@dataclass(frozen=True)
class GeneratedMolecule:
    """A molecule produced by REINVENT 4.

    Parameters
    ----------
    smiles:
        SMILES string of the generated molecule.
    score:
        Composite score assigned by REINVENT's scoring component.
    task_type:
        The generation task type that produced this molecule.
    evidence_level:
        Always ``EvidenceLevel.GENERATED`` — these are unvalidated
        generative-model outputs.
    """

    smiles: str
    score: float
    task_type: str
    evidence_level: EvidenceLevel = field(
        default=EvidenceLevel.GENERATED, init=False
    )


@dataclass(frozen=True)
class ReinventConfig:
    """Typed wrapper around REINVENT 4 configuration values.

    Parameters
    ----------
    run_type:
        The REINVENT run type (e.g. ``"sampling"``, ``"scoring"``).
    task_type:
        One of ``"de_novo"``, ``"scaffold_constrained"``, or ``"optimise"``.
    parameters:
        Additional configuration key-value pairs passed through to REINVENT.
    """

    run_type: str
    task_type: str
    parameters: dict[str, Any] = field(default_factory=dict)


class Reinvent4Client:
    """Client for invoking REINVENT 4 as a subprocess.

    Parameters
    ----------
    reinvent_path:
        Path to the REINVENT 4 executable or entry-point script.
    """

    def __init__(self, reinvent_path: str) -> None:
        self._reinvent_path = Path(reinvent_path)

    def _run_reinvent(self, config_path: Path) -> subprocess.CompletedProcess:
        """Invoke REINVENT 4 as a subprocess.

        Parameters
        ----------
        config_path:
            Path to the TOML configuration file to pass to REINVENT.

        Returns
        -------
        subprocess.CompletedProcess
            The completed process result on success.

        Raises
        ------
        Reinvent4Error
            If the REINVENT process exits with a non-zero return code.
        """
        result = subprocess.run(  # nosec B603 — args are controlled, not user input
            [str(self._reinvent_path), str(config_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise Reinvent4Error(
                stderr=result.stderr, returncode=result.returncode
            )
        return result

    def _parse_output(
        self,
        csv_path: Path,
        task_type: str,
    ) -> list[GeneratedMolecule]:
        """Read REINVENT CSV output from disk and return parsed molecules."""
        try:
            text = csv_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("REINVENT output file not found: %s", csv_path)
            return []

        return parse_reinvent_output(text, task_type)

    def build_config(
        self,
        task_type: str,
        parameters: dict[str, Any],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a REINVENT 4 configuration dictionary.

        Parameters
        ----------
        task_type:
            One of ``"de_novo"``, ``"scaffold_constrained"``, or ``"optimise"``.
        parameters:
            Task-specific parameters (e.g. ``scaffold``, ``starting_molecules``).
        constraints:
            Optional parsed constraints to inject into the configuration.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary ready for JSON serialisation.

        Raises
        ------
        ValueError
            If *task_type* is not one of the recognised types.
        """
        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Unknown task_type {task_type!r}; "
                f"expected one of {sorted(_VALID_TASK_TYPES)}"
            )

        parsed = ParsedConstraints(constraints) if constraints is not None else {}
        params = dict(parameters)
        output_csv = Path(params.pop("output_csv", "/dev/null"))

        if task_type == "de_novo":
            config = self._build_de_novo_config(parsed, output_csv)
        elif task_type == "scaffold_constrained":
            scaffold = params.pop("scaffold", None)
            config = self._build_scaffold_constrained_config(
                parsed, scaffold=scaffold, output_csv=output_csv
            )
        else:  # optimise
            smiles = params.pop("starting_molecules", None)
            config = self._build_optimise_config(
                parsed, smiles=smiles, output_csv=output_csv
            )

        # Merge any remaining caller-supplied parameters into the config
        config["parameters"].update(params)
        return config

    # ------------------------------------------------------------------
    # Public generation methods — compose config, run, parse
    # ------------------------------------------------------------------

    def generate_de_novo(
        self,
        constraints: ParsedConstraints,
    ) -> list[GeneratedMolecule]:
        """Generate molecules from scratch using REINVENT 4."""
        return self._run_generation("de_novo", constraints, parameters={})

    def generate_scaffold_constrained(
        self,
        constraints: ParsedConstraints,
        scaffold: str,
    ) -> list[GeneratedMolecule]:
        """Generate molecules constrained to a scaffold using REINVENT 4."""
        return self._run_generation(
            "scaffold_constrained", constraints, parameters={"scaffold": scaffold}
        )

    def generate_optimise(
        self,
        constraints: ParsedConstraints,
        smiles: list[str],
    ) -> list[GeneratedMolecule]:
        """Optimise existing molecules using REINVENT 4."""
        return self._run_generation(
            "optimise", constraints, parameters={"starting_molecules": smiles}
        )

    def _run_generation(
        self,
        task_type: str,
        constraints: ParsedConstraints,
        parameters: dict[str, Any],
    ) -> list[GeneratedMolecule]:
        """Shared orchestration: build config, write temp file, run, parse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "output.csv"
            parameters["output_csv"] = str(output_csv)
            config = self.build_config(task_type, parameters, constraints)

            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

            self._run_reinvent(config_path)
            return self._parse_output(output_csv, task_type)

    # ------------------------------------------------------------------
    # Private TOML config-dict builders — one per task type
    # ------------------------------------------------------------------

    def _build_de_novo_config(
        self,
        constraints: ParsedConstraints,
        output_csv: Path,
    ) -> dict[str, Any]:
        """Build a REINVENT 4 config dict for de-novo generation."""
        config: dict[str, Any] = {
            "run_type": "sampling",
            "task_type": "de_novo",
            "parameters": {},
            "output_csv": str(output_csv),
        }
        if constraints:
            config["scoring"] = _scoring_section_from_constraints(constraints)
        return config

    def _build_scaffold_constrained_config(
        self,
        constraints: ParsedConstraints,
        scaffold: str | None = None,
        output_csv: Path = Path("/dev/null"),
    ) -> dict[str, Any]:
        """Build a REINVENT 4 config dict for scaffold-constrained generation."""
        config: dict[str, Any] = {
            "run_type": "sampling",
            "task_type": "scaffold_constrained",
            "parameters": {},
            "output_csv": str(output_csv),
        }
        if scaffold is not None:
            config["parameters"]["scaffold"] = scaffold
        if constraints:
            config["scoring"] = _scoring_section_from_constraints(constraints)
        return config

    def _build_optimise_config(
        self,
        constraints: ParsedConstraints,
        smiles: list[str] | None = None,
        output_csv: Path = Path("/dev/null"),
    ) -> dict[str, Any]:
        """Build a REINVENT 4 config dict for molecule optimisation."""
        config: dict[str, Any] = {
            "run_type": "sampling",
            "task_type": "optimise",
            "parameters": {},
            "output_csv": str(output_csv),
        }
        if smiles is not None:
            config["parameters"]["starting_molecules"] = list(smiles)
        if constraints:
            config["scoring"] = _scoring_section_from_constraints(constraints)
        return config


def _scoring_section_from_constraints(
    constraints: ParsedConstraints,
) -> dict[str, Any]:
    """Convert parsed constraints into a REINVENT 4 scoring section."""
    components: list[dict[str, Any]] = []
    for prop_name, bound in constraints.items():
        components.append({"name": prop_name, "weight": 1.0, "params": bound})
    return {"components": components}


def build_config(
    task_type: str,
    parameters: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Module-level convenience for building REINVENT config dicts."""
    client = Reinvent4Client("/unused")
    return client.build_config(task_type, parameters, constraints)


def invoke_reinvent(
    client: Reinvent4Client,
    config: dict[str, Any],
) -> list[GeneratedMolecule]:
    """Run REINVENT 4 with the given config and return generated molecules."""
    result = subprocess.run(  # nosec B603 — args are from client config, not user input
        [str(client._reinvent_path), "--config", "-"],
        input=str(config),
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_reinvent_output(result.stdout, config.get("task_type", "de_novo"))


def parse_reinvent_output(
    raw_output: str,
    task_type: str,
) -> list[GeneratedMolecule]:
    """Parse REINVENT 4 CSV output into ``GeneratedMolecule`` instances.

    Expects a CSV with at least ``SMILES`` and ``Score`` columns.
    Malformed lines are skipped with a warning.
    """
    if not raw_output.strip():
        return []

    molecules: list[GeneratedMolecule] = []
    reader = csv.DictReader(io.StringIO(raw_output))

    for line_num, row in enumerate(reader, start=2):
        try:
            smiles = row["SMILES"]
            score = float(row["Score"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed line %d: %s", line_num, exc)
            continue
        molecules.append(GeneratedMolecule(smiles=smiles, score=score, task_type=task_type))

    return molecules
