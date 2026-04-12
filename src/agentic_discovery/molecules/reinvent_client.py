"""REINVENT 4 client for molecular generation.

Wraps REINVENT 4 as a subprocess, providing config construction,
invocation, and output parsing for de-novo, scaffold-constrained,
and optimisation task types.
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess  # nosec B404 — subprocess is the intended interface to REINVENT 4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_discovery.shared.evidence import EvidenceLevel

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
        # Never use shell=True — args list prevents shell injection.
        # Uses the full path stored at construction time to satisfy B603.
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
        """Read REINVENT CSV output from disk and return parsed molecules.

        Parameters
        ----------
        csv_path:
            Path to the CSV file written by REINVENT 4.
        task_type:
            The generation task type to stamp on each molecule.

        Returns
        -------
        list[GeneratedMolecule]
            Successfully parsed molecules. Empty list if the file is
            empty or contains only a header.
        """
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
        # Default output_csv from parameters, or None — callers that need
        # a real path should use the private _build_*_config methods directly.
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
    # Private TOML config-dict builders — one per task type
    # ------------------------------------------------------------------

    def _build_de_novo_config(
        self,
        constraints: ParsedConstraints,
        output_csv: Path,
    ) -> dict[str, Any]:
        """Build a REINVENT 4 config dict for de-novo generation.

        Parameters
        ----------
        constraints:
            Parsed property constraints to inject as scoring components.
        output_csv:
            Path where REINVENT should write its CSV output.

        Returns
        -------
        dict[str, Any]
            Nested dict matching REINVENT 4 TOML structure.
        """
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
        """Build a REINVENT 4 config dict for scaffold-constrained generation.

        Parameters
        ----------
        constraints:
            Parsed property constraints.
        scaffold:
            SMILES of the scaffold to constrain generation around.
        output_csv:
            Path where REINVENT should write its CSV output.

        Returns
        -------
        dict[str, Any]
            Nested dict matching REINVENT 4 TOML structure.
        """
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
        """Build a REINVENT 4 config dict for molecule optimisation.

        Parameters
        ----------
        constraints:
            Parsed property constraints.
        smiles:
            Starting SMILES for optimisation.
        output_csv:
            Path where REINVENT should write its CSV output.

        Returns
        -------
        dict[str, Any]
            Nested dict matching REINVENT 4 TOML structure.
        """
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
    """Convert parsed constraints into a REINVENT 4 scoring section.

    Each constraint becomes a scoring component keyed by property name.
    Chose a flat component list over nested transform groups — REINVENT 4
    accepts both, and the flat form is simpler to construct and test.
    """
    components: list[dict[str, Any]] = []
    for prop_name, bound in constraints.items():
        components.append({"name": prop_name, "weight": 1.0, "params": bound})
    return {"components": components}


def build_config(
    task_type: str,
    parameters: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Module-level convenience for building REINVENT config dicts.

    Delegates to a throwaway ``Reinvent4Client``. Prefer the class
    method when the client path matters.
    """
    client = Reinvent4Client("/unused")
    return client.build_config(task_type, parameters, constraints)


def invoke_reinvent(
    client: Reinvent4Client,
    config: dict[str, Any],
) -> list[GeneratedMolecule]:
    """Run REINVENT 4 with the given config and return generated molecules.

    Parameters
    ----------
    client:
        A configured ``Reinvent4Client``.
    config:
        Configuration dict (as returned by ``build_config``).

    Returns
    -------
    list[GeneratedMolecule]
        Parsed molecules from REINVENT's output.

    Raises
    ------
    subprocess.CalledProcessError
        If the REINVENT process exits with a non-zero return code.
    """
    # Serialize config to a temp file would happen here in production;
    # for now we pass it via stdin as a simplified interface.
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

    Parameters
    ----------
    raw_output:
        Raw CSV text from REINVENT's stdout.
    task_type:
        The task type to stamp on each molecule.

    Returns
    -------
    list[GeneratedMolecule]
        Successfully parsed molecules.
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
