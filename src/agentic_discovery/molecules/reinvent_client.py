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

        config: dict[str, Any] = {
            "run_type": "sampling",
            "task_type": task_type,
            "parameters": dict(parameters),
        }

        if task_type == "scaffold_constrained":
            config["parameters"].setdefault("scaffold", parameters.get("scaffold"))

        if task_type == "optimise":
            config["parameters"].setdefault(
                "starting_molecules", parameters.get("starting_molecules")
            )

        if constraints is not None:
            config["constraints"] = dict(constraints)

        return config


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
