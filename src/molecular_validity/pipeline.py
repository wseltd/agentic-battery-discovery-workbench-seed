"""Molecular validity pipeline — ordered stage execution.

Runs a sequence of validation and pre-processing stages against an RDKit
molecule.  Each stage is a named callable; results are collected in
execution order so callers can inspect per-stage outcomes.

Stages 1-2 are reserved for future parsing/valence checks.
Stages 3-5 are wired here:
  3. formal_charge — rejects atoms with charges outside an allowed range
  4. stereocentre   — flags chiral centres (informational, never rejects)
  5. salt_strip     — strips counterions, keeping the largest organic fragment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from rdkit import Chem

from molecular_validity.formal_charge import check_formal_charge
from molecular_validity.salt_strip import strip_salts
from molecular_validity.stereocentre import flag_stereocentres

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Outcome of a single pipeline stage.

    Attributes
    ----------
    stage:
        Integer stage number (execution order).
    name:
        Human-readable stage name.
    result:
        The value returned by the stage callable.
    """

    stage: int
    name: str
    result: Any


@dataclass(frozen=True)
class PipelineResult:
    """Aggregate outcome of running the full pipeline.

    Attributes
    ----------
    mol:
        The molecule that was validated (the original input; salt_strip
        result is available inside the stage output, not threaded back).
    stage_results:
        Per-stage outcomes in execution order.
    """

    mol: Chem.Mol
    stage_results: list[StageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
# Each entry: (stage_number, name, callable(mol) -> result)
# Callables must accept a single positional Chem.Mol argument.
# Chose a flat list over a registry/base-class hierarchy — only three stages
# exist and YAGNI applies.  If stage count grows past ~8 a registry might
# earn its keep, but not today.

_STAGES: list[tuple[int, str, Callable[[Chem.Mol], Any]]] = [
    (3, "formal_charge", check_formal_charge),
    (4, "stereocentre", flag_stereocentres),
    (5, "salt_strip", strip_salts),
]


def run_pipeline(mol: Chem.Mol) -> PipelineResult:
    """Execute all registered validation stages against *mol* in order.

    Every stage runs regardless of prior outcomes — the pipeline is
    collect-all, not fail-fast.  This is deliberate: callers often need the
    full diagnostic picture (e.g. a molecule may have both a bad charge *and*
    salt fragments).

    Parameters
    ----------
    mol:
        RDKit molecule to validate.

    Returns
    -------
    PipelineResult
        The input molecule plus a list of per-stage results.
    """
    stage_results: list[StageResult] = []

    for stage_num, name, fn in _STAGES:
        result = fn(mol)
        stage_results.append(StageResult(stage=stage_num, name=name, result=result))
        logger.debug("stage %d (%s) complete", stage_num, name)

    return PipelineResult(mol=mol, stage_results=stage_results)


def get_registered_stages() -> list[tuple[int, str]]:
    """Return ``(stage_number, name)`` pairs for all registered stages.

    Useful for introspection and testing without running the pipeline.
    """
    return [(num, name) for num, name, _ in _STAGES]
