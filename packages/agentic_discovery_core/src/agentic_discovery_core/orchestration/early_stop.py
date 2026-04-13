"""Early-stop evaluation re-exported for the orchestration package.

Delegates to agentic_discovery_core.shared.early_stop, which owns the
implementation.  This module exists so callers within the orchestration
layer import from a single namespace.
"""

from agentic_discovery_core.shared.early_stop import (
    DUPLICATE_THRESHOLD,
    INVALIDITY_THRESHOLD,
    INVALIDITY_WINDOW,
    PLATEAU_THRESHOLD,
    PLATEAU_WINDOW,
    CycleStats,
    StopDecision,
    evaluate_stop,
)

__all__ = [
    "DUPLICATE_THRESHOLD",
    "INVALIDITY_THRESHOLD",
    "INVALIDITY_WINDOW",
    "PLATEAU_THRESHOLD",
    "PLATEAU_WINDOW",
    "CycleStats",
    "StopDecision",
    "evaluate_stop",
]
