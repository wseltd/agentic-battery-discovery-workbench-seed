"""Orchestration loop public API.

Re-exports the core loop controller, pipeline protocol, result types,
cycle-level utilities, and early-stop evaluation for iterative discovery
workflows.
"""

from agentic_discovery_core.orchestration.dedup import deduplicate_across_cycles
from agentic_discovery_core.orchestration.early_stop import (
    DUPLICATE_THRESHOLD,
    INVALIDITY_THRESHOLD,
    INVALIDITY_WINDOW,
    PLATEAU_THRESHOLD,
    PLATEAU_WINDOW,
    CycleStats,
    StopDecision,
    evaluate_stop,
)
from agentic_discovery_core.shared.agent_loop import (
    AgentLoopController,
    CycleResult,
    DiscoveryPipeline,
    LoopResult,
    assemble_shortlist,
)

__all__ = [
    "AgentLoopController",
    "CycleResult",
    "CycleStats",
    "DUPLICATE_THRESHOLD",
    "DiscoveryPipeline",
    "INVALIDITY_THRESHOLD",
    "INVALIDITY_WINDOW",
    "LoopResult",
    "PLATEAU_THRESHOLD",
    "PLATEAU_WINDOW",
    "StopDecision",
    "assemble_shortlist",
    "deduplicate_across_cycles",
    "evaluate_stop",
]
