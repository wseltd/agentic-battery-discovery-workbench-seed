"""Orchestration loop public API.

Re-exports the core loop controller, pipeline protocol, result types,
cycle-level utilities, and early-stop evaluation for iterative discovery
workflows.
"""

from agentic_workbench.orchestration.dedup import deduplicate_across_cycles
from agentic_workbench.orchestration.early_stop import (
    CycleStats,
    StopDecision,
    evaluate_stop,
)
from discovery_workbench.shared.agent_loop import (
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
    "DiscoveryPipeline",
    "LoopResult",
    "StopDecision",
    "assemble_shortlist",
    "deduplicate_across_cycles",
    "evaluate_stop",
]
