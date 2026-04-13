"""Orchestration loop public API.

Re-exports the core loop controller, pipeline protocol, result types,
and cycle-level utilities for iterative discovery workflows.
"""

from agentic_workbench.orchestration.dedup import deduplicate_across_cycles
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
    "DiscoveryPipeline",
    "LoopResult",
    "assemble_shortlist",
    "deduplicate_across_cycles",
]
