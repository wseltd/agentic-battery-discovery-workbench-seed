# agentic-discovery-core

Shared core library for agentic scientific discovery workbenches.

This package contains domain-agnostic infrastructure used by both the
molecule and materials discovery products. It is intentionally free of
domain-specific dependencies (no rdkit, no pymatgen) so that both
product packages can depend on it without pulling in each other's
toolchains.

## Key components

- **Evidence tracking** -- ordered `EvidenceLevel` enum (9 tiers from
  `REQUESTED` to `EXPERIMENTAL_REPORTED`) and `attach_evidence` helper
  for stamping credibility metadata onto results.

- **Budget control** -- `BudgetController` with configurable cycle and
  batch limits, plus three early-stop heuristics (plateau, invalidity
  spike, duplicate surge).

- **Ranking** -- multi-objective Pareto ranker using NSGA-II
  non-dominated sort and crowding distance, with full audit logging.

- **Routing** -- deterministic keyword-based domain router with
  confidence scoring and threshold-gated actions (auto / clarify /
  unsupported).

- **Reporting** -- report schema, renderer with provenance annotation,
  banned-word scanning for scientific honesty, and approved wording
  constants.

- **Agent loop** -- `AgentLoopController` implementing the
  generate-validate-score-filter protocol with budget enforcement and
  cross-cycle deduplication.

- **Orchestration** -- higher-level loop controller with pluggable
  early-stop evaluators, cycle-level deduplication, and shortlist
  assembly.

- **Constraints** -- range-string parser (`ParsedRange`) and structured
  constraint parser (`ParsedConstraints`) for user-facing property
  constraints.

- **Schemas** -- shared request and routing data models
  (`MoleculeRequest`, `MaterialsRequest`, `RoutingResult`).

## Installation

```bash
pip install -e packages/agentic_discovery_core
```

## Testing

```bash
pytest packages/agentic_discovery_core/tests
```
