# Agentic Battery Discovery Workbench — Seed Repository

This repository is a **preservation clone** of the Agentic Molecular and
Materials Discovery Workbench, restructured as a workspace from which
clean open-source products can be extracted.

## This Is Not the Battery Pivot

The repository name is forward-looking — it is intended to become the
base for a battery-focused discovery workbench at a later date. The
code inside this repo is **not** battery-specific. See
[`FUTURE_BATTERY_PIVOT_NOTE.md`](FUTURE_BATTERY_PIVOT_NOTE.md).

## What This Repo Contains

```
.
├── src/                                     # Full unmodified copy of the original repo
├── packages/
│   ├── agentic_discovery_core/              # Shared reusable infrastructure
│   ├── agentic_molecule_discovery_workbench/  # Clean molecule-only product
│   └── agentic_materials_discovery_workbench/ # Clean materials-only product
├── docs/
│   └── research-pack.md                     # Full architectural design document
├── ARCHIVE_CURRENT_PRODUCT_STATE.md         # Pre-split system description
├── SPLIT_STRATEGY.md                        # Why and how the split was done
├── EXTRACTED_PRODUCT_BOUNDARIES.md          # Which modules live where
└── FUTURE_BATTERY_PIVOT_NOTE.md             # Deferred battery-pivot plan
```

## The Three Packages

### `packages/agentic_discovery_core/`
Domain-agnostic shared infrastructure: evidence levels, budget control,
multi-objective ranking, routing, agent loop, reporting, validation
models. No rdkit, no pymatgen — can be used by any scientific discovery
domain.

### `packages/agentic_molecule_discovery_workbench/`
Small-molecule drug design workbench. Uses REINVENT 4 for generation,
RDKit for validation, ChEMBL for novelty, xTB for QC handoff.
Depends on `agentic-discovery-core`.

### `packages/agentic_materials_discovery_workbench/`
Inorganic crystalline materials discovery workbench. Uses MatterGen for
generation, MatterSim for ML relaxation, pymatgen for validation,
Materials Project and Alexandria for novelty, VASP/atomate2 for DFT
handoff. Depends on `agentic-discovery-core`.

## Test Status

| Package | Tests |
|---------|-------|
| `agentic_discovery_core` | 365 passing |
| `agentic_molecule_discovery_workbench` | 465 passing |
| `agentic_materials_discovery_workbench` | 567 passing |
| **Total in extracted products** | **1397 passing** |

The original `src/` tree is preserved unmodified and its test suite
(1432 tests) also still passes.

## Key Documents

- [`ARCHIVE_CURRENT_PRODUCT_STATE.md`](ARCHIVE_CURRENT_PRODUCT_STATE.md) —
  Snapshot of the unified system before the split.
- [`SPLIT_STRATEGY.md`](SPLIT_STRATEGY.md) — Rationale and
  preservation/extraction approach.
- [`EXTRACTED_PRODUCT_BOUNDARIES.md`](EXTRACTED_PRODUCT_BOUNDARIES.md) —
  Exact module-by-module mapping from `src/` to `packages/`.
- [`FUTURE_BATTERY_PIVOT_NOTE.md`](FUTURE_BATTERY_PIVOT_NOTE.md) —
  Deferred battery-pivot scope.

## Relationship to the Original Repo

The original repo at
`Agentic-Molecular-and-Materials-Discovery-Workbench` is untouched.
It remains the master working copy and is tagged
`pre_split_dual_domain_snapshot` at the preservation point.

This seed repo was created via a full file-system copy followed by the
addition of the `packages/` layout. The `src/` tree in this seed is
byte-identical to the original at the preservation tag.

## License

MIT — Copyright (c) 2026 WayneStark Enterprises Limited
