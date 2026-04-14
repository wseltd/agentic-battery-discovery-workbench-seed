# Agentic Battery Discovery Workbench — Seed Repository

This repo is the **cleaner post-hoc split** of the architectural experiment at [wseltd/Agentic-Molecular-and-Materials-Discovery-Workbench](https://github.com/wseltd/Agentic-Molecular-and-Materials-Discovery-Workbench).

The original experiment combined a small-molecule branch and a crystalline materials branch under a shared agentic control loop. Building that system made one thing obvious: the orchestration concerns (run state, budgets, evidence tracking, ranking, reporting) genuinely are shared; the scientific concerns (object representation, validators, scoring, simulation handoff) genuinely are not.

This repo captures that lesson. It contains:

1. The full **preserved original** under `src/` (byte-identical to the `pre_split_dual_domain_snapshot` tag on the original repo)
2. Three **cleanly extracted packages** under `packages/` — a reusable core and two specialised workbenches

The name "agentic-battery-discovery-workbench-seed" is forward-looking — the original repo is planned to pivot toward a battery-focused vertical later (see [`FUTURE_BATTERY_PIVOT_NOTE.md`](FUTURE_BATTERY_PIVOT_NOTE.md)). The code in this repo is **not** battery-specific.

## Repo Layout

```
.
├── src/                                         # Full unmodified copy of the original repo
├── packages/
│   ├── agentic_discovery_core/                  # Shared reusable infrastructure
│   ├── agentic_molecule_discovery_workbench/    # Clean molecule-only product
│   └── agentic_materials_discovery_workbench/   # Clean materials-only product
├── docs/
│   └── research-pack.md                         # Full architectural design document
├── ARCHIVE_CURRENT_PRODUCT_STATE.md             # Pre-split system description
├── SPLIT_STRATEGY.md                            # Why and how the split was done
├── EXTRACTED_PRODUCT_BOUNDARIES.md              # Which modules live where
└── FUTURE_BATTERY_PIVOT_NOTE.md                 # Deferred battery-pivot plan
```

## The Three Packages

### `packages/agentic_discovery_core/`

Domain-agnostic shared infrastructure. No `rdkit`, no `pymatgen` — can be used by any scientific discovery domain.

- Evidence framework (9 levels from `generated` to `experimental_reported`)
- Budget controller with early stopping
- Multi-objective ranker (Pareto fronts, crowding distance)
- Two-stage domain router (keyword + confidence)
- Agent loop controller (Protocol-based contract: generate → validate → score → filter)
- Orchestration primitives (loop controller, dedup, shortlist)
- Report rendering with scientific-honesty rules (banned overclaiming language)

Dependencies: `numpy`, `pydantic`.

### `packages/agentic_molecule_discovery_workbench/`

Small-molecule drug design workbench. Depends on `agentic-discovery-core`.

- Generation: [REINVENT 4](https://github.com/MolecularAI/REINVENT4)
- Validation: [RDKit](https://www.rdkit.org/) — formal charge, salt strip, stereocentre, PAINS
- Properties: MW, logP, TPSA, H-bonds, rotatable bonds, atom/ring counts
- Constraints: SMARTS and property parsing
- Novelty: [ChEMBL](https://www.ebi.ac.uk/chembl/) InChIKey + Morgan Tanimoto
- QC handoff: [xTB](https://github.com/grimme-lab/xtb) (XYZ + SDF + run script)

Dependencies: `rdkit`, `numpy`, `agentic-discovery-core`. Optional: `reinvent`, `xtb`.

### `packages/agentic_materials_discovery_workbench/`

Inorganic crystalline materials discovery workbench. Depends on `agentic-discovery-core`.

- Generation: [MatterGen](https://github.com/microsoft/mattergen)
- Relaxation: [MatterSim](https://github.com/microsoft/mattersim) ML force field
- Validation: [pymatgen](https://pymatgen.org/) — lattice, elements, distances, coordination, dimensionality
- Novelty: [Materials Project](https://materialsproject.org/) + [Alexandria](https://alexandria.icams.rub.de/)
- Stability: energy-above-hull vs. competing phases
- Scoring: complexity, stability, symmetry, target satisfaction
- DFT handoff: CIF + POSCAR + VASP params + [atomate2](https://github.com/materialsproject/atomate2) workflow stub

Dependencies: `pymatgen`, `ase`, `numpy`, `agentic-discovery-core`. Optional: `mattergen`, `mattersim`, `mp-api`.

## Test Status

| Package | Tests passing |
|---------|---------------|
| `agentic_discovery_core` | 365 |
| `agentic_molecule_discovery_workbench` | 465 |
| `agentic_materials_discovery_workbench` | 567 |
| **Total in extracted packages** | **1397** |

The preserved `src/` tree's own test suite (1432 tests) also still passes.

## Quick Start

Each package can be installed independently:

```bash
# Install the shared core
pip install -e packages/agentic_discovery_core

# Install a specific workbench (depends on the core)
pip install -e packages/agentic_molecule_discovery_workbench
# or
pip install -e packages/agentic_materials_discovery_workbench
```

For the full unified system with deployed models (REINVENT 4, MatterGen, MatterSim, xTB, ChEMBL, Materials Project), see the original repo's [`INSTALL.md`](https://github.com/wseltd/Agentic-Molecular-and-Materials-Discovery-Workbench/blob/master/INSTALL.md). The same setup applies here — the `src/` tree is identical.

## Key Documents

- [`ARCHIVE_CURRENT_PRODUCT_STATE.md`](ARCHIVE_CURRENT_PRODUCT_STATE.md) — Snapshot of the unified system before the split
- [`SPLIT_STRATEGY.md`](SPLIT_STRATEGY.md) — Why and how the split was done
- [`EXTRACTED_PRODUCT_BOUNDARIES.md`](EXTRACTED_PRODUCT_BOUNDARIES.md) — Exact module-by-module mapping from `src/` to `packages/`
- [`FUTURE_BATTERY_PIVOT_NOTE.md`](FUTURE_BATTERY_PIVOT_NOTE.md) — Deferred battery-pivot scope

## License

MIT — Copyright (c) 2026 WayneStark Enterprises Limited
