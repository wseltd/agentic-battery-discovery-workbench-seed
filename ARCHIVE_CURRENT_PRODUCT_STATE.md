# Archive: Pre-Split Dual-Domain Product State

This document archives the state of the Agentic Molecular and Materials
Discovery Workbench immediately prior to the preservation-and-fork
operation that created this restructured seed repository.

## The Original Combined System

The source project is a unified dual-domain scientific discovery workbench
that generates, validates, and ranks candidates across two scientific
branches under a single shared infrastructure:

### Small-Molecule Branch
- De novo molecule generation via REINVENT 4 (AstraZeneca, GPU)
- Validation via RDKit (valence, charge, PAINS, salt stripping, stereocentre)
- Property scoring: QED, SA, molecular weight, logP, TPSA, H-bonds, rotatable bonds
- Novelty assessment against ChEMBL (2.47M reference compounds)
- Semi-empirical QC handoff via xTB

### Inorganic Crystalline Materials Branch
- Crystal structure generation via MatterGen (Microsoft, GPU)
- ML relaxation via MatterSim (Microsoft)
- Structural validation via pymatgen (distances, symmetry, dimensionality, charge balance)
- Novelty assessment against Materials Project and Alexandria
- Energy-above-hull computation against competing phases
- DFT handoff (CIF, POSCAR, VASP parameters, atomate2 workflow stubs)

### Shared Infrastructure
- Two-stage domain router (deterministic keywords + confidence scoring)
- Constraint parser with unit normalisation
- Nine-level evidence framework (generated → experimental_reported)
- Multi-objective ranker with Pareto fronts and crowding distance
- Budget controller with early stopping
- Agent loop controller (generate → validate → score → filter cycles)
- Structured reporting with scientific honesty rules
  (banned overclaiming language such as "discovered" and "proven")

## Branches and Packages in the Original Repo

The original `src/` tree was organised across several packages that
accumulated during iterative development:

- `discovery_workbench/` — main workbench (molecules + materials + shared)
- `agentic_discovery/` — core agent with domain submodules
- `agentic_discovery_workbench/` — materials pipeline (coarse/strict passes, novelty, hull)
- `agentic_workbench/` — orchestration (loop controller, dedup, shortlist)
- `amdw/` — dual workbench (molecule + materials + reporting)
- `ammd/` — parsing and schemas (materials structure parsing)
- `workbench/` — molecule-focused modules (novelty, duplicates, PAINS)
- `molecular_constraints/`, `molecular_properties/`, `molecular_validity/` — molecule utilities
- `validation/` — validation framework

Tests in the original repo: **1432 passing** across 99 test files.

## Why This State Is Being Preserved

The combined dual-domain architecture is a sound technical base but is
conceptually too broad for a clean open-source product launch. Two
distinct user communities (medicinal chemists and materials scientists)
are served by fundamentally different code paths, external models
(REINVENT 4 vs MatterGen), and reference databases (ChEMBL vs Materials
Project). Shipping them as one product conflates the audience and makes
marketing incoherent.

The original repo captures:
1. The full integrated system as proven end-to-end (both branches working
   against real models on real GPUs).
2. All tests passing.
3. The full installation tooling (setup.sh, INSTALL.md, verify_install.py).
4. Installed dependencies and deployed external models.

## What Future Forks Derive From This Snapshot

Three derivations are planned:

1. **Preservation clone** (this repo, `agentic-battery-discovery-workbench-seed`)
   Contains a full copy of the original at the moment of preservation,
   plus a restructured `packages/` layout that extracts the unified
   system into cleaner components.

2. **Open-source molecule product** (`agentic_molecule_discovery_workbench`
   within this seed) — a clean, molecule-only workbench.

3. **Open-source materials product** (`agentic_materials_discovery_workbench`
   within this seed) — a clean, materials-only workbench.

A future fourth derivation will take the original main repo and pivot it
toward a battery-focused vertical. That work is outside the scope of this
preservation-and-fork operation and is documented in
`FUTURE_BATTERY_PIVOT_NOTE.md`.

## Preservation Anchors

- Git tag: `pre_split_dual_domain_snapshot`
  (on the original repo at `wseltd/Agentic-Molecular-and-Materials-Discovery-Workbench`)
- This seed: `wseltd/agentic-battery-discovery-workbench-seed`
- Original repo: `wseltd/Agentic-Molecular-and-Materials-Discovery-Workbench`
