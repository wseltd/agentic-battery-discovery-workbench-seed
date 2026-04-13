# Split Strategy

## Why Split

The original unified workbench serves two distinct scientific communities
(medicinal chemistry and materials science) with different:
- Generative models (REINVENT 4 vs MatterGen)
- Reference databases (ChEMBL vs Materials Project + Alexandria)
- Validation tools (RDKit vs pymatgen)
- QC handoffs (xTB vs VASP/atomate2)
- User vocabularies and expectations

A single product cannot coherently address both markets. The split
separates the two verticals while preserving all shared infrastructure
in a reusable core.

## What Is Preserved

Nothing is deleted. The split is additive:

- **Original repo** stays untouched at
  `Agentic-Molecular-and-Materials-Discovery-Workbench`, tagged
  `pre_split_dual_domain_snapshot` at the moment of preservation.
- **Full clone** at `agentic-battery-discovery-workbench-seed` retains
  the complete source tree under `src/` (unchanged from the original)
  alongside the new `packages/` layout.
- **All tests from the original** are copied and re-homed in the
  appropriate packages, with imports rewritten to the new namespaces.

## What Becomes Shared Core

`packages/agentic_discovery_core/` contains reusable infrastructure that
is domain-agnostic:

- `evidence.py` — 9-level evidence framework (generated → experimental_reported)
- `budget.py` — budget controller with early stopping thresholds
- `ranker.py`, `ranker_validation.py` — multi-objective ranking
- `pareto.py`, `crowding.py` — Pareto-front helpers
- `routing/` — domain router (keyword + confidence scoring)
- `shared/agent_loop.py` — Protocol-based agent loop contract
- `orchestration/` — loop controller, dedup, early stop, shortlist
- `validation/models.py` — domain-neutral validation data model
- `reporting/` — report rendering, schema, and scientific-honesty rules
- `constraints.py` — unit-normalising range constraints
- `schemas.py` — shared request/result types

Core has **no hard dependency on rdkit or pymatgen**. Either domain
package can use the core without pulling in the other domain's tools.

## What Becomes Molecule Product

`packages/agentic_molecule_discovery_workbench/` contains everything
specific to small-molecule drug design:

- `generation/reinvent_client.py` — REINVENT 4 subprocess wrapper
- `validation/` — RDKit-based molecular validity (formal charge, salts, stereocentre)
- `properties/` — MW, logP, TPSA, H-bonds, rotatable bonds, atom/ring counts
- `constraints/` — SMARTS and property constraint parsing
- `scoring/` — property scoring and aggregation
- `novelty/` — ChEMBL novelty checker, SMILES duplicate detector, PAINS filter
- `handoff/xtb_handoff.py` — xTB geometry bundle builder
- `benchmarks/` — GuacaMol metrics and molecular benchmark utilities
- `reporting/report.py` — molecule-specific report annex
- `molecule.py` — CanonicalMolecule data model

Package name: `agentic-molecule-discovery-workbench`
Depends on: `rdkit`, `numpy`, `agentic-discovery-core`
Optional deps: `reinvent`, `xtb`

## What Becomes Materials Product

`packages/agentic_materials_discovery_workbench/` contains everything
specific to inorganic crystalline materials design:

- `generation/mattergen_client.py` — MatterGen client
- `relaxation/mattersim_client.py` — MatterSim ML relaxer
- `validation/` — lattice, elements, atom count, distances, coordination, dimensionality
- `novelty/` — Materials Project + Alexandria novelty, structure dedup, Niggli reduction
- `stability/` — energy-above-hull, energy corrections, competing phases
- `scoring/` — complexity, stability, symmetry, target satisfaction
- `structure/` — crystal parsing, symmetry, stoichiometry, space-group parsing
- `handoff/dft_handoff.py` — CIF/POSCAR/VASP/atomate2 bundle
- `ranking/ranker.py` — materials ranker
- `benchmarks/` — materials benchmark utilities
- `reporting/report_annex.py` — materials-specific report annex

Package name: `agentic-materials-discovery-workbench`
Depends on: `pymatgen`, `ase`, `numpy`, `agentic-discovery-core`
Optional deps: `mattergen`, `mattersim`, `mp-api`

## Public-Language Cleanup

The extracted products use domain-specific language only:

- Molecule product docs/README reference only drug-like molecules, SMILES,
  REINVENT, RDKit, ChEMBL, xTB. No crystal/MatterGen/DFT language.
- Materials product docs/README reference only crystal structures, MatterGen,
  MatterSim, pymatgen, Materials Project, Alexandria, VASP. No SMILES/drug
  language.

Internal shared concepts (evidence levels, budget control, Pareto ranking)
remain in the core and are referenced identically by both products.

## What Is Explicitly NOT Done Here

- **No battery pivot.** The original repo is untouched and will be the
  base for a separate battery-focused initiative, documented in
  `FUTURE_BATTERY_PIVOT_NOTE.md`.
- **No deletion of cross-domain code in the clone's `src/`.** The old
  `src/` tree is preserved verbatim alongside the new `packages/` tree
  so nothing is lost.
- **No renaming of the original product.** The original remains
  "Agentic Molecular and Materials Discovery Workbench".

## Verification

After the split, all three packages run their tests independently:

```
packages/agentic_discovery_core/          365 passed
packages/agentic_molecule_discovery_workbench/    465 passed
packages/agentic_materials_discovery_workbench/   567 passed
```

Total: 1397 tests passing across the three extracted packages.
(The original `src/` tree's tests also still pass in the clone.)
