# Extracted Product Boundaries

Strict mapping of which source modules belong in which extracted package.

## agentic_discovery_core

Domain-agnostic reusable infrastructure. No rdkit, no pymatgen, no
domain-specific logic.

### Core modules (top level)
| Module | Origin |
|--------|--------|
| `evidence.py` | `src/discovery_workbench/evidence.py` (canonical) |
| `budget.py` | `src/discovery_workbench/budget.py` |
| `ranker.py` | `src/discovery_workbench/ranker.py` |
| `ranker_validation.py` | `src/discovery_workbench/ranker_validation.py` |
| `pareto.py` | `src/discovery_workbench/pareto.py` |
| `crowding.py` | `src/discovery_workbench/crowding.py` |
| `schemas.py` | `src/discovery_workbench/schemas.py` |
| `constraints.py` | `src/discovery_workbench/constraints.py` |
| `report_renderer.py` | `src/discovery_workbench/report_renderer.py` |
| `report_constants.py` | `src/discovery_workbench/report_constants.py` |
| `report_schema.py` | `src/discovery_workbench/report_schema.py` |
| `banned_words.py` | `src/discovery_workbench/banned_words.py` |

### `routing/`
| Module | Origin |
|--------|--------|
| `router.py` | `src/discovery_workbench/routing/router.py` |
| `confidence.py` | `src/discovery_workbench/routing/confidence.py` |
| `keywords.py` | `src/discovery_workbench/routing/keywords.py` |
| `scorer.py` | `src/discovery_workbench/routing/scorer.py` |
| `thresholds.py` | `src/discovery_workbench/routing/thresholds.py` |
| `tokeniser.py` | `src/discovery_workbench/routing/tokeniser.py` |

### `shared/`
| Module | Origin |
|--------|--------|
| `agent_loop.py` | `src/discovery_workbench/shared/agent_loop.py` |
| `constraints.py` | `src/discovery_workbench/shared/constraints.py` |
| `early_stop.py` | `src/discovery_workbench/shared/early_stop.py` |

### `orchestration/`
| Module | Origin |
|--------|--------|
| `loop_controller.py` | `src/agentic_workbench/orchestration/loop_controller.py` |
| `dedup.py` | `src/agentic_workbench/orchestration/dedup.py` |
| `early_stop.py` | `src/agentic_workbench/orchestration/early_stop.py` |
| `shortlist.py` | `src/agentic_workbench/orchestration/shortlist.py` |

### `validation/`
| Module | Origin |
|--------|--------|
| `models.py` | `src/validation/models.py` (domain-neutralised — `Optional[Any]` for mol field) |

### `reporting/`
| Module | Origin |
|--------|--------|
| `schema.py` | `src/amdw/reporting/schema.py` |

### Intentionally NOT in core
- `validation/stages.py`, `validation/stage_lattice.py` — pymatgen-dependent; stays in materials product
- `molecule.py`, `molecular_*` — molecule-specific; stays in molecule product
- All `src/discovery_workbench/molecules/*` — molecule-specific
- All `src/discovery_workbench/materials/*` — materials-specific
- All `src/agentic_discovery_workbench/materials/*` — materials-specific

---

## agentic_molecule_discovery

Small-molecule drug design only. Depends on `rdkit`, `numpy`, `agentic-discovery-core`.

### Module layout
| Package | Contents | Origin |
|---------|----------|--------|
| `generation/` | REINVENT 4 client | `src/agentic_discovery/molecules/reinvent_client.py` |
| `validation/` | formal charge, salt strip, stereocentre, pipeline, validity | `src/molecular_validity/*`, `src/discovery_workbench/molecules/validity.py` |
| `properties/` | MW, logP, TPSA, H-bonds, rotatable bonds, atom/ring counts | `src/molecular_properties/*` |
| `constraints/` | SMARTS parser, property parser, constraint checker | `src/molecular_constraints/*`, `src/agentic_discovery/molecules/constraint_checker.py` |
| `scoring/` | scoring aggregator, property scores | `src/agentic_discovery/molecules/scoring_aggregator.py`, `src/workbench/molecules/property_scores.py` |
| `novelty/` | ChEMBL novelty checker, SMILES dedup, PAINS filter | `src/workbench/molecules/*`, `src/discovery_workbench/molecules/novelty.py` |
| `handoff/xtb_handoff.py` | xTB handoff bundle builder | `src/agentic_discovery/molecules/xtb_handoff.py` (chose the more complete version over `src/amdw/molecules/xtb_handoff.py`) |
| `benchmarks/` | GuacaMol benchmarks, molecular benchmarks | `src/discovery_workbench/molecules/benchmarks.py`, `src/amdw/molecules/mol_benchmark.py` |
| `reporting/report.py` | molecule report annex | `src/ammd/molecules/report.py` |
| `molecule.py` | CanonicalMolecule data model | `src/discovery_workbench/molecule.py` |

### Explicitly NOT in molecule product
- Any pymatgen, MatterGen, MatterSim, ASE, VASP, or crystal-related code
- Materials reference databases (Materials Project, Alexandria)

---

## agentic_materials_discovery

Inorganic crystalline materials discovery only. Depends on `pymatgen`,
`ase`, `numpy`, `agentic-discovery-core`.

### Module layout
| Package | Contents | Origin |
|---------|----------|--------|
| `generation/mattergen_client.py` | MatterGen client | `src/discovery_workbench/materials/generation/mattergen_client.py` |
| `relaxation/mattersim_client.py` | MatterSim relaxer | `src/discovery_workbench/materials/relaxation/mattersim_client.py` |
| `validation/` | lattice/elements/atoms/distances/coordination, dimensionality, post-relaxation | `src/agentic_discovery/materials/validation.py`, `src/agentic_discovery_workbench/materials/post_relaxation_validator.py`, `coarse_pass.py`, `strict_pass.py`, `src/discovery_workbench/materials/validation/dimensionality.py` |
| `novelty/` | MP + Alexandria novelty, composition/matcher/dedup, Niggli reduction | `src/agentic_discovery_workbench/materials/*` |
| `stability/` | energy above hull, corrections, competing phases, hull classifier | `src/agentic_discovery_workbench/materials/energy_*.py`, `competing_phases.py`, `src/amdw/materials/hull.py` |
| `scoring/` | complexity, stability, symmetry, target satisfaction | `src/discovery_workbench/materials/scoring/*` |
| `structure/` | crystal, symmetry, constraints, element/numeric/stoichiometry/space-group parsers | `src/ammd/materials/*` plus `src/agentic_discovery_workbench/materials/niggli_reduce.py` |
| `handoff/dft_handoff.py` | CIF/POSCAR/VASP/atomate2 bundle | `src/discovery_workbench/materials/dft_handoff.py` (chose the more complete version over `src/amdw/materials/dft_handoff.py`) |
| `ranking/ranker.py` | materials ranker | `src/discovery_workbench/materials/ranker.py` |
| `benchmarks/` | benchmark.py, mat_benchmark.py | `src/discovery_workbench/materials/benchmark.py`, `src/amdw/materials/mat_benchmark.py` |
| `reporting/report_annex.py` | materials report annex | `src/discovery_workbench/materials/report_annex.py` |

### Explicitly NOT in materials product
- Any rdkit, REINVENT, xTB, SMILES, or drug-design-related code
- Molecular reference databases (ChEMBL)
- QED, SA score, PAINS filters, molecular property calculators

---

## Deduplication Decisions

Multiple `evidence.py` files existed across `src/agentic_discovery/shared`,
`src/workbench/shared`, `src/amdw/shared`, and `src/discovery_workbench`.
They were all thin re-exports of the canonical `discovery_workbench.evidence`.
Consolidated into one authoritative `agentic_discovery_core/evidence.py`.

Multiple `xtb_handoff.py` files existed (`agentic_discovery/molecules`,
`amdw/molecules`). Selected the more complete version from
`agentic_discovery/molecules/xtb_handoff.py` which includes RDKit ETKDG
conformer generation and XYZ/SDF export.

Multiple `dft_handoff.py` files existed (`discovery_workbench/materials`,
`amdw/materials`). Selected the more complete version from
`discovery_workbench/materials/dft_handoff.py` which produces CIF, POSCAR,
VASP JSON, and atomate2 workflow stubs.
