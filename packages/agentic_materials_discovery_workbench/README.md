# Agentic Materials Discovery Workbench

A clean, modular Python package for agentic inorganic crystalline materials discovery.

## Capabilities

- **Generation** with MatterGen -- conditional crystal structure generation with chemistry, space group, and property targets.
- **Relaxation** with MatterSim -- ML force-field geometry optimisation using LBFGS with Frechet cell filtering.
- **Validation** with pymatgen -- lattice sanity, allowed elements, atom count, interatomic distances, coordination sanity, and dimensionality analysis.
- **Novelty checking** against Materials Project and Alexandria databases using StructureMatcher.
- **Energy above hull** computation with MP2020 corrections, competing phase fetching, and three-band stability classification.
- **Scoring and ranking** -- composite scores from stability, symmetry, complexity, and target satisfaction.
- **DFT handoff** -- CIF, POSCAR, VASP parameter, and atomate2 workflow stub generation.
- **Benchmarking** -- validity, uniqueness, novelty, stability proxy, and target satisfaction metrics.

## Installation

```bash
pip install agentic-materials-discovery-workbench
```

With optional backends:

```bash
pip install "agentic-materials-discovery-workbench[mattergen,mattersim,mp-api]"
```

## Quick Usage

```python
from agentic_materials_discovery.generation.mattergen_client import MatterGenConfig, generate
from agentic_materials_discovery.relaxation.mattersim_client import relax
from agentic_materials_discovery.validation.core import validate_structure
from agentic_materials_discovery.ranking.ranker import MaterialsPropertyRanker

# Configure generation
config = MatterGenConfig(
    chemistry_scope=["Li", "Fe", "P", "O"],
    space_group_number=62,
    num_samples=10,
)

# Generate candidate structures
structures = generate(config)

# Relax with MatterSim
results = [relax(s) for s in structures]

# Validate
from agentic_materials_discovery.validation.core import validate_structure
for r in results:
    report = validate_structure(r.relaxed_structure)

# Rank candidates
ranker = MaterialsPropertyRanker()
ranked = ranker.rank_candidates(candidate_dicts)
```

## Package Structure

- `generation/` -- MatterGen client for conditional crystal generation
- `relaxation/` -- MatterSim client for ML force-field relaxation
- `validation/` -- Lattice sanity, element, distance, coordination, dimensionality checks
- `novelty/` -- Novelty checking against Materials Project and Alexandria
- `stability/` -- Energy above hull, energy corrections, competing phases, hull classification
- `scoring/` -- Complexity, stability, symmetry, target satisfaction scoring
- `structure/` -- Crystal canonical representation, symmetry utilities, constraint parsing
- `handoff/` -- DFT handoff bundle generation (CIF, POSCAR, VASP, atomate2)
- `ranking/` -- Composite property ranker with configurable weights
- `benchmarks/` -- Aggregate benchmark metrics
- `reporting/` -- Report annex builder

## License

MIT
