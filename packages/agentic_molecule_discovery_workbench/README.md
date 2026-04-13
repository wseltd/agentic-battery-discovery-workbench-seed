# Agentic Molecule Discovery Workbench

A clean, modular small-molecule discovery workbench for agentic drug-design pipelines.

## Capabilities

- **Generation** -- invoke REINVENT 4 for de-novo, scaffold-constrained, and optimisation-mode molecular generation.
- **Validation** -- RDKit-backed pipeline: SMILES parsing, valence checks, formal-charge filtering, stereocentre flagging, and salt stripping.
- **Properties** -- molecular weight, cLogP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds, atom and ring counts.
- **Constraints** -- parse and evaluate numeric property windows and SMARTS substructure constraints against candidates.
- **Scoring** -- composite scoring with QED, SA score, property fitness, PAINS penalty, novelty, and diversity weighting.
- **Novelty** -- ChEMBL-backed novelty classification (exact, close-analogue, novel) and SMILES deduplication with fingerprint near-duplicate detection.
- **PAINS filtering** -- RDKit FilterCatalog-based PAINS A/B/C screening.
- **xTB handoff** -- generate 3-D conformers, XYZ files, SDF blocks, and run scripts for semiempirical quantum-chemistry calculations.
- **Benchmarks** -- GuacaMol/MOSES-style metrics: validity, uniqueness, novelty, diversity, target satisfaction, and shortlist quality.
- **Reporting** -- structured report annexes with validity, uniqueness, novelty, constraint satisfaction, and export-path metadata.

## Installation

```bash
pip install agentic-molecule-discovery-workbench
```

With optional REINVENT 4 support:

```bash
pip install "agentic-molecule-discovery-workbench[reinvent]"
```

With optional xTB support:

```bash
pip install "agentic-molecule-discovery-workbench[xtb]"
```

## Quick usage

```python
from agentic_molecule_discovery.molecule import CanonicalMolecule
from agentic_molecule_discovery.validation.validity import validate_molecule
from agentic_molecule_discovery.properties.clogp import calc_clogp
from agentic_molecule_discovery.scoring.scoring_aggregator import (
    MolecularScoringAggregator,
    ScoringWeights,
)

# Validate a SMILES string
result = validate_molecule("CCO")
assert result.is_valid

# Build a canonical molecule
mol = CanonicalMolecule.from_smiles("CCO", evidence_level="generated")
print(mol.canonical_smiles, mol.inchikey)

# Compute a property
from rdkit import Chem
rdmol = Chem.MolFromSmiles("CCO")
print(calc_clogp(rdmol))

# Score a batch
aggregator = MolecularScoringAggregator(ScoringWeights())
scored = aggregator.score_molecules([
    {"smiles": "CCO", "pains_pass": True, "property_score": 0.8},
])
print(scored[0].composite_score)
```
