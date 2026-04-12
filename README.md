# Agentic Molecular and Materials Discovery Workbench

A dual-domain agentic discovery workbench for AI-guided generation, validation, and ranking of small-molecule drug candidates and inorganic crystalline materials.

## Overview

This workbench provides two scientific branches under a shared infrastructure:

- **Small-molecule design** — de novo generation, scaffold-constrained generation, and multi-objective property optimisation using REINVENT 4, scored and validated with RDKit.
- **Inorganic materials design** — crystal structure generation conditioned on chemistry, symmetry, and target properties using MatterGen, relaxed with MatterSim, and validated with pymatgen.

Both branches share a request router, constraint parser, evidence-level tracking, duplicate/novelty policies, and a structured reporting pipeline.

## Trade-Offs

- **ML relaxation over DFT for screening**: MatterSim is fast but less accurate than DFT. Structures are labelled `ml_relaxed` and DFT handoff is provided for final validation.
- **REINVENT 4 as primary generator**: Chosen for maturity and documentation over GenMol's broader task coverage. GenMol is available as a v1-stretch plugin.
- **Heuristic scoring over ML property predictors**: v1 uses RDKit descriptors and rule-based filters rather than ML toxicity/ADMET models to keep predictions auditable and avoid overconfident estimates.
- **20-atom unit cell cap**: Limits materials diversity but keeps generation and relaxation tractable within compute budgets.

## Limitations

- Generated candidates are not experimentally validated. All outputs are computational proposals.
- Novelty is defined relative to specific reference databases (ChEMBL, Materials Project, Alexandria) and does not establish global novelty.
- Property estimates are heuristic or ML-predicted and may be inaccurate. Evidence levels are tracked per value.
- No support for polymers, biologics, MOFs/COFs, or process/synthesis planning in v1.
- Convex hull stability is approximate — reference competing phases may be incomplete.

## Non-Goals

- Replacing experimental validation or DFT workflows. This workbench generates and ranks candidates; it does not claim to validate them.
- Providing synthesis routes or retrosynthesis planning.
- Supporting non-crystalline materials (polymers, glasses, amorphous solids).
- Training or fine-tuning generative models. v1 uses pre-trained checkpoints only.
- Real-time or interactive molecular dynamics.

## Status

Under active development. See `docs/research-pack.md` for the full architectural design.

## License

MIT — Copyright (c) 2026 WayneStark Enterprises Limited
