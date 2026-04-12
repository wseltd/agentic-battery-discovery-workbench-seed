# Dual-Domain Agentic Discovery Workbench — Research Pack

> Reference document for architectural decisions, tool selection, dataset licensing,
> and scientific methodology across both the small-molecule and inorganic-materials
> discovery branches.

---

## SECTION 1. SHARED PRODUCT AND ROUTING

### Q1. Routing signals — small_molecule_design vs inorganic_materials_design vs unsupported

**Recommendation.** Two-stage router: deterministic domain gates first (high precision),
then a lightweight confidence scorer over remaining cases.

- Stage 1: ruleset that triggers on representations and constraint vocabularies.
- Stage 2: assigns confidence and chooses between auto-route, ask-one-clarifier, or reject.

**Alternatives considered.** Single LLM-based router; supervised text classifier;
embedding similarity to domain exemplars; always-ask clarification.

**Reason.** Deterministic gates are auditable and stable for production. They reduce
false routing when users paste domain artefacts (SMILES, CIF, space group numbers).

**Production-safe.** Yes (ruleset-first routing is auditable and testable).

#### Keyword cues

**Small-molecule** (route to `small_molecule_design`):
SMILES, InChI, ligand, docking, ADMET, cLogP/logP, TPSA, HBD/HBA, QED, PAINS,
scaffold, linker, R-group, lead optimisation, hit generation, SAR, QSAR,
IC50/EC50/Ki, Lipinski, fragment, SAFE, fragment remasking, PMO benchmark.

**Inorganic crystalline materials** (route to `inorganic_materials_design`):
crystal, unit cell, lattice, fractional coordinates, space group, CIF, POSCAR,
primitive cell, Niggli reduction, convex hull, energy above hull (eV/atom),
formation energy, phonon, k-points, VASP, bulk modulus (GPa), band gap (eV),
magnetic density, structure relaxation, MatterGen, MatterSim, S.U.N. structures.

**Unsupported** (route to `unsupported` unless scope explicitly expanded):
polymers (non-crystalline), proteins/biologics, mixtures/solutions as primary object,
devices, process design, synthesis planning as primary request,
MOF/COF (not "inorganic crystalline" in the narrow sense).

#### Structured constraint cues

- Small-molecule: MW/logP/TPSA/HBD/HBA ranges, SMARTS substructure constraints,
  ring count limits, PAINS avoidance.
- Materials: element lists (Li-Fe-P-O), stoichiometry (ABO3), space-group number 1-230,
  max atoms in unit cell, symmetry system, stability thresholds in eV/atom.

#### Ambiguity cases (must not auto-route)

Catalysts, battery materials vs electrolytes, semiconductors (organic vs inorganic),
molecular magnets vs crystalline magnets, perovskites (inorganic vs hybrid),
nanoparticles (not crystal unless explicitly periodic).

#### Confidence thresholds

| Confidence | Action |
|---|---|
| >= 0.80 | Auto-route |
| 0.55-0.79 | Ask one clarifying question |
| < 0.55 | Return `unsupported` with examples |

---

### Q2. Mandatory vs optional user inputs

**Production-safe.** Yes.

#### Small molecules — mandatory

| Field | Type | Notes |
|---|---|---|
| `task_type` | enum: de_novo, scaffold_constrained, optimise_existing | v1 subset |
| `objective` | text | one sentence describing "good" |
| `constraints` | object | numeric windows and/or substructure rules (may be empty) |
| `output_count` | int | requested proposals + shortlist size |
| `reference_set` | enum | e.g. ChEMBL_36 |

#### Small molecules — optional

starting_scaffold (SMARTS/SMILES), starting_molecules (SMILES list),
property_priorities (weights), allowed/forbidden elements, max_synthesis_risk flag.

#### Inorganic crystals — mandatory

| Field | Type | Notes |
|---|---|---|
| `chemistry_scope` | allowed_elements list or chemical_system string | MatterGen conditioning |
| `structure_size_limit` | int | max_atoms, cap at 20 for v1 |
| `symmetry_request` | crystal_system or space_group_number or "any" | MatterGen supports SG# |
| `stability_target` | float | default <= 0.1 eV/atom |
| `output_count` | int | generated + shortlist size |

#### Inorganic crystals — optional

band_gap_eV, bulk_modulus_GPa, magnetic_density, hhi_score,
exclude_elements (default excludes Tc/Pm/Z>84/noble gases),
allow_P1 (boolean), strict_symmetry mode,
metastability_allowance_eV_per_atom (default 0.1).

---

### Q3. Scientifically valid object representations

**Recommendation.** One internal canonical object per branch; all other formats as
import/export views.

- **Molecules:** RDKit Mol + canonical SMILES + InChIKey.
  Formats: SMILES, InChI, Morgan fingerprints, MolBlock/SDF, 3D conformer (ETKDG).
  GenMol-specific: SAFE sequences as model internal representation.

- **Crystals:** pymatgen Structure + reduced/standardised lattice + CIF/POSCAR export.
  Fields: composition, lattice vectors (3x3), fractional coordinates, atomic numbers,
  space group number/symbol, site properties for magnetism.

**Tools:** RDKit (BSD-3), pymatgen (MIT).

**Caveats.** Cross-format conversions lose information: salts/tautomers in SMILES;
disorder/partial occupancy in CIF; magnetic order not fully represented without
site properties.

---

### Q4. Shared vs branch-specific components

| Component | Shared | Branch-specific |
|---|---|---|
| **Parsing** | Request language, constraint DSL, unit/number normalisation | SMILES/SMARTS (RDKit) vs CIF/POSCAR (pymatgen/ASE) |
| **Validation** | Schema, required fields, range sanity, typing | Valence/PAINS vs symmetry/distance/charge-balance |
| **Duplicate filtering** | Caching, hashing infra, "seen" ID storage | Canonical SMILES/InChIKey vs StructureMatcher |
| **Ranking** | Multi-objective framework, Pareto front, audit logs | RDKit descriptors vs relaxer energies/symmetry scores |
| **Novelty** | "novelty relative to X under Y" template | Fingerprint similarity vs structural matching |
| **Export** | JSON run bundle, provenance, citations, warnings | SMILES/SDF/XYZ/xTB vs CIF/POSCAR/VASP/QE |
| **Reporting** | Common skeleton (inputs, methods, budgets, evidence, shortlist) | Branch-specific sections, methods, caveats |

---

### Q5. Routing benchmark tasks

Curated ~30 prompts: 10 molecule, 10 materials, 10 ambiguous/unsupported.

**Molecule examples:**
1. "Generate 200 novel drug-like molecules as SMILES with MW 300-450, cLogP 2-4, TPSA 40-90, HBD <=2, HBA <=8, avoid PAINS."
2. "Optimise this scaffold to improve QED while keeping MW < 500: c1ccc(cc1)N2CCNCC2"
3. "Design linker molecules connecting these two fragments (two SMILES + attachment points)."

**Materials examples:**
1. "Generate 500 candidate oxides in Li-Fe-P-O, <=20 atoms/unit cell, prefer orthorhombic."
2. "Design a cubic crystal (SG 225) in Al-N with band gap 5-6 eV."
3. "Generate metastable candidates <=0.1 eV/atom above hull for Mg-Si."

**Ambiguous/unsupported:**
1. "Design a battery electrolyte for 4.5 V" (could be solvent molecule or crystal).
2. "Design a polymer with Tg > 150C" (unsupported).
3. "Design a catalyst for CO2 reduction" (ambiguous: ligand vs crystal surface).

---

### Q6. Shared evidence levels

**Recommendation.** Single enum mapping to two axes: provenance and evidence type.

| Level | Definition |
|---|---|
| `requested` | User-specified targets/constraints, not measured |
| `generated` | Output of generative model before validation |
| `heuristic_estimated` | Rule-based calculators, RDKit descriptors, simple symmetry scores |
| `ml_predicted` | Property predicted by ML model; must store model name/version |
| `ml_relaxed` | Geometry relaxed by ML potential (e.g. MatterSim) |
| `semiempirical_qc` | Molecules only; e.g. xTB geometry/energy |
| `dft_verified` | DFT relaxation and/or properties |
| `experimental_reported` | Linked to citation or DOI |
| `unknown` | Provenance missing |

---

### Q7. Stop conditions and budget controls

| Parameter | Default | Notes |
|---|---|---|
| `max_cycles` | 5 | agent loop iterations |
| `max_batches` | 8 (mol) / 6 (mat) | materials relaxation is heavier |
| `shortlist_size` | 25 | expandable to 50 for exploratory |
| Compute budget (mol) | CPU-bound; 1 GPU-hr if heavy generator | |
| Compute budget (mat) | 2 GPU-hr generation + 4 GPU-hr relaxation | MatterGen sampling can take hours |

**Early stop triggers:**
- Plateau: top-10 score improves <1% for 2 consecutive cycles.
- Invalidity spike: >50% of batch fails validation twice in a row.
- Duplicate surge: >70% near-duplicates in generated batch.

---

## SECTION 2. SMALL-MOLECULE BRANCH

### Q8. Molecular generation backend

**v1 recommendation:** REINVENT 4 as primary backbone.
**v1 stretch:** GenMol as optional plugin for fragment-constrained tasks.

| Aspect | REINVENT 4 | GenMol | DrugEx |
|---|---|---|---|
| Licence (code) | Apache-2.0 | Apache-2.0 | MIT |
| Licence (weights) | Apache-2.0 | NVIDIA Open Model License | MIT |
| Task support | de novo, scaffold hop, R-group, linker, optimisation | de novo, linker, motif, scaffold, hit gen, lead opt | multi-objective RL, multiple architectures |
| Maturity | Production-grade, well-documented | Early versioning, powerful but newer | Established open-source RL |
| Local deploy | Linux fully validated | High | High |
| Scoring workflow | Multi-component RL score, configurable | MCG guidance, SAFE-based | Scoring tools, multi-objective |
| Integration ease | Configuration-driven, strong docs | SAFE representation conversion needed | API-based |

**Repos:**
- REINVENT: `MolecularAI/REINVENT4`
- GenMol: `NVIDIA-Digital-Bio/genmol`, HF: `nvidia/NV-GenMol-89M-v1`, `nvidia/NV-GenMol-89M-v2`
- DrugEx: `CDDLeiden/DrugEx`

---

### Q9. Molecular task types for v1

**Include in v1:**
- `de_novo` — no starting structure
- `property_window_optimisation` — multi-objective RDKit + alerts + novelty
- `scaffold_constrained` — single scaffold SMARTS with required substructure match

**Defer:**
- Linker design / two-fragment constrained (unless GenMol is primary)
- Docking-guided optimisation (too compute-heavy, easy to overclaim)

---

### Q10. Molecular constraints (v1 parser)

**Implement (robust for v1):**
MW, cLogP (RDKit Crippen), TPSA, HBD, HBA, rotatable_bonds, heavy_atom_count,
ring_count, aromatic_ring_count, formal_charge_range, element_whitelist/blacklist,
substructure_required (SMARTS), substructure_forbidden (SMARTS),
PAINS + medchem alert filters (heuristic).

**Defer (v2+):**
pKa windows, solubility predictions, permeability predictions,
reaction-synthesis constraints, retrosynthesis feasibility.

---

### Q11. Molecular validity pipeline

Ordered chain using RDKit (BSD-3) + InChI (MIT v1.07):

1. **Syntax validity:** `MolFromSmiles` with sanitisation. Parse failure = reject.
2. **Valence validity:** RDKit sanitisation failures = hard reject (store error).
3. **Charge sanity:** Formal charge; reject outside -1..+1 unless user overrides.
4. **Stereochemistry:** Preserve given stereochem; flag undefined stereocentres.
5. **Salt/fragment rules:** Keep largest organic fragment; strip common counterions; record.
6. **Duplicate rules:** See Q12.
7. **Novelty rules:** See Q13; require explicit reference DB.

---

### Q12. Duplicate policy (molecules)

**Exact duplicate (v1 default):**
After standardisation, canonical SMILES identical OR InChIKey identical.

**Near-duplicate (v1 default):**
Morgan/ECFP (radius 2, 2048 bits) Tanimoto >= 0.85 within the run
→ down-rank or cap per cluster (keep best-scoring representative).

**Caveat:** Define tautomer policy explicitly; tautomer handling can turn
"distinct" molecules into duplicates.

---

### Q13. Novelty policy (molecules)

**Reference DB:** ChEMBL_36 (CC BY-SA 3.0).

| Classification | Criterion |
|---|---|
| Exact known | InChIKey match in reference DB |
| Close analogue | Max Tanimoto >= 0.70 (Morgan r=2) |
| Novel-like | Max Tanimoto <= 0.40 |

**Language:** Report as "no match found under these criteria", never as "new molecule".

**Caveat:** ChEMBL not exhaustive; CC BY-SA ShareAlike obligations may matter
for substantial derived datasets.

---

### Q14. Molecular property estimators (v1)

**Cheap scoring stack (production-suitable):**
- Hard filters: valence validity, formal charge range, PAINS/medchem filters (heuristic).
- Soft objectives: QED (heuristic composite), SA score proxy (heuristic),
  novelty penalty vs ChEMBL, diversity reward.

**Docking (experimental in v1):**
- AutoDock Vina (Apache-2.0) behind feature flag only.
- Label outputs as `heuristic_estimated`; include tool/version.

**NOT in v1:** Toxicity ML predictors, QM for all candidates.

---

### Q15. QC handoff (molecules)

**v1 target:** xTB geometry optimisation inputs (LGPL-3.0-or-later).

**Handoff bundle per molecule:**
- Canonical SMILES + InChIKey + RDKit Mol
- 3D conformer as SDF and XYZ
- charge=0, multiplicity=1 placeholders — flagged "user must confirm"
- xTB input: XYZ + run script template for geometry optimisation

**Alternatives for later:** ORCA, Psi4 (LGPL/GPL), Gaussian (commercial).

---

### Q16. Molecule-side datasets

**No-training-needed references (v1):**
- ChEMBL_36 (CC BY-SA 3.0) — novelty reference
- PubChem (public domain) — optional additional reference

**Benchmarks/evaluation:**
- MOSES (MIT-licensed codebase)
- GuacaMol (open-source framework)

**Avoid:** ZINC redistribution (no major portions without permission).

**Custom scoring (v1.5+):** TDC tasks (check per-dataset licence).

**Fine-tuning (later):** Internal or licenced ChEMBL subset.

---

### Q17. Molecular benchmarks

| Metric | What to compute |
|---|---|
| Validity | % RDKit-valid molecules |
| Uniqueness | % unique canonical SMILES |
| Novelty | % not in reference DB by InChIKey + similarity histogram |
| Target satisfaction | fraction within each constraint window |
| Diversity | internal diversity via fingerprint distances (distribution) |
| Shortlist quality | % passing PAINS/alerts, median QED/SA, cluster coverage |

**Frameworks:** GuacaMol (goal-directed), MOSES (generation metrics).

---

## SECTION 3. INORGANIC MATERIALS BRANCH

### Q18. MatterGen capabilities for v1

**Core v1 (production-suitable with scope limits):**
- Unconditional structure generation
- Chemical system conditioning
- Space group number conditioning

**Property conditioning (disclosed by Microsoft):**
- Formation energy above convex hull
- Band gap (eV)
- Bulk modulus (GPa)
- Magnetic density (A^-3)
- Magnetic density + HHI

**Scope limits (must enforce):**
- <= 20 atoms/unit cell
- Exclude Tc (43), Pm (61), Z>84, noble gases
- Training filtered to <= 0.1 eV/atom above hull

**Model:** `microsoft/mattergen` (MIT), checkpoint `checkpoints/mp_20_base/checkpoints/last.ckpt`.
**Training data:** MP v2022.10.28 (CC BY 4.0), Alexandria (CC BY 4.0).

**Needs your own fine-tuning:** Any property beyond disclosed set.
**Needs labelled data:** Any new property target (dielectric, thermal conductivity, etc.).

---

### Q19. Materials scope (v1)

| Parameter | Default |
|---|---|
| Allowed elements | Z <= 83, excluding Tc (43) and Pm (61); exclude Z>84; exclude noble gases |
| Max atoms/cell | 20 |
| Crystal type | Ordered only (no partial occupancies) |
| Alloys | Only if representable as ordered structures within 20-atom cap |
| Metastable | Allow up to 0.1 eV/atom above hull |
| P1 outputs | Only in "permissive" mode; default prefers higher symmetry |

---

### Q20. Symmetry handling

**Two modes:**
- **Strict symmetry:** Condition on allowed space group range; verify post-generation.
- **Preferred symmetry:** Soft preference by sampling distribution across space groups.

**Crystal system → space group mapping:**

| System | SG range |
|---|---|
| Triclinic | 1-2 |
| Monoclinic | 3-15 |
| Orthorhombic | 16-74 |
| Tetragonal | 75-142 |
| Trigonal | 143-167 |
| Hexagonal | 168-194 |
| Cubic | 195-230 |

**P1 policy:** Allowed only when `allow_P1=true`, otherwise auto-down-ranked and flagged.
**Always:** Run post-generation symmetry verification; record found SG, tolerance, match status.

---

### Q21. Relaxer backend

**v1 recommendation:** MatterSim (MIT) — MatterGen's own pipeline uses it.

| Aspect | MatterSim | CHGNet | MACE-MP |
|---|---|---|---|
| Licence | MIT | BSD-3-like | MIT (HF checkpoints) |
| Integration | Built-in relaxer class + ASE calculator | ASE calculator | ASE calculator |
| Chemistry coverage | Broad (elements, temps, pressures) | 146k compounds, full periodic table | 89 elements, MPTrj dataset |
| Coherence with MatterGen | Direct (designed together) | Separate project | Separate project |

**Repos:** `microsoft/mattersim`, `CederGroupHub/chgnet`, `ACEsuit/mace`.

**Caveat:** Any ML force field can fail out of distribution. Label outputs as `ml_relaxed`.

---

### Q22. Materials sanity-validation pipeline

**Phase A — Pre-relaxation:**

1. Format validity: parse to pymatgen Structure; reject failures.
2. Periodicity: lattice matrix finite, volume > 0, no NaN/Inf, not near-singular.
3. Allowed elements: enforce v1 whitelist.
4. Atom count: 1-20 atoms/cell.
5. Distance checks: minimum interatomic distance vs covalent radii sum.
6. Coordination sanity (soft): CrystalNN, flag extreme coordination.
7. Oxidation-state/charge-balance (soft): SMACT plausibility; fail only in "ionic strict" mode.
8. Symmetry verification: compute SG with fixed tolerance; compare to requested.
9. Dimensionality artefact rejection: reject effectively 2D/1D in 3D cell.

**Phase B — Post-relaxation:**

10. Repeat distance, symmetry, and duplicate checks after relaxation.

**Tools:** pymatgen (MIT), spglib, SMACT.

---

### Q23. Duplicate policy (materials)

**Two-pass:**
- **Pre-relax (coarse):** composition + reduced cell + loose StructureMatcher → cut redundancy.
- **Post-relax (strict):** primitive + Niggli reduction + StructureMatcher
  (ltol~0.2, stol~0.3, angle_tol~5 deg) + identical composition.

**Near-duplicate (post-relax):** Same matcher with looser tolerances (for clustering).

---

### Q24. Novelty policy (materials)

**Definition:** "No matching structure found in reference dataset R under matcher M
and tolerances T."

**Reference sets:** MP + Alexandria (both CC BY 4.0) — disclosed as MatterGen training sources.

**Report with:** Reference dataset versions, matcher tolerances,
whether matching was pre- or post-relaxation.

**Caveat:** Absence from MP/Alexandria does not imply synthesability.

---

### Q25. Reference databases for known structures

**Primary:** Materials Project (CC BY 4.0) — API-based access, v2022.10.28 used by MatterGen.
**Complementary:** Alexandria (CC BY 4.0) — broader hypothetical coverage, MatterGen training source.
**Optional cross-check:** JARVIS-DFT 3D (`jdft_3d.json`, Figshare DOI 10.6084/m9.figshare.6815699, CC BY 4.0).

**Avoid as primary:** OQMD (ICSD partnership complicates redistribution).

---

### Q26. Energy above hull

**Two-tier v1 policy:**
- **Screening:** ML-relaxed energy + reference competing phases from MP/Alex-MP.
  Label explicitly as proxy.
- **Shortlist:** DFT-consistent energy above hull for final shortlist only.

**Reference competing phases:** MP (CC BY 4.0) and/or MatterGen-provided
`reference_MP2020correction.gz`, `reference_TRI2024correction.gz`.

**Stability threshold:** <= 0.1 eV/atom (consistent with MatterGen definition).

**Caveat:** Hull completeness never guaranteed. Label as "incomplete reference coverage"
when hull is missing competing phases.

---

### Q27. Property estimators for materials ranking

**Realistic v1 stack:**
- Stability proxy: ML-relaxed convergence quality + energy/atom consistency + approximate hull (proxy).
- Symmetry score: requested vs verified SG match (with tolerance).
- Complexity score: number of elements, atoms/cell, density extremes.
- Target satisfaction: only for categorical constraints and conditioned targets (not "measured").

**Defer to DFT:** Band gap and bulk modulus prediction (no cheap reliable surrogates in v1).

---

### Q28. DFT handoff policy

**VASP-aligned defaults (where feasible):**

| Parameter | Default |
|---|---|
| Relaxation | Start from ML-relaxed; DFT relax with PBE or r2SCAN |
| Cutoff | Fixed ENCUT consistent with PAW set; MP r2SCAN uses 680 eV |
| k-points | Automatic meshes; KSPACING 0.22-0.44 A^-1 |
| Pseudopotentials | Document exactly which PAW/POTCAR family; note MP caveat about older POTCARs |
| Band gap | Static calculation after relaxation; hybrid/GW only for tiny final shortlist |
| Magnetic init | ISPIN=2; explicit MAGMOM initial moments; pymatgen magnetism tools |
| Phonons | Phonopy (finite displacement) with DFT forces for shortlisted stable structures |

**Open-source alternative:** Quantum ESPRESSO (GPL).
**Workflow tooling:** atomate2 (relax -> static -> band structure).

---

### Q29. Extra datasets for custom property conditioning

**No new dataset needed:** If staying within disclosed MatterGen targets
(energy above hull, DFT band gap, DFT bulk modulus, DFT magnetic density, HHI)
using Alex-MP-20 style datasets.

**Labelled dataset needed:** Any additional property not already labelled.

**Do not attempt in v1:** Complex downstream performance metrics
(battery cycle life, catalytic turnover) unless you have proprietary labels.

**Caveat:** Mixing labelled data from different DFT codes/functionals breaks conditioning
unless carefully normalised.

---

### Q30. Materials benchmarks

**Use MatterGen's S.U.N. framework:**

| Metric | Definition |
|---|---|
| Validity | % passing structural sanity checks (Q22) |
| Uniqueness | % unique by post-relax structure matcher |
| Novelty | % not matching reference set by matcher |
| Stability proxy | % meeting proxy stability threshold; DFT-verified subset |
| Target satisfaction | % meeting chemistry/symmetry constraints; property after DFT only |
| Shortlist usefulness | fraction stable after ML relax + charge-balance checks |
| DFT conversion rate | fraction of top-N remaining stable after DFT relaxation |

**Stable** = <= 0.1 eV/atom above hull (MatterGen definition).

---

## SECTION 4. CROSS-DOMAIN SCIENTIFIC HONESTY

### Q31. Approved vs banned language

**Approved wording:**

| Context | Wording |
|---|---|
| Generated candidate | "Generated candidate (not experimentally validated)." |
| Estimated property | "Estimated by [tool/model, version]; heuristic/ML prediction, may be inaccurate." |
| Relaxed structure | "ML-relaxed using [MatterSim/CHGNet/etc.]; not DFT-relaxed." |
| QC handoff (mol) | "Prepared for semi-empirical QC (xTB); charge/multiplicity require confirmation." |
| DFT handoff | "Prepared DFT input set; no DFT calculations run unless explicitly marked." |
| Novelty | "No match found in [ref DB/version] under [metric + threshold]. Does not establish global novelty." |
| Shortlist | "Ranked by [scoring components]. Rankings are decision-support only." |

**Banned words:** "discovered", "proven", "guaranteed stable", "clinically relevant",
"synthesizable", "optimal", "validated" (without specifying method), "novel material" (unqualified).

---

### Q32. Failure modes from mixing domains

| Category | Risk |
|---|---|
| Conceptual | Treating RDKit descriptors as analogous to energy-above-hull; docking ~ band gaps |
| Dataset | Mixing licences (ZINC restrictions vs MP CC BY 4.0); CC BY-SA propagation |
| UX | Users paste SMILES into materials branch or stoichiometries into molecule branch |
| Credibility | Calling ML-relaxed crystals "stable" without qualifiers |

**Mitigation:** Two scientific products sharing infrastructure, never one blended model.

---

### Q33. Report schema

#### Shared top-level

- Run ID, timestamp, branch, tool versions
- User brief (verbatim), parsed schema, constraints
- Budget settings and stop conditions triggered
- Evidence level legend
- Shortlist summary table (IDs, key scores, evidence level)

#### Molecule annex

- Generator details (REINVENT config / GenMol checkpoint)
- Validity / uniqueness / novelty results and definitions
- Constraint satisfaction breakdown
- Export bundle (SMILES, InChIKey, SDF/XYZ, xTB handoff)
- Warnings (heuristics, docking if used)

#### Materials annex

- Generator details (MatterGen checkpoint)
- Scope filters (<=20 atoms, element exclusions)
- Relaxation backend (MatterSim version)
- Validity / uniqueness / novelty definitions (matcher tolerances, references)
- DFT handoff: CIF/POSCAR, atomate2 workflow, parameter defaults

---

### Q34. Ablation experiments

**Design:** Compare with only one component changed.

| Experiment | Comparison |
|---|---|
| One-shot vs agentic | Generate N once vs generate N per cycle with constraint updates |
| With vs without relaxer/QC | Score raw vs score after ML relaxation |
| With vs without validation | Accept all vs reject invalid/duplicates |
| With vs without duplicate filtering | Allow duplicates vs enforce novelty |

**Measure:** top-k target satisfaction improvement, diversity, invalid/duplicate reduction,
DFT conversion rate (materials).

---

### Q35. Licensing and reproducibility risks

| Asset | Licence | Local deploy | Commercial risk | Note |
|---|---|---|---|---|
| REINVENT 4 | Apache-2.0 | High | Low | Linux fully validated |
| GenMol (code) | Apache-2.0 | High | Low | Code is permissive |
| GenMol (weights) | NVIDIA Open Model License | Medium | Medium | Additional obligations |
| RDKit | BSD-3-Clause | High | Low | Widely used |
| InChI v1.07 | MIT | High | Low | Pin version |
| xTB | LGPL-3.0-or-later | High | Medium | Binary linking obligations |
| AutoDock Vina | Apache-2.0 | High | Low | Docking scientifically risky |
| MatterGen | MIT | High | Low | Enforce scope limits |
| MatterSim | MIT | High | Low | Not DFT-equivalent |
| Materials Project | CC BY 4.0 | Medium | Medium | Use API, don't scrape |
| Alexandria | CC BY 4.0 | High | Low-Medium | Attribution required |
| JARVIS-DFT | CC BY 4.0 | High | Low-Medium | Explicit Figshare licence |
| ChEMBL | CC BY-SA 3.0 | High | Medium | ShareAlike propagation |
| ZINC | Redistribution restricted | Medium | High | No major portion redistribution |
| OQMD | Unclear (ICSD) | Medium | High | Avoid as primary |

---

### Q36. Recommended stacks

#### v1 Minimal Credible

**Shared:** Ruleset router + strict schemas + evidence enum + provenance logging + deterministic validation.

**Molecules:** REINVENT 4 generation + RDKit validation/scoring + novelty vs ChEMBL_36.

**Materials:** MatterGen (<=20 atoms, element exclusions) + MatterSim relaxation + pymatgen validation + novelty vs MP+Alexandria.

**Outputs:** Molecule shortlist with xTB handoff; materials shortlist with CIF/POSCAR + DFT handoff templates.

#### v1 Stretch

- Add GenMol plugin for fragment-constrained generation (NVIDIA Open Model License for weights).
- Add optional AutoDock Vina docking behind feature flag.
- Add atomate2 workflow export for materials.
- Add JARVIS as extra novelty reference.

#### v2 Expansion

- Alternative relaxers (MACE-MP or CHGNet) with cross-check stability proxy disagreement.
- Phonons (Phonopy) for top candidates; store as DFT-verified only with DFT forces.
- Expand property-conditioned generation within disclosed MatterGen targets or new labelled datasets.
- Calibrated novelty clustering across reference DBs.
