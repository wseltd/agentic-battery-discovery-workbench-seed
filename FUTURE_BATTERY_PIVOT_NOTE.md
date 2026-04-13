# Future Battery Pivot Note

## Scope

This note records a deferred decision — it is NOT a task description.

## What Is Planned

The **original** repo at
`Agentic-Molecular-and-Materials-Discovery-Workbench` will at some future
point be repurposed into a battery-focused vertical. That work will target
battery materials discovery specifically: cathodes, anodes, solid
electrolytes, interface chemistries, and battery-relevant property
constraints (ionic conductivity, voltage, capacity, cycling stability).

## What Is Explicitly NOT Done Here

The preservation-and-fork operation that produced this seed repository
did **not** perform any battery pivot. Specifically:

- The original repo was not modified beyond adding a preservation tag
  (`pre_split_dual_domain_snapshot`). No battery-related code was added
  or removed.
- The extracted open-source products in
  `packages/agentic_molecule_discovery_workbench/` and
  `packages/agentic_materials_discovery_workbench/` are
  general-purpose — they do not hard-code battery use cases, battery
  property targets, or battery-specific reference databases.
- This seed repo is named `agentic-battery-discovery-workbench-seed`
  only because it is intended as the eventual starting point for the
  battery-focused fork. The code inside does not assume a battery
  application.

## Reason for the Deferral

Preserving the original work and producing clean open-source products
needed to happen first. The battery pivot requires separate design work
— battery chemistry targets, battery-specific property scoring, battery
reference datasets, and battery domain vocabulary — and will be driven
by a distinct instruction set at a later date.

## Reference Anchors

- Original repo (unmodified): `Agentic-Molecular-and-Materials-Discovery-Workbench`
- Preservation tag: `pre_split_dual_domain_snapshot`
- This seed repo: `agentic-battery-discovery-workbench-seed`
- Open-source molecule product: `packages/agentic_molecule_discovery_workbench/`
- Open-source materials product: `packages/agentic_materials_discovery_workbench/`
