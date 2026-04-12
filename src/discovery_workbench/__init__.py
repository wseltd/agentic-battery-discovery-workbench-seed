"""Agentic Molecular and Materials Discovery Workbench.

Dual-domain agentic discovery workbench providing a shared request router
and constraint parser for small-molecule (REINVENT 4 / RDKit / ChEMBL) and
inorganic materials (MatterGen / MatterSim / pymatgen) design workflows.
"""

__version__ = "0.1.0"

# Lazy re-export: rdkit is a compiled C++ extension that may not be
# available in every environment (e.g. minimal pip-only venvs without
# platform wheels).  Deferring the import keeps `import discovery_workbench`
# safe regardless, while `from discovery_workbench import CanonicalMolecule`
# still works when rdkit is installed.
__all__ = ["CanonicalMolecule"]


def __getattr__(name: str):
    if name == "CanonicalMolecule":
        from discovery_workbench.molecule import CanonicalMolecule
        return CanonicalMolecule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
