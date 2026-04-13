"""PAINS A/B/C filter using RDKit's built-in FilterCatalog.

Wraps RDKit's FilterCatalog with PAINS_A, PAINS_B, and PAINS_C catalogs
to flag molecules containing pan-assay interference substructures.  This is
a hard filter — molecules that match are flagged, not silently dropped, so
downstream ranking can incorporate the result.

The FilterCatalog is constructed once at module level (~50ms) and reused
for all subsequent calls.  Thread safety is inherited from RDKit's
FilterCatalog, which is safe for concurrent reads after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit.Chem import Mol
from rdkit.Chem.FilterCatalog import (
    FilterCatalog,
    FilterCatalogParams,
)


def _build_pains_catalog() -> FilterCatalog:
    """Build a FilterCatalog containing PAINS A, B, and C filters.

    Called once at module load.  Not a per-call operation — FilterCatalog
    construction involves parsing hundreds of SMARTS patterns and is too
    expensive to repeat per molecule.
    """
    params = FilterCatalogParams()
    # Add all three PAINS sub-catalogs
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


# Module-level singleton — built once, reused for every call to run_pains_filter.
PAINS_CATALOG: FilterCatalog = _build_pains_catalog()


@dataclass(frozen=True, slots=True)
class PAINSResult:
    """Result of running the PAINS filter on a molecule.

    Attributes
    ----------
    passed:
        True if the molecule triggered no PAINS alerts.
    matched_filters:
        Canonical RDKit filter names for each match (e.g.
        ``'Hao_PAINS_A(370)'``).  Empty when *passed* is True.
    """

    passed: bool
    matched_filters: list[str] = field(default_factory=list)


def run_pains_filter(mol: Mol) -> PAINSResult:
    """Screen a molecule against PAINS A/B/C filters.

    Parameters
    ----------
    mol:
        RDKit Mol object to screen.

    Returns
    -------
    PAINSResult
        Pass/fail flag and list of matched PAINS filter names.

    Raises
    ------
    TypeError
        If *mol* is not an RDKit Mol instance (including None).
    """
    if not isinstance(mol, Mol):
        raise TypeError(
            f"Expected rdkit.Chem.Mol, got {type(mol).__name__}"
        )

    matches: list[str] = []
    # GetMatches returns all FilterCatalogEntry objects that match
    for entry in PAINS_CATALOG.GetMatches(mol):
        matches.append(entry.GetDescription())

    return PAINSResult(
        passed=len(matches) == 0,
        matched_filters=matches,
    )
