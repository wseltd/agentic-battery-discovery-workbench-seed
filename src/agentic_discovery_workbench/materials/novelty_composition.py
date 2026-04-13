"""Composition-based utilities for novelty pre-filtering.

Extracts reduced formulas from pymatgen Structures and builds API-query
filters for Materials Project and Alexandria databases.  Grouping by
composition allows batch novelty checks to share a single API call per
unique formula instead of one per structure.

These are thin wrappers around pymatgen's Composition class — the value
is in centralising the formula-to-filter mapping so novelty_checker.py
and any future callers use identical query formats.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def get_reduced_formula(structure: Structure) -> str:
    """Return the reduced formula string for a crystal structure.

    Uses pymatgen's Composition.reduced_formula which normalises element
    ordering and stoichiometry (e.g. Na4Cl4 -> NaCl).

    Args:
        structure: A pymatgen Structure (typically post-relaxation).

    Returns:
        Reduced formula string (e.g. 'NaCl', 'Fe2O3').

    Raises:
        ValueError: If structure has no sites.
    """
    if len(structure) == 0:
        raise ValueError("Cannot compute reduced formula for empty structure")
    formula = structure.composition.reduced_formula
    logger.info("Reduced formula for structure: %s", formula)
    return formula


def build_composition_filter(reduced_formula: str) -> dict:
    """Build a filter dict for MP and Alexandria API composition queries.

    Both the Materials Project REST API and the Alexandria API accept
    a 'formula' field to filter structures by reduced composition.
    This function produces the canonical filter shape used by both.

    Chose a single 'formula' key rather than separate MP/Alexandria
    formats because both APIs accept this field name. If they diverge
    in the future, this is the one place to update.

    Args:
        reduced_formula: Reduced composition formula (e.g. 'NaCl').

    Returns:
        Dict with 'formula' key mapping to the reduced formula string.

    Raises:
        ValueError: If reduced_formula is empty.
    """
    if not reduced_formula or not isinstance(reduced_formula, str):
        raise ValueError(
            "reduced_formula must be a non-empty string, "
            f"got {reduced_formula!r}"
        )
    return {"formula": reduced_formula}


def group_structures_by_composition(
    structures: list[Structure],
) -> dict[str, list[Structure]]:
    """Group structures by their reduced formula for batch querying.

    Enables one API call per unique composition instead of one per
    structure, which matters when a generation run produces many
    structures sharing the same stoichiometry.

    Args:
        structures: List of pymatgen Structures to group.

    Returns:
        Dict mapping reduced formula strings to lists of structures
        sharing that composition.  Ordering within each group matches
        the input order.

    Raises:
        ValueError: If any structure in the list has no sites.
    """
    groups: dict[str, list[Structure]] = defaultdict(list)
    for i, structure in enumerate(structures):
        if len(structure) == 0:
            raise ValueError(
                f"Structure at index {i} has no sites — cannot determine "
                "composition"
            )
        formula = structure.composition.reduced_formula
        groups[formula].append(structure)
    logger.info(
        "Grouped %d structures into %d composition groups",
        len(structures),
        len(groups),
    )
    return dict(groups)
