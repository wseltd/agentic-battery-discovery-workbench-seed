"""Competing-phase fetcher for convex-hull stability analysis.

Fetches reference phase entries from Materials Project for a given
chemical system, caching results per system to avoid redundant API
calls when processing multiple candidates in the same elemental space.
"""

from __future__ import annotations

import logging

from pymatgen.entries.computed_entries import ComputedEntry

logger = logging.getLogger(__name__)

# Guarded import -- mp_api is an optional dependency.
try:
    from mp_api.client import MPRester as _MPRester
except ImportError:
    _MPRester = None  # type: ignore[assignment,misc]

# Module-level cache: chemical_system -> list of ComputedEntry.
_chemsys_cache: dict[str, list[ComputedEntry]] = {}


def fetch_competing_phases(
    chemical_system: str,
    mp_api_key: str | None = None,
) -> list[ComputedEntry]:
    """Fetch competing phase entries for a chemical system from Materials Project.

    Args:
        chemical_system: Dash-separated elemental system (e.g. 'Li-Fe-O').
        mp_api_key: Materials Project API key.  If None, MPRester falls
            back to the MP_API_KEY environment variable.

    Returns:
        List of ComputedEntry objects for all known phases in the system.

    Raises:
        ValueError: If chemical_system is empty.
        ImportError: If mp_api is not installed.
    """
    if not chemical_system or not chemical_system.strip():
        raise ValueError(
            f"chemical_system must be a non-empty string, got {chemical_system!r}"
        )

    if chemical_system in _chemsys_cache:
        logger.info(
            "Cache hit for chemical_system=%s (%d entries)",
            chemical_system,
            len(_chemsys_cache[chemical_system]),
        )
        return _chemsys_cache[chemical_system]

    logger.info("Fetching competing phases for chemical_system=%s", chemical_system)

    if _MPRester is None:
        raise ImportError(
            "mp_api is required for fetching competing phases. "
            "Install with: pip install mp-api"
        )

    with _MPRester(api_key=mp_api_key) as mpr:
        entries: list[ComputedEntry] = mpr.get_entries_in_chemsys(chemical_system)

    _chemsys_cache[chemical_system] = entries
    logger.info(
        "Cached %d entries for chemical_system=%s",
        len(entries),
        chemical_system,
    )
    return entries


def clear_phase_cache() -> None:
    """Reset the per-system entry cache."""
    _chemsys_cache.clear()
    logger.info("Phase cache cleared")
