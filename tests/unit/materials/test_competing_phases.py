"""Tests for competing-phase fetcher.

Mocks _MPRester at the module boundary — never hits the real Materials
Project API.  Focuses on caching behaviour, input validation, and error
propagation since the actual phase data comes from an external service.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.entries.computed_entries import ComputedEntry

from agentic_discovery_workbench.materials.competing_phases import (
    _chemsys_cache,
    clear_phase_cache,
    fetch_competing_phases,
)

# Patch target for the guarded MPRester import in competing_phases.py
_MPRESTER_PATH = "agentic_discovery_workbench.materials.competing_phases._MPRester"


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    """Ensure cache is empty before and after every test."""
    _chemsys_cache.clear()
    yield
    _chemsys_cache.clear()


def _fake_entries(n: int = 3) -> list[ComputedEntry]:
    """Create n minimal ComputedEntry objects for testing."""
    return [
        ComputedEntry(composition=f"Fe{i + 1}", energy=float(-4.0 * (i + 1)))
        for i in range(n)
    ]


def _mock_mprester_class(
    return_value: list[ComputedEntry] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Build a mock MPRester class whose instances work as context managers.

    Args:
        return_value: Entries returned by get_entries_in_chemsys.
        side_effect: Exception raised by get_entries_in_chemsys.

    Returns:
        MagicMock that behaves like the MPRester class.
    """
    mock_instance = MagicMock()
    if side_effect is not None:
        mock_instance.get_entries_in_chemsys.side_effect = side_effect
    else:
        mock_instance.get_entries_in_chemsys.return_value = (
            return_value if return_value is not None else []
        )
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


# --- Input validation --------------------------------------------------------


def test_empty_string_raises_value_error() -> None:
    """Empty chemical_system is rejected before any API call."""
    with pytest.raises(ValueError) as exc_info:
        fetch_competing_phases("")
    assert "non-empty" in str(exc_info.value)


def test_whitespace_only_raises_value_error() -> None:
    """Whitespace-only chemical_system is rejected before any API call."""
    with pytest.raises(ValueError) as exc_info:
        fetch_competing_phases("   ")
    assert "non-empty" in str(exc_info.value)


# --- Caching behaviour -------------------------------------------------------


def test_cache_avoids_redundant_api_calls() -> None:
    """Second call for the same system returns cached entries without API hit."""
    fake = _fake_entries(5)
    mock_cls = _mock_mprester_class(return_value=fake)
    mock_instance = mock_cls.return_value

    with patch(_MPRESTER_PATH, mock_cls):
        first = fetch_competing_phases("Li-Fe-O")
        second = fetch_competing_phases("Li-Fe-O")

    assert first is second
    assert len(first) == 5
    # MPRester was constructed only once — cache prevented the second call
    mock_instance.get_entries_in_chemsys.assert_called_once_with("Li-Fe-O")


def test_different_systems_cached_independently() -> None:
    """Entries for different systems are cached separately."""
    fe_entries = _fake_entries(2)
    li_entries = _fake_entries(4)

    mock_cls = _mock_mprester_class()
    mock_instance = mock_cls.return_value
    mock_instance.get_entries_in_chemsys.side_effect = [fe_entries, li_entries]

    with patch(_MPRESTER_PATH, mock_cls):
        fe_result = fetch_competing_phases("Fe")
        li_result = fetch_competing_phases("Li")

    assert len(fe_result) == 2
    assert len(li_result) == 4
    assert mock_instance.get_entries_in_chemsys.call_count == 2


def test_clear_phase_cache_empties_cache() -> None:
    """clear_phase_cache removes all cached entries."""
    fake = _fake_entries(3)
    mock_cls = _mock_mprester_class(return_value=fake)
    mock_instance = mock_cls.return_value

    with patch(_MPRESTER_PATH, mock_cls):
        fetch_competing_phases("Fe")
        assert len(_chemsys_cache) == 1

        clear_phase_cache()
        assert len(_chemsys_cache) == 0

        # After clearing, next fetch should hit the API again
        fetch_competing_phases("Fe")
        assert mock_instance.get_entries_in_chemsys.call_count == 2


# --- API key forwarding ------------------------------------------------------


def test_api_key_forwarded_to_mprester() -> None:
    """Explicit API key is passed through to MPRester constructor."""
    mock_cls = _mock_mprester_class()

    with patch(_MPRESTER_PATH, mock_cls):
        fetch_competing_phases("Fe", mp_api_key="test-key-123")

    assert mock_cls.call_count == 1
    assert mock_cls.call_args.kwargs["api_key"] == "test-key-123"


def test_none_api_key_passed_to_mprester() -> None:
    """When no API key is given, None is forwarded (MPRester uses env var)."""
    mock_cls = _mock_mprester_class()

    with patch(_MPRESTER_PATH, mock_cls):
        fetch_competing_phases("Fe")

    assert mock_cls.call_count == 1
    assert mock_cls.call_args.kwargs["api_key"] is None


# --- Error propagation -------------------------------------------------------


def test_api_failure_propagates() -> None:
    """ConnectionError from MPRester is not silenced — propagates to caller."""
    mock_cls = _mock_mprester_class(
        side_effect=ConnectionError("MP API unavailable"),
    )

    with patch(_MPRESTER_PATH, mock_cls):
        with pytest.raises(ConnectionError) as exc_info:
            fetch_competing_phases("Fe")
        assert "MP API unavailable" in str(exc_info.value)


# --- Return type -------------------------------------------------------------


def test_returns_list_of_computed_entries() -> None:
    """Fetched entries are ComputedEntry instances, not generic objects."""
    fake = _fake_entries(2)
    mock_cls = _mock_mprester_class(return_value=fake)

    with patch(_MPRESTER_PATH, mock_cls):
        result = fetch_competing_phases("Fe")

    assert len(result) == 2
    assert all(isinstance(e, ComputedEntry) for e in result)
