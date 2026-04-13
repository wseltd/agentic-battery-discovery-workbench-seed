"""Tests for AlexandriaReferenceBackend — rate-limited, cached API client."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
from pymatgen.core import Structure

from agentic_discovery_workbench.materials.novelty_alexandria_backend import (
    ALEXANDRIA_API_URL_ENV_VAR,
    DEFAULT_ALEXANDRIA_API_URL,
    AlexandriaReferenceBackend,
    _validate_url_scheme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_api_response(
    species: list[str],
    coords: list[list[float]],
    a: float = 5.64,
    b: float = 5.64,
    c: float = 5.64,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
) -> list[dict]:
    """Build a minimal Alexandria API response with one structure entry."""
    return [
        {
            "lattice": {
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            },
            "species": species,
            "coords": coords,
        }
    ]


def _nacl_api_response() -> list[dict]:
    """Alexandria-style JSON response for a NaCl structure."""
    return _make_api_response(
        species=["Na", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _mock_urlopen_response(data: list[dict] | dict) -> Mock:
    """Create a mock context manager mimicking urlopen response."""
    body = json.dumps(data).encode("utf-8")
    mock_response = Mock()
    mock_response.read.return_value = body
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    return mock_response


# ---------------------------------------------------------------------------
# URL scheme validation
# ---------------------------------------------------------------------------


class TestURLSchemeValidation:
    """B310: Only HTTPS URLs should be accepted."""

    def test_https_url_accepted(self) -> None:
        """Valid HTTPS URL should not raise — constructor succeeds."""
        backend = AlexandriaReferenceBackend(
            api_url="https://example.com/api"
        )
        assert backend.api_url == "https://example.com/api"

    def test_http_url_rejected(self) -> None:
        """Plain HTTP must be rejected — API traffic should be encrypted."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            _validate_url_scheme("http://example.com/api")
        assert "http" in str(exc_info.value)

    def test_file_scheme_rejected(self) -> None:
        """file:/ scheme must be rejected — this is the core B310 concern."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            _validate_url_scheme("file:///etc/passwd")
        assert "file" in str(exc_info.value)

    def test_ftp_scheme_rejected(self) -> None:
        """FTP scheme must be rejected."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            _validate_url_scheme("ftp://example.com/data")
        assert "ftp" in str(exc_info.value)

    def test_empty_scheme_rejected(self) -> None:
        """URL without a scheme must be rejected."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            _validate_url_scheme("no-scheme.example.com/api")
        assert "''" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor:
    """Verify constructor behaviour: env var, default URL, scheme check."""

    def test_default_url_when_no_args_no_env(self) -> None:
        """Without explicit URL or env var, use the default endpoint."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the env var if present
            import os
            os.environ.pop(ALEXANDRIA_API_URL_ENV_VAR, None)
            backend = AlexandriaReferenceBackend()
        assert backend.api_url == DEFAULT_ALEXANDRIA_API_URL

    def test_explicit_url_overrides_env(self) -> None:
        """An explicit api_url argument takes precedence over env var."""
        with patch.dict(
            "os.environ",
            {ALEXANDRIA_API_URL_ENV_VAR: "https://env.example.com/api"},
        ):
            backend = AlexandriaReferenceBackend(
                api_url="https://explicit.example.com/api"
            )
        assert backend.api_url == "https://explicit.example.com/api"

    def test_env_var_used_when_no_explicit_url(self) -> None:
        """Env var is used when api_url is not passed."""
        with patch.dict(
            "os.environ",
            {ALEXANDRIA_API_URL_ENV_VAR: "https://env.example.com/v2"},
        ):
            backend = AlexandriaReferenceBackend()
        assert backend.api_url == "https://env.example.com/v2"

    def test_trailing_slash_stripped(self) -> None:
        """Trailing slashes should be stripped for consistent URL joining."""
        backend = AlexandriaReferenceBackend(
            api_url="https://example.com/api/"
        )
        assert backend.api_url == "https://example.com/api"

    def test_http_url_rejected_at_construction(self) -> None:
        """Non-HTTPS URL must be rejected during __init__."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            AlexandriaReferenceBackend(api_url="http://insecure.com/api")
        assert "'http'" in str(exc_info.value)

    def test_file_url_rejected_at_construction(self) -> None:
        """file:/ URL must be rejected during __init__ (B310)."""
        with pytest.raises(ValueError, match="not allowed") as exc_info:
            AlexandriaReferenceBackend(api_url="file:///etc/passwd")
        assert "'file'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch_reference_structures tests
# ---------------------------------------------------------------------------


class TestFetchReferenceStructures:
    """Verify API querying, parsing, caching, and error handling."""

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_successful_fetch_returns_structures(
        self, mock_urlopen: Mock
    ) -> None:
        """Valid API response should return parsed pymatgen Structures."""
        response_data: list[dict] = _nacl_api_response()
        mock_urlopen.return_value = _mock_urlopen_response(response_data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result = backend.fetch_reference_structures("NaCl")

        assert len(result) == 1
        assert isinstance(result[0], Structure)
        assert len(result[0]) == 2  # Na + Cl

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_cache_avoids_second_api_call(
        self, mock_urlopen: Mock
    ) -> None:
        """Second call with same formula should use cache, not API."""
        response_data: list[dict] = _nacl_api_response()
        mock_urlopen.return_value = _mock_urlopen_response(response_data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result1 = backend.fetch_reference_structures("NaCl")
        result2 = backend.fetch_reference_structures("NaCl")

        assert mock_urlopen.call_count == 1
        assert result1 == result2

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_different_formulas_make_separate_calls(
        self, mock_urlopen: Mock
    ) -> None:
        """Different compositions should each trigger their own API call."""
        response_data: list[dict] = _nacl_api_response()
        mock_urlopen.return_value = _mock_urlopen_response(response_data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        backend.fetch_reference_structures("NaCl")
        backend.fetch_reference_structures("Fe")

        assert mock_urlopen.call_count == 2

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_api_failure_returns_empty_list(
        self, mock_urlopen: Mock
    ) -> None:
        """Network failure should return empty list, not crash."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result = backend.fetch_reference_structures("NaCl")

        assert result == []

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_api_failure_cached_as_empty(
        self, mock_urlopen: Mock
    ) -> None:
        """After an API failure, subsequent calls for the same formula
        should return cached empty list without hitting the API again."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        backend.fetch_reference_structures("NaCl")
        backend.fetch_reference_structures("NaCl")

        # Only one actual API call despite two fetch_reference_structures calls
        assert mock_urlopen.call_count == 1

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_malformed_json_returns_empty_list(
        self, mock_urlopen: Mock
    ) -> None:
        """Invalid JSON response should return empty, not crash."""
        mock_response = Mock()
        mock_response.read.return_value = b"not valid json"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result = backend.fetch_reference_structures("NaCl")

        assert result == []

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_malformed_entry_skipped_with_warning(
        self, mock_urlopen: Mock
    ) -> None:
        """An entry missing required fields should be skipped, not crash.

        Good entries in the same response should still be returned.
        """
        data: list[dict] = [
            {"lattice": {"a": 5.0}, "species": ["Na"]},  # incomplete
            {
                "lattice": {
                    "a": 5.64, "b": 5.64, "c": 5.64,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                },
                "species": ["Na", "Cl"],
                "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            },
        ]
        mock_urlopen.return_value = _mock_urlopen_response(data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result = backend.fetch_reference_structures("NaCl")

        # First entry fails to parse, second succeeds
        assert len(result) == 1

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_empty_api_response_returns_empty_list(
        self, mock_urlopen: Mock
    ) -> None:
        """API returns no structures for this composition."""
        empty_data: list[dict] = []
        mock_urlopen.return_value = _mock_urlopen_response(empty_data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        result = backend.fetch_reference_structures("XeF99")

        assert result == []

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_url_contains_formula_parameter(
        self, mock_urlopen: Mock
    ) -> None:
        """Request URL should include the formula as a query parameter."""
        empty_data: list[dict] = []
        mock_urlopen.return_value = _mock_urlopen_response(empty_data)
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        backend.fetch_reference_structures("Fe2O3")

        # Inspect the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert "formula=Fe2O3" in request_obj.full_url


# ---------------------------------------------------------------------------
# get_db_version tests
# ---------------------------------------------------------------------------


class TestGetDBVersion:
    """Verify database version retrieval."""

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_returns_version_string(self, mock_urlopen: Mock) -> None:
        """Successful info endpoint should return the version."""
        mock_urlopen.return_value = _mock_urlopen_response(
            {"version": "v2024.1"}
        )
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        version = backend.get_db_version()

        assert version == "v2024.1"

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_returns_unknown_on_failure(
        self, mock_urlopen: Mock
    ) -> None:
        """API failure should return 'unknown', not crash."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("timeout")
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        version = backend.get_db_version()

        assert version == "unknown"

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_returns_unknown_when_key_missing(
        self, mock_urlopen: Mock
    ) -> None:
        """Response without 'version' key should return 'unknown'."""
        mock_urlopen.return_value = _mock_urlopen_response(
            {"status": "ok"}
        )
        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api"
        )

        version = backend.get_db_version()

        assert version == "unknown"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Verify rate-limiting behaviour between API requests."""

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.time"
    )
    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_rate_limit_sleeps_when_calls_too_fast(
        self, mock_urlopen: Mock, mock_time: Mock
    ) -> None:
        """Consecutive API calls faster than rate_limit should trigger sleep.

        Uses two different formulas to bypass cache and force two real
        API calls.
        """
        empty_data: list[dict] = []
        mock_urlopen.return_value = _mock_urlopen_response(empty_data)

        # First _respect_rate_limit: monotonic returns 10.0,
        # _last_call_time is 0.0, elapsed=10.0 — no sleep.
        # After first call, _last_call_time set to 10.0.
        # Second _respect_rate_limit: monotonic returns 10.1,
        # elapsed=0.1 < 1.0 — triggers sleep(0.9).
        mock_time.monotonic.side_effect = [10.0, 10.0, 10.1, 11.0]
        mock_time.sleep = Mock()

        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api",
            rate_limit=1.0,
        )

        backend.fetch_reference_structures("NaCl")
        backend.fetch_reference_structures("Fe")

        assert mock_time.sleep.call_count == 1
        sleep_duration = mock_time.sleep.call_args[0][0]
        assert sleep_duration == pytest.approx(0.9)

    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.time"
    )
    @patch(
        "agentic_discovery_workbench.materials."
        "novelty_alexandria_backend.urlopen"
    )
    def test_no_sleep_when_enough_time_elapsed(
        self, mock_urlopen: Mock, mock_time: Mock
    ) -> None:
        """If enough time has elapsed, no sleep should occur."""
        empty_data: list[dict] = []
        mock_urlopen.return_value = _mock_urlopen_response(empty_data)

        # Both calls have elapsed > rate_limit — no sleep needed.
        mock_time.monotonic.side_effect = [10.0, 10.0, 20.0, 20.0]
        mock_time.sleep = Mock()

        backend = AlexandriaReferenceBackend(
            api_url="https://test.example.com/api",
            rate_limit=1.0,
        )

        backend.fetch_reference_structures("NaCl")
        backend.fetch_reference_structures("Fe")

        assert mock_time.sleep.call_count == 0
