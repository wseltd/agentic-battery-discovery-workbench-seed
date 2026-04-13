"""Alexandria Materials Database reference backend for novelty checking.

Provides ``AlexandriaReferenceBackend`` which queries the Alexandria REST
API (https://alexandria.icams.rub.de) for crystal structures filtered by
reduced composition, then converts the JSON response into pymatgen
Structure objects for downstream StructureMatcher comparison.
"""

from __future__ import annotations

import json
import logging
import os
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pymatgen.core import Lattice, Structure

logger = logging.getLogger(__name__)

# Default Alexandria API endpoint.
DEFAULT_ALEXANDRIA_API_URL: str = (
    "https://alexandria.icams.rub.de/pbe/v1"
)

# Env var name for overriding the API URL at deployment time.
ALEXANDRIA_API_URL_ENV_VAR: str = "ALEXANDRIA_API_URL"

# Only HTTPS is permitted for API requests.
_ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"https"})


def _validate_url_scheme(url: str) -> None:
    """Reject URLs with schemes other than HTTPS.

    Args:
        url: Full URL string to validate.

    Raises:
        ValueError: If the URL scheme is not in _ALLOWED_URL_SCHEMES.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"URL scheme {parsed.scheme!r} is not allowed; "
            f"only {sorted(_ALLOWED_URL_SCHEMES)} are permitted: {url}"
        )


class AlexandriaReferenceBackend:
    """Rate-limited, cached client for the Alexandria Materials Database.

    Queries the Alexandria REST API for structures matching a given
    reduced composition formula.  Results are cached per formula within
    a run so repeated checks for the same composition avoid redundant
    network calls.

    Args:
        api_url: Base URL for the Alexandria REST API.  Read from the
            ALEXANDRIA_API_URL environment variable if set, otherwise
            falls back to the default published endpoint.
        rate_limit: Minimum seconds between consecutive API requests.

    Raises:
        ValueError: If api_url uses a scheme other than HTTPS.
    """

    def __init__(
        self,
        api_url: str | None = None,
        rate_limit: float = 1.0,
    ) -> None:
        env_url = os.getenv(ALEXANDRIA_API_URL_ENV_VAR)
        resolved_url: str
        if api_url is not None:
            resolved_url = api_url
        elif env_url is not None:
            resolved_url = env_url
        else:
            resolved_url = DEFAULT_ALEXANDRIA_API_URL
        resolved_url = resolved_url.rstrip("/")
        _validate_url_scheme(resolved_url)
        self._api_url = resolved_url
        self._rate_limit = rate_limit
        self._last_call_time: float = 0.0
        self._cache: dict[str, list[Structure]] = {}
        logger.info(
            "AlexandriaReferenceBackend initialised: "
            "api_url=%s rate_limit=%s",
            self._api_url,
            self._rate_limit,
        )

    @property
    def api_url(self) -> str:
        """Base URL for API requests."""
        return self._api_url

    def fetch_reference_structures(
        self, reduced_formula: str
    ) -> list[Structure]:
        """Query Alexandria for structures matching a reduced composition.

        Returns cached results if the same formula was already queried in
        this run.  Otherwise makes an HTTPS request to the Alexandria REST
        endpoint with a composition filter, parses the JSON response into
        pymatgen Structure objects, caches the result, and returns the
        candidates.

        Args:
            reduced_formula: Reduced composition formula (e.g. 'NaCl').

        Returns:
            List of pymatgen Structure objects from Alexandria matching
            the composition.  Empty list if no matches or on API failure.
        """
        logger.info(
            "Fetching Alexandria structures for formula=%s",
            reduced_formula,
        )
        if reduced_formula in self._cache:
            logger.info(
                "Cache hit for formula=%s (%d structures)",
                reduced_formula,
                len(self._cache[reduced_formula]),
            )
            return self._cache[reduced_formula]

        self._respect_rate_limit()

        url = f"{self._api_url}/search?formula={reduced_formula}"
        _validate_url_scheme(url)

        try:
            request = Request(url)  # noqa: S310
            request.add_header("Accept", "application/json")
            with urlopen(request, timeout=30) as response:  # nosec B310
                raw = response.read().decode("utf-8")
            data: list[dict] = json.loads(raw)
        except (URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Alexandria API request failed for formula=%s: %s",
                reduced_formula,
                exc,
            )
            self._cache[reduced_formula] = []
            return []

        structures = self._parse_structures(data, reduced_formula)
        self._cache[reduced_formula] = structures
        logger.info(
            "Fetched %d structures from Alexandria for formula=%s",
            len(structures),
            reduced_formula,
        )
        return structures

    def get_db_version(self) -> str:
        """Return the Alexandria database version string.

        Returns:
            Version string (e.g. 'v2024.1') or 'unknown' on failure.
        """
        url = f"{self._api_url}/info"
        _validate_url_scheme(url)
        try:
            request = Request(url)  # noqa: S310
            request.add_header("Accept", "application/json")
            with urlopen(request, timeout=10) as response:  # nosec B310
                raw = response.read().decode("utf-8")
            info: dict = json.loads(raw)
            version = info.get("version", "unknown")
            logger.info("Alexandria DB version: %s", version)
            return str(version)
        except (URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to fetch Alexandria DB version: %s", exc
            )
            return "unknown"

    def _respect_rate_limit(self) -> None:
        """Block until the rate-limit window has elapsed."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call_time = time.monotonic()

    @staticmethod
    def _parse_structures(
        data: list[dict], formula: str
    ) -> list[Structure]:
        """Parse Alexandria API response into pymatgen Structure objects."""
        structures: list[Structure] = []
        for i, entry in enumerate(data):
            try:
                lattice_data = entry["lattice"]
                lattice = Lattice.from_parameters(
                    a=lattice_data["a"],
                    b=lattice_data["b"],
                    c=lattice_data["c"],
                    alpha=lattice_data["alpha"],
                    beta=lattice_data["beta"],
                    gamma=lattice_data["gamma"],
                )
                species = entry["species"]
                coords = entry["coords"]
                structures.append(
                    Structure(lattice, species, coords)
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Failed to parse Alexandria entry %d for "
                    "formula=%s: %s",
                    i,
                    formula,
                    exc,
                )
        return structures
