"""Verify that the discovery_workbench package is importable and version is set."""

import importlib


def test_version_is_semver_string():
    """Package version must be a non-empty dotted string (e.g. '0.1.0')."""
    mod = importlib.import_module("discovery_workbench")
    assert isinstance(mod.__version__, str)
    parts = mod.__version__.split(".")
    assert len(parts) == 3, f"Expected 3-part semver, got {mod.__version__!r}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version segment: {part!r}"
