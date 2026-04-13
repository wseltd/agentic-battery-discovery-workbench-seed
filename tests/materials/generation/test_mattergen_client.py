"""Tests for MatterGen client: config validation, conditioning, and generation."""

from __future__ import annotations

import logging
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from discovery_workbench.materials.generation.mattergen_client import (
    MatterGenClient,
    MatterGenConfig,
)


# ---------------------------------------------------------------------------
# Config defaults and validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = MatterGenConfig(chemistry_scope=["Si"])
    assert cfg.space_group_number is None
    assert cfg.num_samples == 1
    assert cfg.property_targets is None
    assert cfg.max_atoms == 20


def test_config_max_atoms_capped_at_20() -> None:
    cfg = MatterGenConfig(chemistry_scope=["Si"], max_atoms=50)
    assert cfg.max_atoms == 20


def test_config_chemistry_scope_stored() -> None:
    elements = ["Li", "Fe", "P", "O"]
    cfg = MatterGenConfig(chemistry_scope=elements)
    assert cfg.chemistry_scope == elements
    assert cfg.to_chemical_system() == "Fe-Li-O-P"


def test_config_optional_property_targets() -> None:
    targets = {"band_gap_eV": 1.5, "bulk_modulus_GPa": 100.0}
    cfg = MatterGenConfig(chemistry_scope=["Si"], property_targets=targets)
    assert cfg.property_targets == targets


def test_config_empty_scope_raises() -> None:
    with pytest.raises(ValueError, match="chemistry_scope"):
        MatterGenConfig(chemistry_scope=[])


def test_config_num_samples_zero_raises() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        MatterGenConfig(chemistry_scope=["Si"], num_samples=0)


def test_config_max_atoms_at_boundary() -> None:
    cfg = MatterGenConfig(chemistry_scope=["C"], max_atoms=20)
    assert cfg.max_atoms == 20


def test_to_chemical_system_single_element() -> None:
    cfg = MatterGenConfig(chemistry_scope=["Fe"])
    assert cfg.to_chemical_system() == "Fe"


# ---------------------------------------------------------------------------
# from_dict factory
# ---------------------------------------------------------------------------


def test_from_dict_populates_all_fields() -> None:
    data = {
        "chemistry_scope": ["Li", "Fe", "P", "O"],
        "space_group_number": 62,
        "num_samples": 5,
        "property_targets": {"band_gap_eV": 1.5},
        "max_atoms": 15,
    }
    cfg = MatterGenConfig.from_dict(data)
    assert cfg.chemistry_scope == ["Li", "Fe", "P", "O"]
    assert cfg.space_group_number == 62
    assert cfg.num_samples == 5
    assert cfg.property_targets == {"band_gap_eV": 1.5}
    assert cfg.max_atoms == 15


def test_from_dict_uses_defaults_for_missing_keys() -> None:
    cfg = MatterGenConfig.from_dict({"chemistry_scope": ["Si"]})
    assert cfg.space_group_number is None
    assert cfg.num_samples == 1
    assert cfg.property_targets is None
    assert cfg.max_atoms == 20


def test_from_dict_missing_chemistry_scope_raises() -> None:
    with pytest.raises(KeyError):
        MatterGenConfig.from_dict({"space_group_number": 62})


# ---------------------------------------------------------------------------
# Conditioning dict
# ---------------------------------------------------------------------------


def test_conditioning_dict_includes_chemistry() -> None:
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Li", "Fe"])
    cond = client.build_conditioning_dict(cfg)
    assert cond["chemical_system"] == "Fe-Li"


def test_conditioning_dict_includes_sg() -> None:
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Na", "Cl"], space_group_number=225)
    cond = client.build_conditioning_dict(cfg)
    assert cond["space_group"] == 225


def test_conditioning_dict_omits_none_sg() -> None:
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Si"])
    cond = client.build_conditioning_dict(cfg)
    assert "space_group" not in cond


def test_conditioning_dict_includes_property_targets() -> None:
    client = MatterGenClient()
    targets = {"band_gap_eV": 2.0}
    cfg = MatterGenConfig(chemistry_scope=["Ga", "N"], property_targets=targets)
    cond = client.build_conditioning_dict(cfg)
    assert cond["properties"] == {"band_gap_eV": 2.0}


def test_conditioning_dict_omits_none_properties() -> None:
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Si"])
    cond = client.build_conditioning_dict(cfg)
    assert "properties" not in cond


# ---------------------------------------------------------------------------
# Generation — mattergen is mocked via sys.modules patching
# ---------------------------------------------------------------------------


def _mock_mattergen(return_value: list) -> ModuleType:
    """Create a fake mattergen module with a generate function."""
    mod = ModuleType("mattergen")
    mod.generate = MagicMock(return_value=return_value)  # type: ignore[attr-defined]
    return mod


def test_generate_returns_list_of_structures() -> None:
    mock_mod = _mock_mattergen(["struct_a", "struct_b"])
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Si"], num_samples=2)
    with patch.dict("sys.modules", {"mattergen": mock_mod}):
        result = client.generate(cfg)
    assert result == ["struct_a", "struct_b"]
    mock_mod.generate.assert_called_once()


def test_generate_partial_batch_no_raise() -> None:
    # Some samples fail convergence (returned as None by MatterGen).
    mock_mod = _mock_mattergen(["good_struct", None, "another_good"])
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Fe", "O"], num_samples=3)
    with patch.dict("sys.modules", {"mattergen": mock_mod}):
        result = client.generate(cfg)
    assert result == ["good_struct", "another_good"]


def test_generate_empty_batch_returns_empty_list() -> None:
    mock_mod = _mock_mattergen([])
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Si"])
    with patch.dict("sys.modules", {"mattergen": mock_mod}):
        result = client.generate(cfg)
    assert result == []


def test_generate_logs_batch_size(caplog: pytest.LogCaptureFixture) -> None:
    mock_mod = _mock_mattergen(["s1", "s2", "s3"])
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Li", "Co", "O"], num_samples=3)
    with (
        caplog.at_level(logging.INFO),
        patch.dict("sys.modules", {"mattergen": mock_mod}),
    ):
        client.generate(cfg)
    assert "3 sample(s)" in caplog.text
    assert "3/3 valid" in caplog.text


def test_generate_without_mattergen_installed_importerror() -> None:
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Si"])
    with patch.dict("sys.modules", {"mattergen": None}):
        with pytest.raises(ImportError, match="MatterGen is not installed"):
            client.generate(cfg)


def test_generate_whole_batch_failure_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_mod = _mock_mattergen([])
    mock_mod.generate.side_effect = RuntimeError("model crashed")  # type: ignore[attr-defined]
    client = MatterGenClient()
    cfg = MatterGenConfig(chemistry_scope=["Ti", "O"], num_samples=5)
    with (
        caplog.at_level(logging.WARNING),
        patch.dict("sys.modules", {"mattergen": mock_mod}),
    ):
        result = client.generate(cfg)
    assert result == []
    assert "batch failed" in caplog.text


def test_generate_passes_conditioning_to_mattergen() -> None:
    mock_mod = _mock_mattergen(["struct"])
    client = MatterGenClient()
    cfg = MatterGenConfig(
        chemistry_scope=["Na", "Cl"],
        space_group_number=225,
        num_samples=1,
        max_atoms=10,
    )
    with patch.dict("sys.modules", {"mattergen": mock_mod}):
        client.generate(cfg)
    call_kwargs = mock_mod.generate.call_args[1]  # type: ignore[union-attr]
    assert call_kwargs["conditioning"]["chemical_system"] == "Cl-Na"
    assert call_kwargs["conditioning"]["space_group"] == 225
    assert call_kwargs["num_samples"] == 1
    assert call_kwargs["max_atoms"] == 10
