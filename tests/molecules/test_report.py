"""Tests for ammd.molecules.report frozen dataclasses and annex builder."""

from __future__ import annotations

import json

import pytest

from ammd.molecules.report import (
    HEURISTIC_WARNING_TEMPLATES,
    ConstraintResult,
    ExportPaths,
    MoleculeReportAnnex,
    NoveltyStats,
    UniquenessStats,
    ValidityStats,
    build_molecule_annex,
    format_heuristic_warning,
)


# -- fixtures ----------------------------------------------------------------

def _make_validity_stats(**overrides: object) -> ValidityStats:
    defaults = dict(
        total_generated=100,
        syntax_valid=95,
        valence_valid=93,
        charge_valid=92,
        stereo_flagged=3,
        salt_stripped=5,
        final_valid=90,
    )
    defaults.update(overrides)
    return ValidityStats(**defaults)


def _make_uniqueness_stats(**overrides: object) -> UniquenessStats:
    defaults = dict(
        total_valid=90,
        exact_duplicates_removed=5,
        near_duplicates_removed=3,
        unique_count=82,
    )
    defaults.update(overrides)
    return UniquenessStats(**defaults)


def _make_novelty_stats(**overrides: object) -> NoveltyStats:
    defaults = dict(
        reference_db="ChEMBL_36",
        reference_version="36",
        exact_known_count=10,
        close_analogue_count=20,
        novel_like_count=52,
        similarity_threshold=0.70,
    )
    defaults.update(overrides)
    return NoveltyStats(**defaults)


def _make_export_paths() -> ExportPaths:
    return ExportPaths(
        smiles_file="out/smiles.csv",
        inchikey_file="out/inchikeys.csv",
        sdf_dir="out/sdf",
        xyz_dir="out/xyz",
        xtb_handoff_dir="out/xtb",
    )


def _make_constraint_mw() -> ConstraintResult:
    return ConstraintResult(
        constraint_name="MW",
        window_min=300.0,
        window_max=450.0,
        satisfied_count=70,
        violated_count=12,
        satisfaction_rate=70 / 82,
    )


def _make_annex(**overrides: object) -> MoleculeReportAnnex:
    kwargs: dict = dict(
        generator_config={"model": "reinvent4"},
        validity_stats=_make_validity_stats(),
        uniqueness_stats=_make_uniqueness_stats(),
        novelty_stats=_make_novelty_stats(),
        constraint_breakdown=[_make_constraint_mw()],
        export_paths=_make_export_paths(),
        heuristic_warnings=[
            "Estimated by RDKit Crippen; heuristic prediction, may be inaccurate."
        ],
    )
    kwargs.update(overrides)
    return build_molecule_annex(**kwargs)


# -- ValidityStats -----------------------------------------------------------

def test_validity_stats_frozen():
    """Mutating a frozen ValidityStats must raise FrozenInstanceError."""
    vs = _make_validity_stats()
    assert vs.final_valid == 90
    with pytest.raises(AttributeError):
        vs.final_valid = 0  # type: ignore[misc]
    # Value unchanged after failed mutation attempt
    assert vs.final_valid == 90


def test_validity_stats_field_access():
    vs = _make_validity_stats(total_generated=200, final_valid=180)
    assert vs.total_generated == 200
    assert vs.final_valid == 180
    assert vs.syntax_valid == 95


# -- UniquenessStats ---------------------------------------------------------

def test_uniqueness_stats_counts_consistent():
    """unique_count must equal total_valid minus removed duplicates."""
    us = _make_uniqueness_stats()
    expected_unique = us.total_valid - us.exact_duplicates_removed - us.near_duplicates_removed
    assert us.unique_count == expected_unique
    assert us.unique_count == 82


def test_uniqueness_stats_frozen():
    us = _make_uniqueness_stats()
    assert us.total_valid == 90
    with pytest.raises(AttributeError):
        us.total_valid = 0  # type: ignore[misc]
    assert us.total_valid == 90


# -- NoveltyStats ------------------------------------------------------------

def test_novelty_stats_all_novel():
    """When no known or close analogues exist, all candidates are novel."""
    ns = _make_novelty_stats(exact_known_count=0, close_analogue_count=0, novel_like_count=82)
    assert ns.exact_known_count == 0
    assert ns.close_analogue_count == 0
    assert ns.novel_like_count == 82
    assert ns.reference_db == "ChEMBL_36"


def test_novelty_stats_all_known():
    """When all candidates are exact matches, novel_like_count is zero."""
    ns = _make_novelty_stats(exact_known_count=82, close_analogue_count=0, novel_like_count=0)
    assert ns.exact_known_count == 82
    assert ns.novel_like_count == 0
    assert ns.similarity_threshold == 0.70


# -- ConstraintResult --------------------------------------------------------

def test_constraint_result_satisfaction_rate_zero_division():
    """A constraint with zero candidates has zero satisfaction rate."""
    cr = ConstraintResult(
        constraint_name="logP",
        window_min=1.0,
        window_max=5.0,
        satisfied_count=0,
        violated_count=0,
        satisfaction_rate=0.0,
    )
    assert cr.satisfaction_rate == 0.0
    assert cr.satisfied_count == 0
    assert cr.violated_count == 0


def test_constraint_result_boolean_constraint():
    """A boolean constraint uses None for window bounds."""
    cr = ConstraintResult(
        constraint_name="lipinski_pass",
        window_min=None,
        window_max=None,
        satisfied_count=60,
        violated_count=22,
        satisfaction_rate=60 / 82,
    )
    assert cr.window_min is None
    assert cr.window_max is None
    assert cr.satisfied_count == 60
    assert cr.satisfaction_rate == pytest.approx(60 / 82)


def test_constraint_result_numeric_window():
    """A numeric-window constraint carries min/max bounds."""
    cr = _make_constraint_mw()
    assert cr.window_min == 300.0
    assert cr.window_max == 450.0
    assert cr.satisfaction_rate == pytest.approx(70 / 82)
    assert cr.constraint_name == "MW"


def test_constraint_result_frozen():
    """Mutating a frozen ConstraintResult must raise FrozenInstanceError."""
    cr = _make_constraint_mw()
    assert cr.constraint_name == "MW"
    assert cr.satisfied_count == 70
    with pytest.raises(AttributeError):
        cr.satisfied_count = 999  # type: ignore[misc]
    # Value unchanged after failed mutation attempt
    assert cr.satisfied_count == 70


def test_constraint_result_smarts_constraint():
    """A SMARTS-based constraint uses None bounds, tracks match counts."""
    cr = ConstraintResult(
        constraint_name="contains_phenol",
        window_min=None,
        window_max=None,
        satisfied_count=15,
        violated_count=67,
        satisfaction_rate=15 / 82,
    )
    assert cr.constraint_name == "contains_phenol"
    assert cr.satisfied_count == 15
    assert cr.violated_count == 67
    assert cr.satisfaction_rate == pytest.approx(15 / 82)


# -- Heuristic warnings ------------------------------------------------------

def test_heuristic_warnings_are_strings():
    """All approved templates produce non-empty strings; unknown keys raise ValueError."""
    # Every template key must format to a non-empty string
    # Templates that need kwargs get sample values via the fill dict
    fill: dict[str, dict[str, object]] = {
        "similarity_cutoff": {"threshold": 0.70},
        "xtb_semiempirical": {"level": 2},
    }
    for key in HEURISTIC_WARNING_TEMPLATES:
        result = format_heuristic_warning(key, **fill.get(key, {}))
        assert isinstance(result, str), f"Template {key!r} did not produce a string"
        assert len(result) > 0, f"Template {key!r} produced an empty string"

    # Unknown key must raise ValueError with an actionable message
    with pytest.raises(ValueError, match="Unknown heuristic-warning template key"):
        format_heuristic_warning("nonexistent_key")

    # Verify annex-level heuristic warning round-trip still works
    annex = _make_annex()
    assert len(annex.heuristic_warnings) == 1
    assert annex.heuristic_warnings[0] == (
        "Estimated by RDKit Crippen; heuristic prediction, may be inaccurate."
    )


# -- MoleculeReportAnnex / build_molecule_annex ------------------------------

def test_build_annex_roundtrip_all_fields():
    """build_molecule_annex populates all fields and they survive as_dict round-trip."""
    annex = _make_annex()

    assert annex.validity_stats.final_valid == 90
    assert annex.uniqueness_stats.unique_count == 82
    assert annex.novelty_stats.reference_db == "ChEMBL_36"
    assert annex.novelty_stats.similarity_threshold == 0.70
    assert len(annex.constraint_breakdown) == 1
    assert annex.constraint_breakdown[0].constraint_name == "MW"
    assert annex.export_paths.smiles_file == "out/smiles.csv"
    assert annex.generator_config == {"model": "reinvent4"}

    d = annex.as_dict()
    assert d["validity_stats"]["final_valid"] == 90
    assert d["novelty_stats"]["reference_db"] == "ChEMBL_36"
    assert d["export_paths"]["sdf_dir"] == "out/sdf"


def test_build_annex_rejects_wrong_validity_stats_type():
    """Passing a plain dict instead of ValidityStats raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        _make_annex(validity_stats={"final_valid": 90})
    assert "validity_stats must be a ValidityStats" in str(exc_info.value)
    assert "got dict" in str(exc_info.value)


def test_build_annex_rejects_non_dict_generator_config():
    """Passing a non-dict generator_config raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        _make_annex(generator_config="reinvent4")
    assert "generator_config must be a dict" in str(exc_info.value)
    assert "got str" in str(exc_info.value)


def test_build_annex_rejects_non_constraint_in_breakdown():
    """Passing a plain dict in the constraint list raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        _make_annex(constraint_breakdown=[{"name": "MW"}])
    assert "constraint_breakdown[0] must be a ConstraintResult" in str(exc_info.value)
    assert "got dict" in str(exc_info.value)


def test_build_annex_rejects_non_string_warning():
    """Passing a non-string in heuristic_warnings raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        _make_annex(heuristic_warnings=[42])
    assert "heuristic_warnings[0] must be a str" in str(exc_info.value)
    assert "got int" in str(exc_info.value)


def test_annex_asdict_serializable():
    """The annex dict must be JSON-serializable without custom encoders."""
    annex = _make_annex()
    d = annex.as_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["validity_stats"]["total_generated"] == 100
    assert deserialized["novelty_stats"]["similarity_threshold"] == 0.70
    assert deserialized["constraint_breakdown"][0]["constraint_name"] == "MW"
