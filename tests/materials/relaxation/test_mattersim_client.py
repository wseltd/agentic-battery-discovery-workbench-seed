"""Tests for MatterSim relaxation client: RelaxationResult, MatterSimRelaxer, relax."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Lattice, Structure

from discovery_workbench.materials.relaxation.mattersim_client import (
    MatterSimRelaxer,
    RelaxationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nacl_structure() -> Structure:
    """NaCl rock-salt primitive cell for testing."""
    return Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _build_mock_modules(
    converged: bool = True,
    steps: int = 42,
    energy: float = -10.0,
    n_atoms: int = 2,
    relaxed_structure: Structure | None = None,
) -> dict:
    """Build fake sys.modules entries for mattersim, ase, and pymatgen.io.ase.

    Returns a dict with:
      - modules: dict for patch.dict("sys.modules", ...)
      - optimiser: mock optimiser for assertions
      - filter_cls: mock FrechetCellFilter class for assertions
      - atoms: mock Atoms object for assertions
      - adaptor: mock AseAtomsAdaptor for assertions
    """
    if relaxed_structure is None:
        relaxed_structure = _nacl_structure()

    # --- Mock Atoms ---
    mock_atoms = MagicMock()
    mock_atoms.get_potential_energy.return_value = energy
    mock_atoms.__len__ = lambda self: n_atoms
    mock_atoms.calc = None

    # --- Mock optimiser ---
    mock_optimiser = MagicMock()
    mock_optimiser.run.return_value = converged
    mock_optimiser.get_number_of_steps.return_value = steps

    # --- ase.optimize module ---
    mock_lbfgs_cls = MagicMock(return_value=mock_optimiser)
    ase_optimize = ModuleType("ase.optimize")
    ase_optimize.LBFGS = mock_lbfgs_cls  # type: ignore[attr-defined]

    # --- ase.constraints module ---
    mock_filter_cls = MagicMock(return_value=MagicMock())
    ase_constraints = ModuleType("ase.constraints")
    ase_constraints.FrechetCellFilter = mock_filter_cls  # type: ignore[attr-defined]

    # --- ase root module ---
    ase_root = ModuleType("ase")

    # --- mattersim modules ---
    mattersim_ff = ModuleType("mattersim.forcefield")
    mattersim_ff.MatterSimCalculator = MagicMock  # type: ignore[attr-defined]
    mattersim_root = ModuleType("mattersim")
    mattersim_root.forcefield = mattersim_ff  # type: ignore[attr-defined]

    # --- pymatgen.io.ase module ---
    mock_adaptor = MagicMock()
    mock_adaptor.get_atoms.return_value = mock_atoms
    mock_adaptor.get_structure.return_value = relaxed_structure
    pymatgen_io_ase = ModuleType("pymatgen.io.ase")
    pymatgen_io_ase.AseAtomsAdaptor = mock_adaptor  # type: ignore[attr-defined]

    modules = {
        "ase": ase_root,
        "ase.optimize": ase_optimize,
        "ase.constraints": ase_constraints,
        "mattersim": mattersim_root,
        "mattersim.forcefield": mattersim_ff,
        "pymatgen.io.ase": pymatgen_io_ase,
    }

    return {
        "modules": modules,
        "optimiser": mock_optimiser,
        "filter_cls": mock_filter_cls,
        "lbfgs_cls": mock_lbfgs_cls,
        "atoms": mock_atoms,
        "adaptor": mock_adaptor,
    }


def _relax_with_mocks(
    relaxer: MatterSimRelaxer,
    structure: Structure,
    mocks: dict,
) -> RelaxationResult:
    """Run relaxer.relax with all heavy deps mocked via sys.modules."""
    with patch.dict("sys.modules", mocks["modules"]):
        return relaxer.relax(structure)


# ---------------------------------------------------------------------------
# RelaxationResult dataclass tests
# ---------------------------------------------------------------------------


def test_relaxation_result_fields() -> None:
    """All six declared fields exist on RelaxationResult."""
    field_names = {f.name for f in dataclasses.fields(RelaxationResult)}
    assert field_names == {
        "relaxed_structure",
        "energy_eV",
        "energy_per_atom_eV",
        "converged",
        "steps_taken",
        "evidence_level",
    }


def test_relaxation_result_evidence_level_is_ml_relaxed() -> None:
    """Default evidence_level is 'ml_relaxed' — matches EvidenceLevel.ML_RELAXED label."""
    struct = _nacl_structure()
    result = RelaxationResult(
        relaxed_structure=struct,
        energy_eV=-10.0,
        energy_per_atom_eV=-5.0,
        converged=True,
        steps_taken=10,
    )
    assert result.evidence_level == "ml_relaxed"


# ---------------------------------------------------------------------------
# MatterSimRelaxer.relax — happy-path and convergence
# ---------------------------------------------------------------------------


def test_relax_returns_relaxation_result() -> None:
    """relax() returns a RelaxationResult with correct field values."""
    struct = _nacl_structure()
    mocks = _build_mock_modules(converged=True, steps=30, energy=-8.0, n_atoms=2)
    relaxer = MatterSimRelaxer()
    result = _relax_with_mocks(relaxer, struct, mocks)
    assert isinstance(result, RelaxationResult)
    assert result.energy_eV == pytest.approx(-8.0)
    assert result.converged is True
    assert result.steps_taken == 30
    assert result.evidence_level == "ml_relaxed"


def test_relax_converged_true_when_fmax_met() -> None:
    """converged is True when the optimiser reaches the force criterion."""
    struct = _nacl_structure()
    mocks = _build_mock_modules(converged=True, steps=25)
    relaxer = MatterSimRelaxer()
    result = _relax_with_mocks(relaxer, struct, mocks)
    assert result.converged is True
    assert result.steps_taken == 25


def test_relax_converged_false_when_max_steps_hit() -> None:
    """converged is False when the optimiser exhausts max_steps."""
    struct = _nacl_structure()
    mocks = _build_mock_modules(converged=False, steps=500)
    relaxer = MatterSimRelaxer(max_steps=500)
    result = _relax_with_mocks(relaxer, struct, mocks)
    assert result.converged is False
    assert result.steps_taken == 500


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_relax_nan_energy_raises_valueerror() -> None:
    """NaN energy from the calculator must raise ValueError, not silently propagate."""
    struct = _nacl_structure()
    mocks = _build_mock_modules(energy=float("nan"))
    relaxer = MatterSimRelaxer()
    with pytest.raises(ValueError, match="NaN") as exc_info:
        _relax_with_mocks(relaxer, struct, mocks)
    assert "numerical instability" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Cell filter usage
# ---------------------------------------------------------------------------


def test_relax_uses_cell_filter() -> None:
    """The optimiser must wrap atoms in FrechetCellFilter for cell relaxation."""
    struct = _nacl_structure()
    mocks = _build_mock_modules()
    relaxer = MatterSimRelaxer()
    _relax_with_mocks(relaxer, struct, mocks)
    # FrechetCellFilter was called with the mock atoms object.
    assert mocks["filter_cls"].call_count == 1


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


def test_relax_default_fmax() -> None:
    """Default fmax is 0.05 eV/Å."""
    relaxer = MatterSimRelaxer()
    assert relaxer.fmax == pytest.approx(0.05)


def test_relax_default_max_steps() -> None:
    """Default max_steps is 500."""
    relaxer = MatterSimRelaxer()
    assert relaxer.max_steps == 500


# ---------------------------------------------------------------------------
# Energy-per-atom correctness
# ---------------------------------------------------------------------------


def test_relax_energy_per_atom_correct() -> None:
    """energy_per_atom_eV must equal energy_eV / number_of_atoms."""
    total_energy = -12.6
    n_atoms = 3
    struct = _nacl_structure()
    mocks = _build_mock_modules(energy=total_energy, n_atoms=n_atoms)
    relaxer = MatterSimRelaxer()
    result = _relax_with_mocks(relaxer, struct, mocks)
    assert result.energy_per_atom_eV == pytest.approx(total_energy / n_atoms)


# ---------------------------------------------------------------------------
# Composition preservation
# ---------------------------------------------------------------------------


def test_relax_preserves_composition() -> None:
    """Relaxation must not change the chemical composition."""
    struct = _nacl_structure()
    mocks = _build_mock_modules(relaxed_structure=struct)
    relaxer = MatterSimRelaxer()
    result = _relax_with_mocks(relaxer, struct, mocks)
    assert result.relaxed_structure.composition == struct.composition


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------


def test_relaxer_lazy_import() -> None:
    """MatterSimRelaxer construction must NOT import mattersim.

    If mattersim were eagerly imported at class or module level,
    constructing the relaxer would fail when the package is absent.
    The import only happens inside relax().
    """
    with patch.dict("sys.modules", {"mattersim": None, "mattersim.forcefield": None}):
        relaxer = MatterSimRelaxer()
        assert relaxer.fmax == pytest.approx(0.05)
    # Calling relax should trigger the import and fail.
    with patch.dict("sys.modules", {"mattersim": None, "mattersim.forcefield": None}):
        with pytest.raises(ImportError, match="MatterSim is not installed"):
            relaxer.relax(_nacl_structure())
