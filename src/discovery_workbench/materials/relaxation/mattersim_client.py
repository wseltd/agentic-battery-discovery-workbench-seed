"""MatterSim client for ML-driven crystal structure relaxation.

Wraps Microsoft's MatterSim machine-learning force field for
geometry optimisation of inorganic crystal structures.  Import of
the mattersim package is deferred to relaxation time so dataclasses
and configuration work without the model installed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from pymatgen.core import Structure

logger = logging.getLogger(__name__)

__all__ = ["MatterSimRelaxer", "RelaxationResult", "relax"]

# Default force convergence threshold in eV/Å — standard for ML force fields.
_DEFAULT_FMAX_EV_ANG = 0.05

# Default maximum optimisation steps before declaring non-convergence.
_DEFAULT_MAX_STEPS = 500


@dataclass(frozen=True, slots=True)
class RelaxationResult:
    """Result of an ML force-field relaxation.

    Args:
        relaxed_structure: Optimised pymatgen Structure.
        energy_eV: Total potential energy in electronvolts.
        energy_per_atom_eV: Energy divided by the number of sites.
        converged: Whether forces fell below *fmax* within *max_steps*.
        steps_taken: Number of optimiser iterations executed.
        evidence_level: Credibility tag for downstream tracking.
    """

    relaxed_structure: Structure
    energy_eV: float
    energy_per_atom_eV: float
    converged: bool
    steps_taken: int
    evidence_level: str = field(default="ml_relaxed")


class MatterSimRelaxer:
    """Relaxer backed by MatterSim's ML force field.

    Defers import of mattersim to :meth:`relax` so configuration and
    result handling work without the heavy model dependency installed.

    Args:
        fmax: Force convergence criterion in eV/Å.
        max_steps: Maximum LBFGS iterations.
    """

    def __init__(
        self,
        fmax: float = _DEFAULT_FMAX_EV_ANG,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        if fmax <= 0:
            raise ValueError(f"fmax must be positive, got {fmax}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.fmax = fmax
        self.max_steps = max_steps

    def relax(self, structure: Structure) -> RelaxationResult:
        """Relax a crystal structure using MatterSim's ML potential.

        Converts the pymatgen Structure to an ASE Atoms object, attaches
        the MatterSim calculator, wraps with ``FrechetCellFilter`` to allow
        cell relaxation, and runs LBFGS.

        Args:
            structure: Input pymatgen Structure to relax.

        Returns:
            RelaxationResult with relaxed geometry and energetics.

        Raises:
            ImportError: If mattersim is not installed.
            TypeError: If *structure* is not a pymatgen Structure.
            ValueError: If the relaxed energy is NaN (indicates calculator
                failure or numerical instability).
        """
        if not isinstance(structure, Structure):
            raise TypeError(
                f"Expected pymatgen Structure, got {type(structure).__name__}"
            )

        logger.info(
            "Relaxing %s (%d sites), fmax=%.4f, max_steps=%d",
            structure.composition.reduced_formula,
            len(structure),
            self.fmax,
            self.max_steps,
        )

        try:
            from mattersim.forcefield import MatterSimCalculator
        except ImportError:
            logger.error("mattersim package is not installed")
            raise ImportError(
                "MatterSim is not installed. "
                "Install it with: pip install mattersim"
            ) from None

        from ase.constraints import FrechetCellFilter
        from ase.optimize import LBFGS
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = MatterSimCalculator()

        filtered = FrechetCellFilter(atoms)
        optimiser = LBFGS(filtered, logfile=None)
        converged = optimiser.run(fmax=self.fmax, steps=self.max_steps)
        steps_taken = optimiser.get_number_of_steps()

        energy = atoms.get_potential_energy()
        if math.isnan(energy):
            raise ValueError(
                "Relaxed energy is NaN — MatterSim calculator failure or "
                "numerical instability for "
                f"{structure.composition.reduced_formula}"
            )

        num_atoms = len(atoms)
        relaxed_structure: Structure = AseAtomsAdaptor.get_structure(atoms)

        result = RelaxationResult(
            relaxed_structure=relaxed_structure,
            energy_eV=energy,
            energy_per_atom_eV=energy / num_atoms,
            converged=bool(converged),
            steps_taken=steps_taken,
        )

        logger.info(
            "Relaxation complete: converged=%s, steps=%d, E=%.4f eV (%.4f eV/atom)",
            result.converged,
            result.steps_taken,
            result.energy_eV,
            result.energy_per_atom_eV,
        )

        return result


def relax(structure: Structure, **kwargs: float | int) -> RelaxationResult:
    """Module-level convenience for relaxing a structure.

    Delegates to a default :class:`MatterSimRelaxer` instance.

    Args:
        structure: Pymatgen Structure to relax.
        **kwargs: Forwarded to :class:`MatterSimRelaxer` (fmax, max_steps).

    Returns:
        RelaxationResult with relaxed geometry and energetics.

    Raises:
        ImportError: If mattersim is not installed.
        ValueError: If relaxed energy is NaN.
    """
    relaxer = MatterSimRelaxer(**kwargs)  # type: ignore[arg-type]
    return relaxer.relax(structure)
