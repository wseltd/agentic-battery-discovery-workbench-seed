"""MatterGen client for conditional crystal structure generation.

Wraps Microsoft's MatterGen model for property-conditioned generation
of inorganic crystal structures.  Import of the mattergen package is
deferred to generation time so config and conditioning logic work
without the model installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Hard ceiling for atom count per unit cell.  MatterGen's training data
# caps at 20 atoms; requesting more produces unreliable structures.
_MAX_ATOMS_CEILING = 20

# Minimum number of samples per generation batch.
_MIN_NUM_SAMPLES = 1


@dataclass(slots=True)
class MatterGenConfig:
    """Configuration for a MatterGen generation run.

    Args:
        chemistry_scope: Allowed chemical elements (e.g. ["Li", "Fe", "P", "O"]).
        space_group_number: International space group number (1--230), or None
            for unconstrained symmetry.
        num_samples: Number of structures to generate per batch.
        property_targets: Target property values keyed by name
            (e.g. {"band_gap_eV": 1.5, "bulk_modulus_GPa": 100.0}).
        max_atoms: Maximum atoms per unit cell (silently clamped to 20).

    Raises:
        ValueError: If chemistry_scope is empty or num_samples < 1.
    """

    chemistry_scope: list[str]
    space_group_number: int | None = None
    num_samples: int = 1
    property_targets: dict[str, float] | None = None
    max_atoms: int = _MAX_ATOMS_CEILING

    def __post_init__(self) -> None:
        if not self.chemistry_scope:
            raise ValueError("chemistry_scope must contain at least one element")
        if self.num_samples < _MIN_NUM_SAMPLES:
            raise ValueError(
                f"num_samples must be >= {_MIN_NUM_SAMPLES}, got {self.num_samples}"
            )
        # Clamp silently — MatterGen produces unreliable structures above 20 atoms.
        if self.max_atoms > _MAX_ATOMS_CEILING:
            self.max_atoms = _MAX_ATOMS_CEILING

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatterGenConfig:
        """Construct from a plain dictionary (e.g. parsed from JSON/YAML config).

        Args:
            data: Must contain ``chemistry_scope``.  May contain
                ``space_group_number``, ``num_samples``, ``property_targets``,
                ``max_atoms``.

        Returns:
            Populated MatterGenConfig.

        Raises:
            KeyError: If ``chemistry_scope`` is missing from *data*.
            ValueError: If validation fails (see ``__post_init__``).
        """
        sg = data.get("space_group_number")
        targets = data.get("property_targets")
        return cls(
            chemistry_scope=list(data["chemistry_scope"]),
            space_group_number=int(sg) if sg is not None else None,
            num_samples=int(data.get("num_samples", 1)),
            property_targets=dict(targets) if targets is not None else None,
            max_atoms=int(data.get("max_atoms", _MAX_ATOMS_CEILING)),
        )

    def to_chemical_system(self) -> str:
        """Join sorted elements with '-' for MatterGen's chemical system format.

        Returns:
            Chemical system string (e.g. "Fe-Li-O-P").
        """
        return "-".join(sorted(self.chemistry_scope))


class MatterGenClient:
    """Client for generating crystal structures with MatterGen.

    Defers import of the mattergen package to generation time so
    configuration and conditioning logic can be used independently.
    """

    def build_conditioning_dict(self, config: MatterGenConfig) -> dict[str, Any]:
        """Build MatterGen conditioning parameters from config.

        Always includes the chemical system.  Includes space group and
        property targets only when explicitly set — omitting None values
        lets MatterGen use its unconditional defaults.

        Args:
            config: Generation configuration.

        Returns:
            Conditioning dict with chemical_system and optional
            space_group / properties entries.
        """
        conditioning: dict[str, Any] = {
            "chemical_system": config.to_chemical_system(),
        }
        if config.space_group_number is not None:
            conditioning["space_group"] = config.space_group_number
        if config.property_targets:
            conditioning["properties"] = dict(config.property_targets)
        return conditioning

    def generate(self, config: MatterGenConfig) -> list[Any]:
        """Generate crystal structures conditioned on config.

        Args:
            config: Generation configuration specifying chemistry,
                symmetry, and property targets.

        Returns:
            List of generated structures.  Empty if the batch
            produces no valid structures.

        Raises:
            ImportError: If the mattergen package is not installed.
        """
        logger.info(
            "Generating %d sample(s) for %s (sg=%s, max_atoms=%d)",
            config.num_samples,
            config.to_chemical_system(),
            config.space_group_number,
            config.max_atoms,
        )

        try:
            import mattergen
        except ImportError:
            logger.error("mattergen package is not installed")
            raise ImportError(
                "MatterGen is not installed. "
                "Install it with: pip install mattergen"
            ) from None

        conditioning = self.build_conditioning_dict(config)

        try:
            raw_batch = mattergen.generate(
                conditioning=conditioning,
                num_samples=config.num_samples,
                max_atoms=config.max_atoms,
            )
        except Exception as exc:
            # Whole-batch failure — log and return empty rather than crash
            # the pipeline.  Caller decides whether to retry.
            logger.warning("MatterGen batch failed: %s", exc)
            return []

        # Filter out failed samples (None entries from partial convergence
        # failures) without raising.
        structures = [s for s in raw_batch if s is not None]

        logger.info(
            "Batch complete: %d/%d valid structures",
            len(structures),
            config.num_samples,
        )
        return structures


def generate(config: MatterGenConfig) -> list[Any]:
    """Module-level convenience for generating structures.

    Delegates to a default :class:`MatterGenClient` instance.

    Args:
        config: Generation configuration.

    Returns:
        List of generated structures.

    Raises:
        ImportError: If the mattergen package is not installed.
    """
    client = MatterGenClient()
    return client.generate(config)
