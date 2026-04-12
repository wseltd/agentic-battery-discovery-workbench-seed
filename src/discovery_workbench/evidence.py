"""Evidence-level tracking for discovery workbench results.

Provides an ordered enum of evidence credibility levels and a helper
to stamp evidence metadata onto arbitrary dict-like containers.
"""

from __future__ import annotations

import datetime
import enum
import functools
from typing import Any, MutableMapping, Optional


@functools.total_ordering
class EvidenceLevel(enum.Enum):
    """Credibility level attached to a computed or reported property.

    Members are defined in ascending credibility order.  Comparison
    operators use definition order so that e.g.
    ``EvidenceLevel.ML_PREDICTED < EvidenceLevel.DFT_VERIFIED`` is True.

    Each member carries a human-readable *description* attribute.

    Parameters
    ----------
    label : str
        Machine-readable label (becomes ``member.value[0]``).
    description : str
        One-line explanation of this evidence tier.
    """

    # Tuple values: (label, description).  The enum stores the full
    # tuple as .value; .description is extracted in __init__.
    REQUESTED = ("requested", "Property was requested but not yet computed")
    GENERATED = ("generated", "Value produced by a generative model without validation")
    HEURISTIC_ESTIMATED = ("heuristic_estimated", "Estimated via rule-of-thumb or empirical correlation")
    ML_PREDICTED = ("ml_predicted", "Predicted by a trained ML surrogate model")
    ML_RELAXED = ("ml_relaxed", "Structure relaxed with an ML force field")
    SEMIEMPIRICAL_QC = ("semiempirical_qc", "Computed with a semiempirical quantum-chemistry method")
    DFT_VERIFIED = ("dft_verified", "Verified by density-functional theory calculation")
    EXPERIMENTAL_REPORTED = ("experimental_reported", "Reported from experimental measurement")
    UNKNOWN = ("unknown", "Evidence level could not be determined")

    # -- construction helpers ------------------------------------------------

    def __init__(self, label: str, description: str) -> None:  # noqa: D107 — handled by class docstring
        # Python unpacks the tuple value as positional args to __init__.
        # _rank is derived from definition order: first member = 0.
        self._rank: int = len(self.__class__.__members__)
        self.description: str = description

    # -- ordering (total_ordering fills the remaining operators) --------------

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EvidenceLevel):
            return NotImplemented
        return self._rank < other._rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvidenceLevel):
            return NotImplemented
        return self._rank == other._rank

    def __hash__(self) -> int:
        # Enum members must remain hashable after overriding __eq__.
        return hash(self._rank)

    def __repr__(self) -> str:
        return f"EvidenceLevel.{self.name}"


def attach_evidence(
    target: MutableMapping[str, Any],
    level: EvidenceLevel,
    source: Optional[str] = None,
) -> MutableMapping[str, Any]:
    """Stamp evidence metadata onto *target*.

    Parameters
    ----------
    target:
        A mutable dict-like container (plain dict, dataclass ``__dict__``, etc.).
    level:
        The credibility tier of the evidence being recorded.
    source:
        Optional identifier for the tool or model that produced the evidence
        (e.g. ``"xtb-6.6.1"`` or ``"mattergen-v0.1"``).  ``None`` when unknown.

    Returns
    -------
    MutableMapping[str, Any]
        The same *target* object, mutated in place, for call-chaining convenience.
    """
    target["_evidence_level"] = level
    target["_evidence_source"] = source
    target["_evidence_timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    return target
