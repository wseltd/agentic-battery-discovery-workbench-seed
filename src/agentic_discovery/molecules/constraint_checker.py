"""Molecular constraint checking.

Evaluates a molecule (given as SMILES) against a list of constraints:
numeric property bounds (molecular weight, cLogP, HBD, etc.) and
SMARTS substructure requirements (required or forbidden patterns).

Numeric properties are computed via RDKit descriptors.  SMARTS patterns
are matched with RDKit's ``HasSubstructMatch``.  Invalid SMARTS patterns
log a warning and count as a failed constraint rather than raising —
the caller gets a clear signal in the result without a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem  # provided by rdkit-pypi in pyproject.toml
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

# Supported numeric property calculators.
# Mapping from canonical property name to RDKit descriptor function.
_PROPERTY_CALCULATORS: dict[str, Any] = {
    "molecular_weight": Descriptors.MolWt,  # type: ignore[attr-defined]
    "clogp": Descriptors.MolLogP,  # type: ignore[attr-defined]
    "hbd": Descriptors.NumHDonors,  # type: ignore[attr-defined]
    "hba": Descriptors.NumHAcceptors,  # type: ignore[attr-defined]
    "tpsa": Descriptors.TPSA,  # type: ignore[attr-defined]
    "rotatable_bonds": Descriptors.NumRotatableBonds,  # type: ignore[attr-defined]
}


@dataclass(frozen=True)
class SingleMolecularConstraintResult:
    """Result of evaluating one constraint against a molecule.

    Parameters
    ----------
    constraint_name:
        Human-readable name of the constraint (e.g. ``"molecular_weight"``).
    operator:
        The comparison operator applied (e.g. ``">="``).
    target_value:
        The threshold or pattern the constraint checks against.
    actual_value:
        The computed value from the molecule.
    passed:
        Whether the constraint was satisfied.
    reason:
        Optional explanation when the constraint fails.
    """

    constraint_name: str
    operator: str
    target_value: Any
    actual_value: Any
    passed: bool
    reason: str | None = None


@dataclass
class MolecularConstraintResult:
    """Aggregated result of all constraints for a single molecule.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule that was checked.
    all_satisfied:
        True when every individual constraint passed.
        Auto-computed from *results* in ``__post_init__`` if not
        explicitly provided.
    results:
        Per-constraint evaluation results.
    """

    smiles: str
    all_satisfied: bool = field(default=True)
    results: list[SingleMolecularConstraintResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Derive all_satisfied from individual results when the caller
        # did not supply an explicit override.  The default (True) is
        # correct for an empty constraint list.
        if self.results:
            self.all_satisfied = all(r.passed for r in self.results)


class ConstraintChecker:
    """Checks molecules against a set of numeric and SMARTS constraints.

    Parameters
    ----------
    parsed_constraints:
        List of constraint dicts.  Each dict must have a ``"type"`` key
        (``"numeric"`` or ``"smarts"``).

        Numeric constraints require ``"property"``, ``"operator"``, and
        ``"value"`` keys.  The ``"range"`` operator expects ``"value"``
        to be a two-element tuple/list ``(min_val, max_val)`` where
        either element can be ``None`` for open-ended ranges.

        SMARTS constraints require ``"pattern"`` and ``"mode"``
        (``"required"`` or ``"forbidden"``) keys.
    """

    def __init__(self, parsed_constraints: list[dict[str, Any]]) -> None:
        self.parsed_constraints = parsed_constraints

    def _check_numeric(
        self,
        constraint_name: str,
        operator: str,
        actual_value: float,
        target_value: float | tuple[float | None, float | None],
    ) -> SingleMolecularConstraintResult:
        """Evaluate a single numeric comparison.

        Parameters
        ----------
        constraint_name:
            Property name (e.g. ``"molecular_weight"``).
        operator:
            One of ``"<="``, ``">="``, ``"=="``, ``"range"``.
        actual_value:
            The computed property value from the molecule.
        target_value:
            For scalar operators, a single float.  For ``"range"``, a
            tuple ``(min_val, max_val)`` where either can be ``None``
            for an open-ended bound.
        """
        if operator == "range":
            if not isinstance(target_value, (tuple, list)) or len(target_value) != 2:
                return SingleMolecularConstraintResult(
                    constraint_name=constraint_name,
                    operator=operator,
                    target_value=target_value,
                    actual_value=actual_value,
                    passed=False,
                    reason=f"range operator requires a (min, max) tuple, got {target_value!r}",
                )
            min_val, max_val = target_value
            above_min = min_val is None or actual_value >= min_val
            below_max = max_val is None or actual_value <= max_val
            passed = above_min and below_max
            reason = None if passed else (
                f"{constraint_name} {actual_value} not in range "
                f"[{min_val}, {max_val}]"
            )
            return SingleMolecularConstraintResult(
                constraint_name=constraint_name,
                operator=operator,
                target_value=target_value,
                actual_value=actual_value,
                passed=passed,
                reason=reason,
            )

        # At this point target_value is a scalar float (range handled above).
        scalar_target: float = float(target_value)  # type: ignore[arg-type]
        scalar_ops = {
            ">=": actual_value >= scalar_target,
            "<=": actual_value <= scalar_target,
            ">": actual_value > scalar_target,
            "<": actual_value < scalar_target,
            "==": actual_value == scalar_target,
        }
        if operator not in scalar_ops:
            return SingleMolecularConstraintResult(
                constraint_name=constraint_name,
                operator=operator,
                target_value=target_value,
                actual_value=actual_value,
                passed=False,
                reason=f"Unsupported operator '{operator}'",
            )

        passed = scalar_ops[operator]
        reason = None if passed else (
            f"{constraint_name} {actual_value} does not satisfy {operator} {target_value}"
        )
        return SingleMolecularConstraintResult(
            constraint_name=constraint_name,
            operator=operator,
            target_value=target_value,
            actual_value=actual_value,
            passed=passed,
            reason=reason,
        )

    def _check_smarts(
        self,
        constraint_name: str,
        smarts: str,
        mol: Chem.Mol,
        required: bool,
    ) -> SingleMolecularConstraintResult:
        """Evaluate a SMARTS substructure constraint.

        Parameters
        ----------
        constraint_name:
            Human-readable name for the constraint.
        smarts:
            SMARTS pattern string.
        mol:
            RDKit molecule object.
        required:
            If True the pattern must be present; if False it must be absent.
        """
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            logger.warning("Invalid SMARTS pattern: %s", smarts)
            return SingleMolecularConstraintResult(
                constraint_name=constraint_name,
                operator="required" if required else "forbidden",
                target_value=smarts,
                actual_value=None,
                passed=False,
                reason="Invalid SMARTS pattern",
            )

        has_match = mol.HasSubstructMatch(query)

        if required:
            passed = has_match
            reason = None if passed else f"Required SMARTS '{smarts}' not found"
        else:
            passed = not has_match
            reason = None if passed else f"Forbidden SMARTS '{smarts}' is present"

        return SingleMolecularConstraintResult(
            constraint_name=constraint_name,
            operator="required" if required else "forbidden",
            target_value=smarts,
            actual_value=has_match,
            passed=passed,
            reason=reason,
        )

    def check(self, mol: Chem.Mol, smiles: str) -> MolecularConstraintResult:
        """Evaluate all constraints against a molecule.

        The caller is responsible for parsing the SMILES into a Mol object.
        This method does not modify *mol* or filter molecules — it only
        reports pass/fail for each constraint.

        Parameters
        ----------
        mol:
            RDKit Mol object for the molecule.
        smiles:
            SMILES string (used for reporting, not re-parsed).

        Returns
        -------
        MolecularConstraintResult
            Aggregated result with per-constraint details.

        Raises
        ------
        ValueError
            If *mol* is None.
        """
        if mol is None:
            raise ValueError(
                f"mol must not be None (SMILES: {smiles}). "
                "The caller must provide a valid RDKit Mol object."
            )

        results: list[SingleMolecularConstraintResult] = []
        for constraint in self.parsed_constraints:
            ctype = constraint["type"]
            if ctype == "numeric":
                prop_name = constraint["property"]
                calculator = _PROPERTY_CALCULATORS.get(prop_name)
                if calculator is None:
                    results.append(SingleMolecularConstraintResult(
                        constraint_name=prop_name,
                        operator=constraint["operator"],
                        target_value=constraint["value"],
                        actual_value=None,
                        passed=False,
                        reason=f"Unknown property '{prop_name}'",
                    ))
                    continue
                actual = calculator(mol)
                results.append(
                    self._check_numeric(
                        prop_name,
                        constraint["operator"],
                        actual,
                        constraint["value"],
                    )
                )
            elif ctype == "smarts":
                mode = constraint["mode"]
                results.append(
                    self._check_smarts(
                        constraint_name="smarts",
                        smarts=constraint["pattern"],
                        mol=mol,
                        required=(mode == "required"),
                    )
                )

        return MolecularConstraintResult(smiles=smiles, results=results)
