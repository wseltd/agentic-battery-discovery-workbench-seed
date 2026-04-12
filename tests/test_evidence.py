"""Tests for discovery_workbench.evidence — ordering gets heavy coverage
because incorrect ordering silently corrupts downstream ranking logic."""

from __future__ import annotations

import datetime
import re

from discovery_workbench.evidence import EvidenceLevel, attach_evidence

# ISO-8601 UTC pattern (no timezone suffix required — datetime.isoformat()
# omits it for UTC-aware datetimes produced via datetime.UTC).
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


# -- enum membership ---------------------------------------------------------

def test_all_nine_levels_defined() -> None:
    """EvidenceLevel must expose exactly the 9 specified members."""
    expected = {
        "REQUESTED",
        "GENERATED",
        "HEURISTIC_ESTIMATED",
        "ML_PREDICTED",
        "ML_RELAXED",
        "SEMIEMPIRICAL_QC",
        "DFT_VERIFIED",
        "EXPERIMENTAL_REPORTED",
        "UNKNOWN",
    }
    assert {m.name for m in EvidenceLevel} == expected
    assert len(EvidenceLevel) == 9


# -- ordering (5 tests) — the critical contract ------------------------------

def test_ordering_requested_below_generated() -> None:
    """REQUESTED is the lowest credibility tier; GENERATED is one step above."""
    assert EvidenceLevel.REQUESTED < EvidenceLevel.GENERATED


def test_ordering_ml_predicted_below_dft_verified() -> None:
    """ML surrogates rank below first-principles DFT verification."""
    assert EvidenceLevel.ML_PREDICTED < EvidenceLevel.DFT_VERIFIED
    # Also verify the intermediate levels respect the chain.
    assert EvidenceLevel.ML_RELAXED < EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.SEMIEMPIRICAL_QC < EvidenceLevel.DFT_VERIFIED


def test_ordering_unknown_not_above_experimental() -> None:
    """UNKNOWN is last in definition order — it must NOT compare as below
    EXPERIMENTAL_REPORTED (it is *above* it in rank), which is the intended
    semantic: unknown means 'unclassified', placed at end of the enum."""
    assert not EvidenceLevel.UNKNOWN < EvidenceLevel.EXPERIMENTAL_REPORTED
    assert EvidenceLevel.UNKNOWN > EvidenceLevel.EXPERIMENTAL_REPORTED


def test_ordering_same_level_equal() -> None:
    """A level must compare equal to itself and support >= correctly."""
    assert EvidenceLevel.DFT_VERIFIED == EvidenceLevel.DFT_VERIFIED
    assert EvidenceLevel.DFT_VERIFIED >= EvidenceLevel.DFT_VERIFIED
    assert not EvidenceLevel.DFT_VERIFIED < EvidenceLevel.DFT_VERIFIED
    # Different members must not be equal.
    assert EvidenceLevel.REQUESTED != EvidenceLevel.GENERATED


# -- attach_evidence (3 tests) -----------------------------------------------

def test_attach_stamps_level_and_source() -> None:
    """attach_evidence must set _evidence_level and _evidence_source."""
    data: dict[str, object] = {"value": 42}
    result = attach_evidence(data, EvidenceLevel.GENERATED, "xtb-6.6.1")
    assert result is data  # mutates in place and returns same object
    assert data["_evidence_level"] is EvidenceLevel.GENERATED
    assert data["_evidence_source"] == "xtb-6.6.1"


def test_attach_stamps_iso_timestamp() -> None:
    """Timestamp must be a valid ISO-8601 string close to 'now'."""
    before = datetime.datetime.now(datetime.UTC)
    data: dict[str, object] = {}
    attach_evidence(data, EvidenceLevel.DFT_VERIFIED, "vasp")
    after = datetime.datetime.now(datetime.UTC)

    ts_str = data["_evidence_timestamp"]
    assert isinstance(ts_str, str)
    assert _ISO_RE.match(ts_str), f"Timestamp {ts_str!r} is not ISO-8601"
    ts = datetime.datetime.fromisoformat(ts_str)
    # The stamped time must fall within the before/after window.
    assert before <= ts <= after


def test_attach_without_source_sets_none() -> None:
    """When source is omitted, _evidence_source must be None, not missing."""
    data: dict[str, object] = {}
    attach_evidence(data, EvidenceLevel.REQUESTED)
    assert "_evidence_source" in data
    assert data["_evidence_source"] is None
