"""Tests for report_constants — approved wording, banned words, and regex pattern."""

from __future__ import annotations

from discovery_workbench.evidence import EvidenceLevel
from discovery_workbench.report_constants import (
    APPROVED_WORDING,
    BANNED_WORDS,
    BANNED_WORDS_PATTERN,
)


def test_approved_wording_covers_all_evidence_levels():
    """Every EvidenceLevel member label must have approved wording."""
    for member in EvidenceLevel:
        label = member.value[0]
        assert label in APPROVED_WORDING, f"Missing wording for {label!r}"


def test_approved_wording_values_are_nonempty_strings():
    """Each approved wording value must be a non-empty string."""
    for label, wording in APPROVED_WORDING.items():
        assert isinstance(wording, str) and len(wording) > 0, (
            f"Wording for {label!r} is empty or not a string"
        )


def test_banned_words_is_frozenset():
    """BANNED_WORDS must be a frozenset for immutability."""
    assert isinstance(BANNED_WORDS, frozenset)
    assert len(BANNED_WORDS) > 0


def test_banned_words_pattern_matches_each_banned_word():
    """The compiled pattern must match each banned word in isolation."""
    for word in BANNED_WORDS:
        assert BANNED_WORDS_PATTERN.search(word), (
            f"Pattern failed to match banned word: {word!r}"
        )


def test_banned_words_pattern_is_case_insensitive():
    """Pattern must match regardless of case."""
    assert BANNED_WORDS_PATTERN.search("DISCOVERED")
    assert BANNED_WORDS_PATTERN.search("Proven")
    assert BANNED_WORDS_PATTERN.search("Guaranteed Stable")


def test_banned_words_pattern_respects_word_boundaries():
    """Substrings of banned words must NOT match when not at word boundaries."""
    # 'unproven' does not have 'proven' at a word boundary start
    assert BANNED_WORDS_PATTERN.search("unproven") is None
    # 'approvals' does not contain any banned word at boundaries
    assert BANNED_WORDS_PATTERN.search("approvals") is None

    # Positive control: the bare banned word 'proven' must match
    match = BANNED_WORDS_PATTERN.search("this is proven wrong")
    assert match is not None
    assert match.group().lower() == "proven"


def test_approved_wording_no_assurance_language():
    """Approved wording must not use the banned words it polices.

    This catches self-contradictory wording — e.g. using 'validated'
    in the approved provenance string for a heuristic level.
    """
    for label, wording in APPROVED_WORDING.items():
        match = BANNED_WORDS_PATTERN.search(wording)
        assert match is None, (
            f"Approved wording for {label!r} contains banned word "
            f"{match.group()!r}: {wording!r}"
        )

    # Value assertion: verify a known wording string is what we expect
    assert APPROVED_WORDING["heuristic_estimated"] == (
        "Estimated via heuristic rules; not experimentally verified."
    )
