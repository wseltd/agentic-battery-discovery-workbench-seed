"""Tests for report_constants -- approved wording, banned words, and regex pattern."""

from __future__ import annotations

from agentic_discovery_core.evidence import EvidenceLevel
from agentic_discovery_core.report_constants import (
    APPROVED_WORDING,
    BANNED_WORDS,
    BANNED_WORDS_PATTERN,
)


def test_approved_wording_covers_all_evidence_levels():
    for member in EvidenceLevel:
        label = member.value[0]
        assert label in APPROVED_WORDING, f"Missing wording for {label!r}"


def test_approved_wording_values_are_nonempty_strings():
    for label, wording in APPROVED_WORDING.items():
        assert isinstance(wording, str) and len(wording) > 0


def test_banned_words_is_frozenset():
    assert isinstance(BANNED_WORDS, frozenset)
    assert len(BANNED_WORDS) > 0


def test_banned_words_pattern_matches_each_banned_word():
    for word in BANNED_WORDS:
        assert BANNED_WORDS_PATTERN.search(word)


def test_banned_words_pattern_is_case_insensitive():
    assert BANNED_WORDS_PATTERN.search("DISCOVERED")
    assert BANNED_WORDS_PATTERN.search("Proven")
    assert BANNED_WORDS_PATTERN.search("Guaranteed Stable")


def test_banned_words_pattern_respects_word_boundaries():
    assert BANNED_WORDS_PATTERN.search("unproven") is None
    assert BANNED_WORDS_PATTERN.search("approvals") is None

    match = BANNED_WORDS_PATTERN.search("this is proven wrong")
    assert match is not None
    assert match.group().lower() == "proven"


def test_approved_wording_no_assurance_language():
    for label, wording in APPROVED_WORDING.items():
        match = BANNED_WORDS_PATTERN.search(wording)
        assert match is None

    assert APPROVED_WORDING["heuristic_estimated"] == (
        "Estimated via heuristic rules; not experimentally verified."
    )
