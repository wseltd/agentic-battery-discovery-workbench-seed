"""Tests for banned_words.scan_banned_words -- word-boundary regex scanning."""

from __future__ import annotations

from agentic_discovery_core.banned_words import scan_banned_words
from agentic_discovery_core.report_constants import BANNED_WORDS


def test_scan_empty_list_returns_empty():
    assert scan_banned_words([]) == []


def test_scan_no_banned_words_returns_empty():
    assert scan_banned_words(["this is a clean sentence"]) == []


def test_scan_single_banned_word_returns_one_caveat():
    result = scan_banned_words(["We discovered a compound"])
    assert len(result) == 1
    assert '"discovered"' in result[0]
    assert "not substantiated" in result[0]


def test_scan_multiple_banned_words_returns_multiple_caveats():
    result = scan_banned_words(["discovered a compound", "proven effective"])
    assert len(result) == 2
    words_in_caveats = {c.split('"')[1] for c in result}
    assert words_in_caveats == {"discovered", "proven"}


def test_scan_deduplicates_same_word_across_texts():
    result = scan_banned_words(["discovered X", "also discovered Y"])
    assert len(result) == 1
    assert '"discovered"' in result[0]


def test_scan_deduplicates_same_word_within_text():
    result = scan_banned_words(["discovered and then discovered again"])
    assert len(result) == 1


def test_scan_case_insensitive():
    result = scan_banned_words(["DISCOVERED something"])
    assert len(result) == 1
    assert '"discovered"' in result[0]


def test_scan_word_boundary_no_false_positive():
    result = scan_banned_words(["unproven hypothesis", "check approvals"])
    assert result == []


def test_scan_multi_word_banned_term():
    result = scan_banned_words(["the material is guaranteed stable"])
    assert len(result) == 1
    assert '"guaranteed stable"' in result[0]


def test_scan_preserves_first_seen_order():
    result = scan_banned_words(["proven theory", "discovered compound"])
    assert '"proven"' in result[0]
    assert '"discovered"' in result[1]


def test_scan_all_banned_words_matched():
    for word in sorted(BANNED_WORDS):
        result = scan_banned_words([f"this is {word} here"])
        assert len(result) == 1, f"Failed to match banned word: {word!r}"
        assert f'"{word}"' in result[0]


def test_scan_empty_string_in_list():
    assert scan_banned_words(["", "", ""]) == []
