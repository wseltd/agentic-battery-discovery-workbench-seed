"""Render a Report dataclass into a provenance-annotated output dict.

The renderer attaches provenance notes to each shortlist entry, flags
banned words with caveats, and computes a SHA-256 digest of the shortlist
for tamper detection.  No file I/O — pure data transformation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from discovery_workbench.report_constants import (
    APPROVED_WORDING,
    BANNED_WORDS,
    BANNED_WORDS_PATTERN,
)

# Re-export so callers can import from this module.
__all__ = ["render_report", "inject_provenance", "APPROVED_WORDING", "BANNED_WORDS"]
from discovery_workbench.report_schema import Report, ShortlistEntry

# Caveat appended when banned words are detected in the user brief or warnings.
_BANNED_WORD_CAVEAT = (
    "This report contains language that may overstate confidence in "
    "computational predictions. Review flagged terms carefully."
)


def inject_provenance(shortlist: list[ShortlistEntry]) -> list[dict]:
    """Convert ShortlistEntry dataclasses to dicts with provenance notes.

    Each entry is converted via ``dataclasses.asdict`` and augmented with
    a ``provenance_note`` key whose value is the approved wording for the
    entry's evidence level.

    Parameters
    ----------
    shortlist:
        List of ShortlistEntry dataclass instances.

    Returns
    -------
    list[dict]
        One dict per entry, with all dataclass fields plus ``provenance_note``.

    Raises
    ------
    KeyError
        If an entry's ``evidence_level`` has no approved wording in
        ``APPROVED_WORDING``.
    """
    result: list[dict] = []
    for entry in shortlist:
        entry_dict = dataclasses.asdict(entry)
        # Raise KeyError (not silently fall back) if wording is missing.
        entry_dict["provenance_note"] = APPROVED_WORDING[entry.evidence_level]
        result.append(entry_dict)
    return result


def render_report(report: Report) -> dict[str, Any]:
    """Render a Report into a provenance-annotated dict.

    Parameters
    ----------
    report:
        A validated Report dataclass instance.

    Returns
    -------
    dict[str, Any]
        Dict with keys: run_id, timestamp, branch, user_brief, budget,
        shortlist (with provenance_note per entry), warnings, annexes,
        provenance (sha256_shortlist digest and caveat if applicable).
    """
    shortlist_out = inject_provenance(report.shortlist)

    # Deterministic digest: canonical JSON of the shortlist, sorted keys.
    shortlist_json = json.dumps(shortlist_out, sort_keys=True, separators=(",", ":"))
    sha256_digest = hashlib.sha256(shortlist_json.encode("utf-8")).hexdigest()

    # Scan user_brief and warnings for banned words.
    texts_to_scan = [report.user_brief] + list(report.warnings)
    has_banned = any(
        BANNED_WORDS_PATTERN.search(text) for text in texts_to_scan
    )

    provenance: dict[str, Any] = {
        "sha256_shortlist": sha256_digest,
    }
    if has_banned:
        provenance["caveat"] = _BANNED_WORD_CAVEAT

    return {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "branch": report.branch,
        "user_brief": report.user_brief,
        "budget": {
            "max_cycles": report.budget.max_cycles,
            "max_batches": report.budget.max_batches,
            "shortlist_size": report.budget.shortlist_size,
        },
        "shortlist": shortlist_out,
        "warnings": list(report.warnings),
        "annexes": dict(report.annexes),
        "provenance": provenance,
    }
