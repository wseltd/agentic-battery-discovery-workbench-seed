"""Molecular duplicate detection and novelty classification.

Provides a unified import path for the DuplicateDetector (T019) and
ChEMBLNoveltyChecker (T020) APIs under the discovery_workbench package.
"""

from workbench.molecules.duplicate_detector import (
    NEAR_DUPLICATE_THRESHOLD,
    DuplicateDetector,
    DuplicateResult,
    DuplicateStatus,
)
from workbench.molecules.novelty_checker import (
    CLOSE_ANALOGUE_THRESHOLD,
    NOVEL_LIKE_THRESHOLD,
    ChEMBLNoveltyChecker,
    NoveltyClass,
    NoveltyResult,
)

__all__ = [
    "CLOSE_ANALOGUE_THRESHOLD",
    "ChEMBLNoveltyChecker",
    "DuplicateDetector",
    "DuplicateResult",
    "DuplicateStatus",
    "NEAR_DUPLICATE_THRESHOLD",
    "NOVEL_LIKE_THRESHOLD",
    "NoveltyClass",
    "NoveltyResult",
]
