"""Molecular duplicate detection and novelty classification.

Provides a unified import path for the DuplicateDetector and
ChEMBLNoveltyChecker APIs under the novelty sub-package.
"""

from agentic_molecule_discovery.novelty.duplicate_detector import (
    NEAR_DUPLICATE_THRESHOLD,
    DuplicateDetector,
    DuplicateResult,
    DuplicateStatus,
)
from agentic_molecule_discovery.novelty.novelty_checker import (
    CLOSE_ANALOGUE_THRESHOLD,
    ChEMBLNoveltyChecker,
    NOVEL_LIKE_THRESHOLD,
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
