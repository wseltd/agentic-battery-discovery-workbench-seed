"""Re-export EvidenceLevel from the canonical location.

The canonical definition lives in discovery_workbench.evidence (T002).
This module provides the workbench.shared.evidence import path used by
the molecules sub-package.
"""

from discovery_workbench.evidence import EvidenceLevel

__all__ = ["EvidenceLevel"]
