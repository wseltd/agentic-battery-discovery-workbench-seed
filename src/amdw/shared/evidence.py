"""Re-export EvidenceLevel from the canonical location.

Single import path for ``amdw`` consumers — avoids coupling tests
and downstream code to the ``discovery_workbench`` package layout.
"""

from __future__ import annotations

from discovery_workbench.evidence import EvidenceLevel

__all__ = ["EvidenceLevel"]
