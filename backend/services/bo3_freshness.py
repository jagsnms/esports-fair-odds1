"""
Re-export BO3 freshness gate from engine for backward compatibility.
"""
from __future__ import annotations

from engine.ingest.bo3_freshness import (
    CLOCK_REWIND_EPS_S,
    Bo3FreshnessGate,
    coerce_ts_ms,
)

__all__ = ["Bo3FreshnessGate", "CLOCK_REWIND_EPS_S", "coerce_ts_ms"]
