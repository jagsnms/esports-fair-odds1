"""
Canonical round time normalization: raw (ms or s) -> seconds with validation.
Use normalized seconds everywhere downstream.
"""
from __future__ import annotations

from typing import Any


def normalize_round_time(val: Any) -> dict[str, Any]:
    """
    Normalize a round time value (ms or seconds) to canonical seconds.

    Returns dict with:
      - raw: original value (preserved)
      - seconds: float | None (normalized seconds; None if invalid or input None)
      - is_ms: bool (True if input was treated as ms, i.e. abs(val) >= 2000)
      - invalid_reason: str | None ("out_of_range" when seconds < 0 or > 200, else None)

    Rules:
      - if val is None -> seconds None, invalid_reason None
      - if abs(val) >= 2000 -> treat as ms, divide by 1000
      - if resulting seconds < 0 or seconds > 200 -> seconds None, invalid_reason="out_of_range"
      - else valid
    """
    out: dict[str, Any] = {
        "raw": val,
        "seconds": None,
        "is_ms": False,
        "invalid_reason": None,
        # Extra flags for ingest-time diagnostics
        "was_negative": False,
    }
    if val is None:
        return out
    try:
        x = float(val)
    except (TypeError, ValueError):
        out["invalid_reason"] = "not_numeric"
        return out
    is_ms = abs(x) >= 2000
    if is_ms:
        x = x / 1000.0
    was_negative = x < 0
    if x < 0 or x > 200:
        out["seconds"] = None
        out["invalid_reason"] = "out_of_range"
        out["is_ms"] = is_ms
        out["was_negative"] = was_negative
        return out
    out["seconds"] = x
    out["is_ms"] = is_ms
    return out
