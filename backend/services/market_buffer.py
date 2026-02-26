"""
Market delay buffer: store (ts_epoch, bid, ask, mid, ticker) snapshots and return
the snapshot closest to (now - delay_sec) for aligned logging/plotting.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class MarketSnapshot:
    """Single market quote snapshot."""
    ts_epoch: float
    bid: float
    ask: float
    mid: float
    ticker: str


class MarketDelayBuffer:
    """
    Deque of market snapshots. push(snapshot); get_delayed(delay_sec) returns
    snapshot closest to (now - delay_sec), or latest if buffer empty.
    """

    def __init__(self, maxlen: int = 5000) -> None:
        self._deque: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def push(self, snapshot: dict[str, Any]) -> None:
        """Append a snapshot: must have ts_epoch, bid, ask, mid, ticker (or compatible)."""
        self._deque.append(dict(snapshot))

    def get_delayed(self, delay_sec: float) -> dict[str, Any] | None:
        """
        Return the snapshot whose ts_epoch is closest to (now - delay_sec).
        Prefer the latest snapshot at or before target_ts; if none, use closest by abs diff.
        If buffer is empty, return None (caller can fallback to latest elsewhere).
        """
        if not self._deque:
            return None
        target_ts = time.time() - max(0.0, float(delay_sec))
        best: dict[str, Any] | None = None
        best_ts: float | None = None
        # Prefer entry at or before target (most recent that is "delayed")
        for entry in reversed(self._deque):
            ts = entry.get("ts_epoch")
            if ts is None:
                continue
            if ts <= target_ts:
                if best is None or (best_ts is not None and ts > best_ts):
                    best = entry
                    best_ts = ts
        if best is not None:
            return best
        # Fallback: closest by absolute time difference
        for entry in self._deque:
            ts = entry.get("ts_epoch")
            if ts is None:
                continue
            if best is None or abs(ts - target_ts) < abs((best_ts or 0) - target_ts):
                best = entry
                best_ts = ts
        return best
