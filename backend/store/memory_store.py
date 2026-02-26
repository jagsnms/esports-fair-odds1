"""
Thread-safe in-memory store for State, Derived, and history (ring buffer).
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from typing import Any

from engine.config import merge_config
from engine.models import Config, Derived, Frame, HistoryPoint, State


def _state_derived_to_dict(state: State, derived: Derived) -> dict[str, Any]:
    """Serialize State + Derived to JSON-serializable dict."""
    state_d = asdict(state)
    derived_d = asdict(derived)
    # Config/Frame are nested dataclasses; asdict recurses. Tuples become lists.
    return {"state": state_d, "derived": derived_d}


def _history_point_to_wire(p: HistoryPoint) -> dict[str, Any]:
    """Wire format: t (unix s), p, lo, hi, m."""
    return {
        "t": p.time,
        "p": p.p_hat,
        "lo": p.bound_low,
        "hi": p.bound_high,
        "m": p.market_mid,
    }


class MemoryStore:
    """Thread-safe in-memory store. Uses asyncio.Lock."""

    def __init__(self, max_history: int = 2000) -> None:
        self._lock = asyncio.Lock()
        self._state = State(config=Config(poll_interval_s=1.0))
        self._derived = Derived()
        self._history: deque[HistoryPoint] = deque(maxlen=max_history)

    async def get_current(self) -> dict[str, Any]:
        """Serialize current State + Derived."""
        async with self._lock:
            return _state_derived_to_dict(self._state, self._derived)

    async def get_history(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Last `limit` points in wire format."""
        async with self._lock:
            points = list(self._history)[-limit:]
            return [_history_point_to_wire(p) for p in points]

    async def append_point(
        self,
        point: HistoryPoint,
        state: State,
        derived: Derived,
    ) -> None:
        """Append one history point and update current state/derived."""
        async with self._lock:
            self._history.append(point)
            self._state = state
            self._derived = derived

    async def update_config(self, partial: dict[str, Any]) -> None:
        """Merge partial config update; does not touch state/derived/history."""
        async with self._lock:
            self._state = State(
                config=merge_config(self._state.config, partial),
                last_frame=self._state.last_frame,
                team_mapping=self._state.team_mapping,
                map_index=self._state.map_index,
                last_total_rounds=self._state.last_total_rounds,
            )

    async def get_config(self) -> Config:
        """Return current config."""
        async with self._lock:
            return self._state.config
