"""
Async Runner: loop at poll_interval_s, dummy data, append_point, broadcast.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

from engine.models import Derived, Frame, HistoryPoint, State

if TYPE_CHECKING:
    from backend.services.broadcaster import Broadcaster
    from backend.store.memory_store import MemoryStore


def _history_point_to_wire(p: HistoryPoint) -> dict:
    return {"t": p.time, "p": p.p_hat, "lo": p.bound_low, "hi": p.bound_high, "m": p.market_mid}


class Runner:
    """Runs tick loop: dummy p_hat/bounds, HistoryPoint, store.append_point, broadcast."""

    def __init__(self, store: MemoryStore, broadcaster: Broadcaster) -> None:
        self._store = store
        self._broadcaster = broadcaster
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        """Tick: get config, compute dummy data, append_point, broadcast."""
        start_t = time.time()
        tick_count = 0
        while not self._stop.is_set():
            try:
                config = await self._store.get_config()
                interval = max(0.1, getattr(config, "poll_interval_s", 1.0))
            except Exception:
                interval = 1.0
            # Dummy: p_hat sine 0.1..0.9, bounds ±0.1, market_mid lagged, monotonic time
            t = time.time()
            p_hat = 0.5 + 0.4 * math.sin(t * 0.1)
            p_hat = max(0.1, min(0.9, p_hat))
            lo = max(0.0, p_hat - 0.1)
            hi = min(1.0, p_hat + 0.1)
            market_mid = p_hat - 0.02 * math.sin(t * 0.15)
            market_mid = None if market_mid is None else max(0.0, min(1.0, market_mid))
            point = HistoryPoint(time=t, p_hat=p_hat, bound_low=lo, bound_high=hi, market_mid=market_mid)
            state = State(
                config=config,
                last_frame=Frame(timestamp=t, teams=("A", "B"), scores=(0, 0)),
                map_index=0,
                last_total_rounds=0,
            )
            derived = Derived(p_hat=p_hat, bound_low=lo, bound_high=hi, rail_low=lo, rail_high=hi, kappa=0.0)
            await self._store.append_point(point, state, derived)
            await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(point)})
            tick_count += 1
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
        self._task = None
