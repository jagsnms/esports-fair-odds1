"""
Async Runner: loop at poll_interval_s; BO3 live ingest when source=bo3 and match_id set, else dummy.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

from engine.models import Config, Derived, Frame, HistoryPoint, State

if TYPE_CHECKING:
    from backend.services.broadcaster import Broadcaster
    from backend.store.memory_store import MemoryStore


def _history_point_to_wire(p: HistoryPoint) -> dict:
    out = {"t": p.time, "p": p.p_hat, "lo": p.bound_low, "hi": p.bound_high, "m": p.market_mid}
    if hasattr(p, "segment_id"):
        out["seg"] = p.segment_id
    return out


class Runner:
    """Runs tick loop: BO3 snapshot or dummy data, append_point, broadcast."""

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

    async def _tick_bo3(self, config: Config) -> bool:
        """If source=BO3 and match_id set: fetch snapshot, normalize, append_point, broadcast. Return True if did BO3."""
        src = getattr(config, "source", None)
        match_id = getattr(config, "match_id", None)
        if src != "BO3" or match_id is None:
            return False
        mid = int(match_id)
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        try:
            from engine.ingest.bo3_client import get_snapshot
            from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
        except ImportError:
            return False
        try:
            snap = await get_snapshot(mid)
        except Exception as e:
            print(f"BO3 snapshot fetch failed for match_id={mid}: {e}")
            return True
        if not snap:
            print(f"BO3 snapshot fetch failed for match_id={mid}: returned None")
            return True
        frame = bo3_snapshot_to_frame(snap, team_a_is_team_one=team_a_is_team_one)
        t = time.time()
        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state

        old_state = await self._store.get_state()
        new_state = reduce_state(old_state, frame, config)
        bounds = compute_bounds(frame, config, new_state)
        rails = compute_rails(frame, config, new_state, bounds)
        p_hat = resolve_p_hat(frame, config, new_state, rails)
        bound_low, bound_high = bounds
        rail_low, rail_high = rails
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            market_mid=None,
            segment_id=new_state.segment_id,
        )
        derived = Derived(
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
        )
        await self._store.append_point(point, new_state, derived)
        await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(point)})
        return True

    async def _loop(self) -> None:
        """Tick: get config; BO3 or dummy, append_point, broadcast."""
        while not self._stop.is_set():
            try:
                config = await self._store.get_config()
                interval = max(0.1, getattr(config, "poll_interval_s", 1.0))
            except Exception:
                interval = 1.0
            did_bo3 = await self._tick_bo3(config)
            if not did_bo3:
                t = time.time()
                p_hat = 0.5 + 0.4 * math.sin(t * 0.1)
                p_hat = max(0.1, min(0.9, p_hat))
                lo = max(0.0, p_hat - 0.1)
                hi = min(1.0, p_hat + 0.1)
                market_mid = p_hat - 0.02 * math.sin(t * 0.15)
                market_mid = max(0.0, min(1.0, market_mid)) if market_mid is not None else None
                point = HistoryPoint(
                    time=t, p_hat=p_hat, bound_low=lo, bound_high=hi, market_mid=market_mid, segment_id=0
                )
                state = State(
                    config=config,
                    last_frame=Frame(timestamp=t, teams=("A", "B"), scores=(0, 0)),
                    map_index=0,
                    last_total_rounds=0,
                )
                derived = Derived(p_hat=p_hat, bound_low=lo, bound_high=hi, rail_low=lo, rail_high=hi, kappa=0.0)
                await self._store.append_point(point, state, derived)
                await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(point)})
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
        self._task = None
