"""
Async Runner: loop at poll_interval_s; BO3 live, REPLAY from JSONL, or dummy.
Market: poll Kalshi when enabled, push to delay buffer, attach market_mid to points.
"""
from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from engine.models import Config, Derived, Frame, HistoryPoint, State

if TYPE_CHECKING:
    from backend.services.broadcaster import Broadcaster
    from backend.store.memory_store import MemoryStore


def _history_point_to_wire(p: HistoryPoint) -> dict:
    out = {"t": p.time, "p": p.p_hat, "lo": p.bound_low, "hi": p.bound_high, "m": p.market_mid}
    if hasattr(p, "rail_low"):
        out["rail_low"] = p.rail_low
    if hasattr(p, "rail_high"):
        out["rail_high"] = p.rail_high
    if hasattr(p, "segment_id"):
        out["seg"] = p.segment_id
    return out


class Runner:
    """Runs tick loop: BO3 snapshot, REPLAY from JSONL, or dummy; append_point, broadcast."""

    def __init__(self, store: MemoryStore, broadcaster: Broadcaster) -> None:
        self._store = store
        self._broadcaster = broadcaster
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Replay cache (lazy-loaded when source=REPLAY)
        self._replay_payloads: list[dict[str, Any]] = []
        self._replay_index: int = 0
        self._replay_path: str | None = None
        self._replay_match_id: int | None = None
        self._dummy_snapshot_sent = False
        # Market delay buffer + poll throttle
        from backend.services.market_buffer import MarketDelayBuffer
        self._market_buffer = MarketDelayBuffer(maxlen=5000)
        self._last_market_poll_ts: float = 0.0
        self._last_market_error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="runner_market")

    def get_replay_progress(self) -> dict[str, int] | None:
        """Return {index, total} when replay is active and list is loaded."""
        if not self._replay_payloads:
            return None
        return {"index": self._replay_index, "total": len(self._replay_payloads)}

    async def _maybe_poll_market(self, config: Config) -> None:
        """If market_enabled and kalshi_ticker and poll interval elapsed, fetch and push to buffer."""
        if not getattr(config, "market_enabled", True):
            return
        ticker = getattr(config, "kalshi_ticker", None) or ""
        if not ticker or not ticker.strip():
            return
        poll_sec = max(1, int(getattr(config, "market_poll_sec", 5)))
        now = time.time()
        if now - self._last_market_poll_ts < poll_sec:
            return
        self._last_market_poll_ts = now
        ticker = ticker.strip()

        def _fetch() -> None:
            from engine.market.kalshi_client import fetch_kalshi_bid_ask
            bid, ask, mid, ts_epoch = fetch_kalshi_bid_ask(ticker)
            self._market_buffer.push({
                "ts_epoch": ts_epoch,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "ticker": ticker,
            })
            self._last_market_error = None

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, _fetch)
        except Exception as e:
            self._last_market_error = str(e)

    def _get_market_for_point(self, config: Config) -> tuple[float | None, dict[str, Any]]:
        """Return (market_mid, extra_debug). market_mid from delayed buffer; extra_debug has market_error if set."""
        out_debug: dict[str, Any] = {}
        if self._last_market_error:
            out_debug["market_error"] = self._last_market_error
        if not getattr(config, "market_enabled", True) or not getattr(config, "kalshi_ticker", None):
            return (None, out_debug)
        delay_sec = max(0, int(getattr(config, "market_delay_sec", 120)))
        snap = self._market_buffer.get_delayed(delay_sec)
        if snap is None:
            return (None, out_debug)
        mid = snap.get("mid")
        if mid is not None:
            out_debug["market_bid"] = snap.get("bid")
            out_debug["market_ask"] = snap.get("ask")
        return (float(mid) if mid is not None else None, out_debug)

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
        from engine.diagnostics.inter_map_break import detect_inter_map_break

        old_state = await self._store.get_state()
        new_state = reduce_state(old_state, frame, config)
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}

        # Detect inter-map break (between maps) and keep corridors/p_hat stable.
        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            series_width = bound_high - bound_low
            center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
            # Use a modest map width inside series; avoid collapsing to zero.
            map_width = min(series_width * 0.6, 0.30)
            half = 0.5 * map_width
            rail_low = max(bound_low, min(bound_high, center - half))
            rail_high = max(bound_low, min(bound_high, center + half))
            # Try to reuse last p_hat if available; else midpoint.
            cur = await self._store.get_current()
            d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
            last_p = d.get("p_hat")
            if isinstance(last_p, (int, float)):
                p_hat = float(last_p)
            else:
                p_hat = center
            p_hat = max(rail_low, min(rail_high, p_hat))
            dbg: dict[str, Any] = {
                "inter_map_break": True,
                "inter_map_break_reason": break_reason,
                "p_hat_old": last_p,
                "p_hat_final": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
            }
            dbg.update(bounds_debug)
        else:
            bounds = (bound_low, bound_high)
            rails_result = compute_rails(frame, config, new_state, bounds)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}

        market_mid, market_dbg = self._get_market_for_point(config)
        dbg.update(market_dbg)
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=new_state.segment_id,
        )
        derived = Derived(
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug=dbg,
        )
        await self._store.append_point(point, new_state, derived)
        await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(point)})
        return True

    async def _tick_replay(self, config: Config) -> bool:
        """If source=REPLAY: feed next JSONL payload to normalize+compute, append_point, broadcast. Return True if did replay."""
        src = getattr(config, "source", None)
        if src != "REPLAY":
            return False
        path = getattr(config, "replay_path", None) or "logs/bo3_pulls.jsonl"
        match_id = getattr(config, "match_id", None)
        if match_id is not None:
            try:
                match_id = int(match_id)
            except (TypeError, ValueError):
                match_id = None
        replay_loop = getattr(config, "replay_loop", True)
        replay_speed = max(0.1, float(getattr(config, "replay_speed", 1.0)))

        # Lazy-load and cache when path or match_id changes
        if path != self._replay_path or match_id != self._replay_match_id:
            try:
                from engine.replay.bo3_jsonl import load_bo3_jsonl_entries, iter_payloads
            except ImportError:
                return False
            entries = load_bo3_jsonl_entries(path)
            self._replay_payloads = [p for _, p in iter_payloads(entries, match_id)]
            self._replay_index = 0
            self._replay_path = path
            self._replay_match_id = match_id

        if not self._replay_payloads:
            return True  # no entries; sleep and keep trying

        if self._replay_index >= len(self._replay_payloads):
            # End of replay
            if replay_loop:
                # Bump segment so frontend sees loop boundary; then reset index
                state = await self._store.get_state()
                seg = getattr(state, "segment_id", 0) + 1
                from engine.models import State as StateModel
                loop_state = StateModel(
                    config=config,
                    last_frame=state.last_frame,
                    team_mapping=state.team_mapping,
                    map_index=getattr(state, "map_index", 0),
                    last_total_rounds=getattr(state, "last_total_rounds", 0),
                    segment_id=seg,
                    last_series_score=getattr(state, "last_series_score", None),
                    last_map_index=getattr(state, "last_map_index", None),
                )
                cur = await self._store.get_current()
                d = cur.get("derived") or {}
                derived_obj = Derived(
                    p_hat=d.get("p_hat", 0.5),
                    bound_low=d.get("bound_low", 0),
                    bound_high=d.get("bound_high", 1),
                    rail_low=d.get("rail_low", 0),
                    rail_high=d.get("rail_high", 1),
                    kappa=d.get("kappa", 0),
                    debug=d.get("debug", {}),
                )
                pt = HistoryPoint(
                    time=time.time(),
                    p_hat=derived_obj.p_hat,
                    bound_low=derived_obj.bound_low,
                    bound_high=derived_obj.bound_high,
                    rail_low=derived_obj.rail_low,
                    rail_high=derived_obj.rail_high,
                    market_mid=None,
                    segment_id=seg,
                )
                await self._store.append_point(pt, loop_state, derived_obj)
                await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(pt)})
                self._replay_index = 0
            return True

        payload = self._replay_payloads[self._replay_index]
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        # JSONL entry may have team_a_is_team_one
        try:
            from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
        except ImportError:
            return False
        frame = bo3_snapshot_to_frame(payload, team_a_is_team_one=team_a_is_team_one)
        t = time.time()
        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state
        from engine.diagnostics.inter_map_break import detect_inter_map_break

        old_state = await self._store.get_state()
        new_state = reduce_state(old_state, frame, config)
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}

        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            series_width = bound_high - bound_low
            center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
            map_width = min(series_width * 0.6, 0.30)
            half = 0.5 * map_width
            rail_low = max(bound_low, min(bound_high, center - half))
            rail_high = max(bound_low, min(bound_high, center + half))
            cur = await self._store.get_current()
            d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
            last_p = d.get("p_hat")
            if isinstance(last_p, (int, float)):
                p_hat = float(last_p)
            else:
                p_hat = center
            p_hat = max(rail_low, min(rail_high, p_hat))
            dbg: dict[str, Any] = {
                "inter_map_break": True,
                "inter_map_break_reason": break_reason,
                "p_hat_old": last_p,
                "p_hat_final": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
            }
            dbg.update(bounds_debug)
        else:
            bounds = (bound_low, bound_high)
            rails_result = compute_rails(frame, config, new_state, bounds)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}

        market_mid, market_dbg = self._get_market_for_point(config)
        dbg.update(market_dbg)
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=new_state.segment_id,
        )
        derived = Derived(
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug=dbg,
        )
        await self._store.append_point(point, new_state, derived)
        await self._broadcaster.broadcast({"type": "point", "point": _history_point_to_wire(point)})
        self._replay_index += 1
        return True

    async def _loop(self) -> None:
        """Tick: get config; REPLAY or BO3 or dummy, append_point, broadcast."""
        while not self._stop.is_set():
            try:
                config = await self._store.get_config()
                interval = max(0.1, getattr(config, "poll_interval_s", 1.0))
                if getattr(config, "source", None) != "DUMMY":
                    self._dummy_snapshot_sent = False
                await self._maybe_poll_market(config)
            except Exception:
                interval = 1.0
            did_replay = await self._tick_replay(config)
            if did_replay:
                speed = max(0.1, float(getattr(config, "replay_speed", 1.0)))
                sleep_interval = interval / speed
            else:
                sleep_interval = interval
                did_bo3 = await self._tick_bo3(config)
                if not did_bo3:
                    src = getattr(config, "source", None)
                    if src == "DUMMY":
                        # Idle: no history append, no point broadcasts. Optionally one snapshot on first entry.
                        if not self._dummy_snapshot_sent:
                            neutral_derived = Derived(
                                p_hat=0.5,
                                bound_low=0.01,
                                bound_high=0.99,
                                rail_low=0.01,
                                rail_high=0.99,
                                kappa=0.0,
                            )
                            neutral_state = State(
                                config=config,
                                last_frame=Frame(timestamp=time.time(), teams=("A", "B"), scores=(0, 0)),
                                map_index=0,
                                last_total_rounds=0,
                            )
                            await self._store.set_current(neutral_state, neutral_derived)
                            current = await self._store.get_current()
                            history = await self._store.get_history(limit=500)
                            await self._broadcaster.broadcast(
                                {"type": "snapshot", "current": current, "history": history}
                            )
                            self._dummy_snapshot_sent = True
                    else:
                        # Legacy fallback when not BO3/REPLAY and source != DUMMY (e.g. BO3 with no match_id)
                        t = time.time()
                        p_hat = 0.5 + 0.4 * math.sin(t * 0.1)
                        p_hat = max(0.1, min(0.9, p_hat))
                        lo = max(0.0, p_hat - 0.1)
                        hi = min(1.0, p_hat + 0.1)
                        market_mid = p_hat - 0.02 * math.sin(t * 0.15)
                        market_mid = max(0.0, min(1.0, market_mid)) if market_mid is not None else None
                        point = HistoryPoint(
                            time=t,
                            p_hat=p_hat,
                            bound_low=lo,
                            bound_high=hi,
                            rail_low=lo,
                            rail_high=hi,
                            market_mid=market_mid,
                            segment_id=0,
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
                await asyncio.sleep(sleep_interval)
            except asyncio.CancelledError:
                break
        self._task = None
