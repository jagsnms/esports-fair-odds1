#!/usr/bin/env python3
"""
Replay anchor harness: run normalize -> reduce -> bounds -> rails -> resolve_p_hat (midround phase-gated)
over logs/bo3_pulls.jsonl for a given match_id. Print a compact line per tick and optional detailed
blocks for anchor tick indices. No external deps; does not modify engine behavior.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Project root: script lives in scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.compute.bounds import compute_bounds
from engine.compute.midround_v2_cs2 import compute_cs2_midround_features
from engine.compute.rails import compute_rails
from engine.compute.resolve import resolve_p_hat
from engine.models import Config, Frame, State
from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
from engine.replay.bo3_jsonl import load_bo3_jsonl_entries
from engine.state.reducer import reduce_state


def _coerce_match_id(obj: dict) -> int | None:
    mid = obj.get("match_id")
    if mid is None:
        return None
    try:
        return int(mid)
    except (TypeError, ValueError):
        return None


def _entry_t_unix(entry: dict) -> int | None:
    """Parse ts_utc from JSONL entry to integer unix seconds. Handles ISO string or numeric."""
    ts = entry.get("ts_utc")
    if ts is None:
        return None
    if isinstance(ts, (int, float)) and ts >= 0:
        return int(ts)
    if isinstance(ts, str):
        try:
            s = ts.replace("Z", "+00:00")
            return int(datetime.fromisoformat(s).timestamp())
        except (ValueError, TypeError):
            return None
    return None


def _collect_payloads_and_team_a(entries: list[dict], match_id: int) -> list[tuple[dict, bool, int | None]]:
    """Return list of (payload, team_a_is_team_one, t_unix). t_unix is None if ts_utc missing or unparseable."""
    out: list[tuple[dict, bool, int | None]] = []
    for e in entries:
        if _coerce_match_id(e) != match_id:
            continue
        payload = e.get("payload")
        if not isinstance(payload, dict):
            continue
        team_a = e.get("team_a_is_team_one", True)
        if isinstance(team_a, bool):
            pass
        else:
            team_a = bool(team_a) if team_a is not None else True
        t_unix = _entry_t_unix(e)
        out.append((payload, team_a, t_unix))
    return out


def run(
    path: str,
    match_id: int,
    max_ticks: int = 200,
    anchors: list[int] | None = None,
    anchor_times: list[int] | None = None,
    time_tolerance: int = 5,
    *,
    find_close_late: bool = False,
    min_score_sum: int = 18,
    max_abs_diff: int = 2,
    n: int = 6,
) -> None:
    entries = load_bo3_jsonl_entries(path)
    payloads_with_team_a = _collect_payloads_and_team_a(entries, match_id)
    if not payloads_with_team_a:
        print(f"no payloads for match_id={match_id} in {path}", file=sys.stderr)
        return
    team_a_is_team_one = payloads_with_team_a[0][1]

    # Build index: (tick_idx, t) for ticks that have a timestamp (for anchor_times resolution).
    tick_index_list: list[tuple[int, int]] = [
        (tick_idx, t)
        for tick_idx, (_, _, t) in enumerate(payloads_with_team_a)
        if t is not None
    ]
    # Resolve anchor_times to tick indices; track (requested_t, matched_tick_idx, matched_t, delta_seconds).
    time_anchor_info: dict[int, tuple[int, int, int | float]] = {}  # tick_idx -> (requested_t, matched_t, delta_seconds)
    anchor_set = set(anchors or [])
    if anchor_times and not tick_index_list:
        print("WARNING: --anchor_times given but no ts_utc in JSONL entries; skipping time-based anchors", file=sys.stderr)
    if anchor_times and tick_index_list:
        for requested_t in anchor_times:
            best_tick_idx, best_t = min(
                tick_index_list,
                key=lambda item: abs(item[1] - requested_t),
            )
            delta = abs(best_t - requested_t)
            anchor_set.add(best_tick_idx)
            time_anchor_info[best_tick_idx] = (requested_t, best_t, delta)
    config = Config(
        team_a_is_team_one=team_a_is_team_one,
        contract_scope="map",
    )
    state = State(
        config=config,
        last_frame=None,
        map_index=0,
        last_total_rounds=0,
        segment_id=0,
        last_series_score=None,
        last_map_index=None,
    )
    auto_candidates: list[int] = []

    for tick_idx in range(min(max_ticks, len(payloads_with_team_a))):
        payload, _, _ = payloads_with_team_a[tick_idx]
        frame = bo3_snapshot_to_frame(payload, team_a_is_team_one=team_a_is_team_one)
        state = reduce_state(state, frame, config)
        bounds_result = compute_bounds(frame, config, state)
        bound_lo, bound_hi = bounds_result[0], bounds_result[1]
        bounds = (bound_lo, bound_hi)
        rails_result = compute_rails(frame, config, state, bounds)
        rail_lo, rail_hi = rails_result[0], rails_result[1]
        rails_debug = rails_result[2] if len(rails_result) > 2 else {}
        rails = (rail_lo, rail_hi)
        p_hat, dbg = resolve_p_hat(frame, config, state, rails)
        p_hat_old = dbg.get("p_hat_old", p_hat)
        inter_break = bool(dbg.get("inter_map_break", False))
        inter_break_reason = dbg.get("inter_map_break_reason")
        seg = state.segment_id
        scores = getattr(frame, "scores", (0, 0))
        ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
        rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
        score_sum = ra + rb
        score_diff = ra - rb
        # Compact line
        print(
            f"tick={tick_idx} seg={seg} scores={scores[0]}-{scores[1]} "
            f"rails=({rail_lo:.3f},{rail_hi:.3f}) p_hat_old={p_hat_old:.4f} p_hat_final={p_hat:.4f} "
            f"inter_map_break={inter_break}"
        )
        # Collect close/late candidates if requested
        if find_close_late and score_sum >= min_score_sum and abs(score_diff) <= max_abs_diff:
            auto_candidates.append(tick_idx)
        if tick_idx in anchor_set:
            if tick_idx in time_anchor_info:
                req_t, matched_t, delta_sec = time_anchor_info[tick_idx]
                print(
                    f"  requested_time={req_t} matched_tick_idx={tick_idx} matched_t={matched_t} delta_seconds={delta_sec}"
                )
                if delta_sec > time_tolerance:
                    print(
                        f"  WARNING: delta_seconds ({delta_sec}) > time_tolerance ({time_tolerance})",
                        file=sys.stderr,
                    )
            features = compute_cs2_midround_features(frame, config=config)
            round_phase = features.get("round_phase")
            t_remaining = features.get("time_remaining_s")
            bomb = features.get("bomb_planted", 0)
            a_side = features.get("a_side")
            alive = getattr(frame, "alive_counts", (0, 0))
            hp = getattr(frame, "hp_totals", (0.0, 0.0))
            loadout = getattr(frame, "loadout_totals") or (0.0, 0.0)
            series_width = bound_hi - bound_lo
            # Map corridor debug from rails_debug (context_widening may be off -> some keys missing)
            map_width = rail_hi - rail_lo
            map_width_raw_before = rails_debug.get("map_width_raw_before_cap") or rails_debug.get("map_width_before") or map_width
            map_width_after_widen = rails_debug.get("map_width_after_widen") or rails_debug.get("map_width_after") or map_width
            map_width_after_cap = rails_debug.get("map_width_after_cap") or map_width
            width_cap_used = rails_debug.get("width_cap_used")
            context_risk = rails_debug.get("context_risk")
            uncertainty_mult = rails_debug.get("uncertainty_multiplier")
            v2 = dbg.get("midround_v2") or {}
            q_intra = v2.get("q_intra")
            raw_score = v2.get("raw_score")
            urgency = v2.get("urgency")
            p_mid_clamped = v2.get("p_mid_clamped")
            used_bomb = v2.get("used_bomb_direction")
            used_time = v2.get("used_time")
            print("  --- anchor ---")
            print(
                f"  tick_idx={tick_idx} seg={seg} round_phase={round_phase!r} "
                f"t_remaining={t_remaining} bomb={bool(bomb)} a_side={a_side!r}"
            )
            print(
                f"  alive_counts={alive} hp_totals=({hp[0]:.0f},{hp[1]:.0f}) "
                f"loadout_totals=({loadout[0]:.0f},{loadout[1]:.0f})"
            )
            print(
                f"  series corridor: series_low={bound_lo:.4f} series_high={bound_hi:.4f} "
                f"series_width={series_width:.4f}"
            )
            print(
                f"  map corridor: map_low={rail_lo:.4f} map_high={rail_hi:.4f} "
                f"map_width={map_width:.4f}"
            )
            print(
                f"  map_width_raw_before_cap={map_width_raw_before!s} "
                f"map_width_after_widen={map_width_after_widen!s} "
                f"map_width_after_cap={map_width_after_cap!s} width_cap_used={width_cap_used!s}"
            )
            print(
                f"  context_risk={context_risk!s} uncertainty_multiplier={uncertainty_mult!s}"
            )
            print(f"  p_hat_old={p_hat_old:.4f} p_hat_final={p_hat:.4f}")
            print(
                f"  midround_v2: q_intra={q_intra} raw_score={raw_score} urgency={urgency} "
                f"p_mid_clamped={p_mid_clamped} used_bomb_direction={used_bomb} used_time={used_time}"
            )
            if inter_break:
                print(f"  inter_map_break=True reason={inter_break_reason}")
            print("  ---")

    # If anchors were not provided but auto-pick mode is enabled, choose and print them now.
    if find_close_late and not anchors and auto_candidates:
        total = len(auto_candidates)
        if total <= n:
            chosen = auto_candidates
        else:
            # Evenly spaced sample across candidates
            step = max(1, total // n)
            chosen = [auto_candidates[i] for i in range(0, total, step)][:n]
        print(f"auto_chosen_anchors={chosen}")
        # Re-run only the anchor-detail printing for chosen ticks
        anchor_set = set(chosen)
        for tick_idx in chosen:
            if tick_idx >= len(payloads_with_team_a):
                continue
            payload, _, _ = payloads_with_team_a[tick_idx]
            frame = bo3_snapshot_to_frame(payload, team_a_is_team_one=team_a_is_team_one)
            # We need a consistent state trail up to this tick; reuse the same forward pass
            # by re-running reduce_state/compute up to tick_idx.
            # Simple approach: recompute from scratch up to each chosen tick.
            cfg = Config(
                team_a_is_team_one=team_a_is_team_one,
                contract_scope="map",
            )
            st = State(
                config=cfg,
                last_frame=None,
                map_index=0,
                last_total_rounds=0,
                segment_id=0,
                last_series_score=None,
                last_map_index=None,
            )
            for i in range(tick_idx + 1):
                p_i, _, _ = payloads_with_team_a[i]
                f_i = bo3_snapshot_to_frame(p_i, team_a_is_team_one=team_a_is_team_one)
                st = reduce_state(st, f_i, cfg)
            # Now compute bounds/rails/resolve for this tick using st and frame
            bounds_result = compute_bounds(frame, cfg, st)
            bound_lo, bound_hi = bounds_result[0], bounds_result[1]
            bounds = (bound_lo, bound_hi)
            rails_result = compute_rails(frame, cfg, st, bounds)
            rail_lo, rail_hi = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            rails = (rail_lo, rail_hi)
            p_hat, dbg = resolve_p_hat(frame, cfg, st, rails)
            p_hat_old = dbg.get("p_hat_old", p_hat)
            seg = st.segment_id
            scores = getattr(frame, "scores", (0, 0))
            features = compute_cs2_midround_features(frame, config=cfg)
            round_phase = features.get("round_phase")
            t_remaining = features.get("time_remaining_s")
            bomb = features.get("bomb_planted", 0)
            a_side = features.get("a_side")
            alive = getattr(frame, "alive_counts", (0, 0))
            hp = getattr(frame, "hp_totals", (0.0, 0.0))
            loadout = getattr(frame, "loadout_totals") or (0.0, 0.0)
            series_width = bound_hi - bound_lo
            map_width = rail_hi - rail_lo
            map_width_raw_before = rails_debug.get("map_width_raw_before_cap") or rails_debug.get("map_width_before") or map_width
            map_width_after_widen = rails_debug.get("map_width_after_widen") or rails_debug.get("map_width_after") or map_width
            map_width_after_cap = rails_debug.get("map_width_after_cap") or map_width
            width_cap_used = rails_debug.get("width_cap_used")
            context_risk = rails_debug.get("context_risk")
            uncertainty_mult = rails_debug.get("uncertainty_multiplier")
            v2 = dbg.get("midround_v2") or {}
            q_intra = v2.get("q_intra")
            raw_score = v2.get("raw_score")
            urgency = v2.get("urgency")
            p_mid_clamped = v2.get("p_mid_clamped")
            used_bomb = v2.get("used_bomb_direction")
            used_time = v2.get("used_time")
            print("  --- anchor ---")
            print(
                f"  tick_idx={tick_idx} seg={seg} round_phase={round_phase!r} "
                f"t_remaining={t_remaining} bomb={bool(bomb)} a_side={a_side!r}"
            )
            print(
                f"  alive_counts={alive} hp_totals=({hp[0]:.0f},{hp[1]:.0f}) "
                f"loadout_totals=({loadout[0]:.0f},{loadout[1]:.0f})"
            )
            print(
                f"  series corridor: series_low={bound_lo:.4f} series_high={bound_hi:.4f} "
                f"series_width={series_width:.4f}"
            )
            print(
                f"  map corridor: map_low={rail_lo:.4f} map_high={rail_hi:.4f} "
                f"map_width={map_width:.4f}"
            )
            print(
                f"  map_width_raw_before_cap={map_width_raw_before!s} "
                f"map_width_after_widen={map_width_after_widen!s} "
                f"map_width_after_cap={map_width_after_cap!s} width_cap_used={width_cap_used!s}"
            )
            print(
                f"  context_risk={context_risk!s} uncertainty_multiplier={uncertainty_mult!s}"
            )
            print(f"  p_hat_old={p_hat_old:.4f} p_hat_final={p_hat:.4f}")
            print(
                f"  midround_v2: q_intra={q_intra} raw_score={raw_score} urgency={urgency} "
                f"p_mid_clamped={p_mid_clamped} used_bomb_direction={used_bomb} used_time={used_time}"
            )
            print("  ---")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay anchor dump: run pipeline over BO3 JSONL, print compact + anchor blocks.")
    ap.add_argument("--path", default="logs/bo3_pulls.jsonl", help="JSONL path")
    ap.add_argument("--match_id", type=int, required=True, help="Match ID to filter")
    ap.add_argument("--max_ticks", type=int, default=200, help="Max ticks to run")
    ap.add_argument("--anchors", type=str, default="", help="Comma-separated tick indices for detailed blocks (e.g. 0,5,10)")
    ap.add_argument(
        "--anchor_times",
        type=str,
        default="",
        help="Comma-separated integer unix seconds to anchor by timestamp (e.g. 1772106330,1772106340)",
    )
    ap.add_argument(
        "--time_tolerance",
        type=int,
        default=5,
        help="Max seconds delta when matching --anchor_times to tick (default 5); warn if exceeded",
    )
    ap.add_argument("--find_close_late", action="store_true", help="Auto-pick anchors from close, late-round states")
    ap.add_argument("--min_score_sum", type=int, default=18, help="Minimum score_sum (ra+rb) for auto anchors")
    ap.add_argument("--max_abs_diff", type=int, default=2, help="Maximum |ra-rb| for auto anchors")
    ap.add_argument("--n", type=int, default=6, help="Maximum number of auto anchors to select")
    args = ap.parse_args()
    anchors_list: list[int] = []
    if args.anchors.strip():
        for s in args.anchors.split(","):
            s = s.strip()
            if s:
                try:
                    anchors_list.append(int(s))
                except ValueError:
                    pass
    anchor_times_list: list[int] = []
    if args.anchor_times.strip():
        for s in args.anchor_times.split(","):
            s = s.strip()
            if s:
                try:
                    anchor_times_list.append(int(s))
                except ValueError:
                    pass
    run(
        path=args.path,
        match_id=args.match_id,
        max_ticks=args.max_ticks,
        anchors=anchors_list or None,
        anchor_times=anchor_times_list or None,
        time_tolerance=args.time_tolerance,
        find_close_late=args.find_close_late,
        min_score_sum=args.min_score_sum,
        max_abs_diff=args.max_abs_diff,
        n=args.n,
    )


if __name__ == "__main__":
    main()
