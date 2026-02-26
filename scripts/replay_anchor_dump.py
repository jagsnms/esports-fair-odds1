#!/usr/bin/env python3
"""
Replay anchor harness: run normalize -> reduce -> bounds -> rails -> resolve_p_hat (midround_enabled=True)
over logs/bo3_pulls.jsonl for a given match_id. Print a compact line per tick and optional detailed
blocks for anchor tick indices. No external deps; does not modify engine behavior.
"""
from __future__ import annotations

import argparse
import sys
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


def _collect_payloads_and_team_a(entries: list[dict], match_id: int) -> list[tuple[dict, bool]]:
    out: list[tuple[dict, bool]] = []
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
        out.append((payload, team_a))
    return out


def run(
    path: str,
    match_id: int,
    max_ticks: int = 200,
    anchors: list[int] | None = None,
) -> None:
    entries = load_bo3_jsonl_entries(path)
    payloads_with_team_a = _collect_payloads_and_team_a(entries, match_id)
    if not payloads_with_team_a:
        print(f"no payloads for match_id={match_id} in {path}", file=sys.stderr)
        return
    team_a_is_team_one = payloads_with_team_a[0][1]
    config = Config(
        midround_enabled=True,
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
    anchor_set = set(anchors or [])

    for tick_idx in range(min(max_ticks, len(payloads_with_team_a))):
        payload, _ = payloads_with_team_a[tick_idx]
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
        seg = state.segment_id
        scores = getattr(frame, "scores", (0, 0))
        # Compact line
        print(
            f"tick={tick_idx} seg={seg} scores={scores[0]}-{scores[1]} "
            f"rails=({rail_lo:.3f},{rail_hi:.3f}) p_hat_old={p_hat_old:.4f} p_hat_final={p_hat:.4f}"
        )
        if tick_idx in anchor_set:
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
            print("  ---")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay anchor dump: run pipeline over BO3 JSONL, print compact + anchor blocks.")
    ap.add_argument("--path", default="logs/bo3_pulls.jsonl", help="JSONL path")
    ap.add_argument("--match_id", type=int, required=True, help="Match ID to filter")
    ap.add_argument("--max_ticks", type=int, default=200, help="Max ticks to run")
    ap.add_argument("--anchors", type=str, default="", help="Comma-separated tick indices for detailed blocks (e.g. 0,5,10)")
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
    run(path=args.path, match_id=args.match_id, max_ticks=args.max_ticks, anchors=anchors_list or None)


if __name__ == "__main__":
    main()
