from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from engine.compute.midround_v2_cs2 import (
    apply_cs2_midround_adjustment_v2_mixture,
    compute_cs2_midround_features,
)
from engine.models import Config, Frame

CAPTURE_SCHEMA_VERSION = "backend_bo3_live_capture_contract.v1"


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def load_capture_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        data = json.loads(raw)
        data["_source_lineno"] = lineno
        rows.append(data)
    return rows


def get_row_readiness_exclusion_reason(row: dict[str, Any]) -> str | None:
    if row.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        return "schema_version_mismatch"
    if row.get("live_source") != "BO3":
        return "non_bo3_live_source"
    if row.get("bo3_snapshot_status") != "live":
        return "snapshot_not_live"
    if row.get("bo3_health") != "GOOD":
        return "health_not_good"
    if row.get("round_phase") != "IN_PROGRESS":
        return "not_in_progress_phase"
    if row.get("clamp_reason") != "ok":
        return "non_ok_clamp_reason"
    if row.get("q_intra_total") is None:
        return "missing_current_q_intra"
    if row.get("a_side") not in ("T", "CT"):
        return "missing_a_side"
    if bool(row.get("round_time_remaining_was_missing")):
        return "timer_missing"
    if bool(row.get("round_time_remaining_was_out_of_range")):
        return "timer_out_of_range"
    required_numeric = (
        "p_hat",
        "rail_low",
        "rail_high",
        "alive_count_a",
        "alive_count_b",
        "hp_alive_total_a",
        "hp_alive_total_b",
        "loadout_est_total_a",
        "loadout_est_total_b",
        "round_time_remaining_s",
    )
    for field in required_numeric:
        if float_or_none(row.get(field)) is None:
            return f"missing_{field}"
    rail_low = float_or_none(row.get("rail_low"))
    rail_high = float_or_none(row.get("rail_high"))
    p_hat = float_or_none(row.get("p_hat"))
    if rail_low is None or rail_high is None or p_hat is None or rail_high < rail_low:
        return "invalid_rail_bounds"
    if not (rail_low <= p_hat <= rail_high):
        return "p_hat_outside_rails"
    return None


def _build_frame_from_row(row: dict[str, Any]) -> Frame:
    round_phase = row.get("round_phase")
    bomb_planted = bool(row.get("bomb_planted"))
    return Frame(
        timestamp=0.0,
        teams=("", ""),
        scores=(
            int_or_none(row.get("round_score_a")) or 0,
            int_or_none(row.get("round_score_b")) or 0,
        ),
        alive_counts=(
            int_or_none(row.get("alive_count_a")) or 0,
            int_or_none(row.get("alive_count_b")) or 0,
        ),
        hp_totals=(
            float_or_none(row.get("hp_alive_total_a")) or 0.0,
            float_or_none(row.get("hp_alive_total_b")) or 0.0,
        ),
        cash_totals=(
            float_or_none(row.get("cash_total_a")),
            float_or_none(row.get("cash_total_b")),
        ),
        loadout_totals=(
            float_or_none(row.get("loadout_est_total_a")),
            float_or_none(row.get("loadout_est_total_b")),
        ),
        armor_totals=(
            float_or_none(row.get("armor_alive_total_a")),
            float_or_none(row.get("armor_alive_total_b")),
        ),
        loadout_source=row.get("loadout_source"),
        bomb_phase_time_remaining={
            "is_bomb_planted": bomb_planted,
            "round_phase": round_phase,
            "round_time_remaining": float_or_none(row.get("round_time_remaining_s")),
        },
        round_time_remaining_s=float_or_none(row.get("round_time_remaining_s")),
        round_time_remaining_was_ms=bool(row.get("round_time_remaining_was_ms")),
        round_time_remaining_was_out_of_range=bool(row.get("round_time_remaining_was_out_of_range")),
        round_time_remaining_was_missing=bool(row.get("round_time_remaining_was_missing")),
        map_index=int_or_none(row.get("map_index")) or 0,
        series_score=(
            int_or_none(row.get("series_score_a")) or 0,
            int_or_none(row.get("series_score_b")) or 0,
        ),
        map_name=str(row.get("map_name") or ""),
        series_fmt=str(row.get("series_fmt") or ""),
        a_side=row.get("a_side"),
        team_one_id=int_or_none(row.get("team_one_id")),
        team_two_id=int_or_none(row.get("team_two_id")),
        team_one_provider_id=None if row.get("team_one_provider_id") is None else str(row.get("team_one_provider_id")),
        team_two_provider_id=None if row.get("team_two_provider_id") is None else str(row.get("team_two_provider_id")),
    )


def signal_active(row: dict[str, Any]) -> bool:
    alive_delta = abs((int_or_none(row.get("alive_count_a")) or 0) - (int_or_none(row.get("alive_count_b")) or 0))
    hp_delta = abs((float_or_none(row.get("hp_alive_total_a")) or 0.0) - (float_or_none(row.get("hp_alive_total_b")) or 0.0))
    loadout_delta = abs(
        (float_or_none(row.get("loadout_est_total_a")) or 0.0) - (float_or_none(row.get("loadout_est_total_b")) or 0.0)
    )
    bomb_planted = bool(row.get("bomb_planted"))
    return bool(bomb_planted or alive_delta >= 1 or hp_delta >= 25.0 or loadout_delta >= 1000.0)


def classify_mismatch(compared_row: dict[str, Any]) -> str:
    abs_q_delta = float(compared_row["abs_q_delta"])
    current_q = float(compared_row["current_q"])
    v2_q = float(compared_row["v2_q"])
    alive_delta = abs(float(compared_row["alive_delta"]))
    hp_delta = abs(float(compared_row["hp_delta"]))
    loadout_delta = abs(float(compared_row["loadout_delta"]))
    sign_flip = (current_q - 0.5) * (v2_q - 0.5) < 0.0

    if sign_flip and abs_q_delta >= 0.15:
        return "sign_flip_large_gap"
    if abs_q_delta >= 0.15 and loadout_delta >= 1500.0 and alive_delta == 0.0 and hp_delta < 50.0:
        return "loadout_sensitive_large_gap"
    if abs_q_delta >= 0.15:
        return "large_gap"
    if abs_q_delta >= 0.08:
        return "moderate_gap"
    return "close"


def compare_row_to_v2(row: dict[str, Any]) -> dict[str, Any]:
    frame = _build_frame_from_row(row)
    config = Config(
        source="BO3",
        team_a_is_team_one=bool(row.get("team_a_is_team_one")),
    )
    features = compute_cs2_midround_features(frame, config=config)
    v2 = apply_cs2_midround_adjustment_v2_mixture(
        frozen_a=float(row["rail_high"]),
        frozen_b=float(row["rail_low"]),
        features=features,
        config=config,
        frame=frame,
    )
    current_q = float(row["q_intra_total"])
    current_p_hat = float(row["p_hat"])
    v2_q = float(v2["q_intra"])
    v2_p_hat = float(v2["p_mid_clamped"])
    compared = {
        "match_id": int(row["match_id"]),
        "map_index": int(row["map_index"]),
        "game_number": int(row["game_number"]),
        "round_number": int(row["round_number"]),
        "round_phase": row["round_phase"],
        "capture_ts_iso": row["capture_ts_iso"],
        "raw_provider_event_id": row["raw_provider_event_id"],
        "current_q": current_q,
        "v2_q": v2_q,
        "q_delta": v2_q - current_q,
        "abs_q_delta": abs(v2_q - current_q),
        "current_p_hat": current_p_hat,
        "v2_p_hat": v2_p_hat,
        "p_hat_delta": v2_p_hat - current_p_hat,
        "abs_p_hat_delta": abs(v2_p_hat - current_p_hat),
        "alive_delta": float((row["alive_count_a"] or 0) - (row["alive_count_b"] or 0)),
        "hp_delta": float((row["hp_alive_total_a"] or 0.0) - (row["hp_alive_total_b"] or 0.0)),
        "loadout_delta": float((row["loadout_est_total_a"] or 0.0) - (row["loadout_est_total_b"] or 0.0)),
        "bomb_planted": bool(row["bomb_planted"]),
        "a_side": row["a_side"],
        "signal_active": signal_active(row),
        "timer_direction_reason_code": v2.get("timer_direction_reason_code"),
        "hard_boundary_reason_code": v2.get("hard_boundary_reason_code"),
        "weight_profile": v2.get("weight_profile"),
    }
    compared["mismatch_class"] = classify_mismatch(compared)
    return compared


def collapse_distinct_raw_events(compared_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in compared_rows:
        raw_provider_event_id = row.get("raw_provider_event_id")
        key = f"{row.get('match_id')}::{raw_provider_event_id}" if raw_provider_event_id else (
            f"fallback:{row.get('match_id')}:{row.get('game_number')}:{row.get('round_number')}:{row.get('capture_ts_iso')}"
        )
        grouped.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for rows in grouped.values():
        representative = dict(rows[0])
        representative["duplicate_tick_count"] = len(rows)
        for metric in (
            "current_q",
            "v2_q",
            "q_delta",
            "abs_q_delta",
            "current_p_hat",
            "v2_p_hat",
            "p_hat_delta",
            "abs_p_hat_delta",
        ):
            representative[metric] = float(statistics.median(float(row[metric]) for row in rows))
        representative["signal_active"] = any(bool(row["signal_active"]) for row in rows)
        representative["mismatch_class"] = classify_mismatch(representative)
        collapsed.append(representative)
    return collapsed

