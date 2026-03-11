from __future__ import annotations

import argparse
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

SCHEMA_VERSION = "backend_bo3_live_parity_diagnostic.v1"
CAPTURE_SCHEMA_VERSION = "backend_bo3_live_capture_contract.v1"
DECISION_MATERIALLY_WRONG = "materially_wrong"
DECISION_MATERIALLY_CLOSE = "materially_close"
DECISION_INCONCLUSIVE = "inconclusive"
ALLOWED_DECISIONS = {
    DECISION_MATERIALLY_WRONG,
    DECISION_MATERIALLY_CLOSE,
    DECISION_INCONCLUSIVE,
}

DEFAULT_CAPTURE_PATH = Path("automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl")
DEFAULT_REPORT_PATH = Path("automation/reports/backend_bo3_live_parity_diagnostic_report.json")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _counter_to_dict(counter: Counter) -> dict[str, int]:
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


def _choose_primary_match_id(rows: list[dict[str, Any]]) -> tuple[int | None, Counter]:
    counts: Counter = Counter()
    for row in rows:
        match_id = _int_or_none(row.get("match_id"))
        if match_id is not None:
            counts[match_id] += 1
    if not counts:
        return None, counts
    [(match_id, top_count)] = counts.most_common(1)
    tied = [candidate for candidate, count in counts.items() if count == top_count]
    if len(tied) != 1:
        return None, counts
    return match_id, counts


def _build_frame_from_row(row: dict[str, Any]) -> Frame:
    round_phase = row.get("round_phase")
    bomb_planted = bool(row.get("bomb_planted"))
    return Frame(
        timestamp=0.0,
        teams=("", ""),
        scores=(
            _int_or_none(row.get("round_score_a")) or 0,
            _int_or_none(row.get("round_score_b")) or 0,
        ),
        alive_counts=(
            _int_or_none(row.get("alive_count_a")) or 0,
            _int_or_none(row.get("alive_count_b")) or 0,
        ),
        hp_totals=(
            _float_or_none(row.get("hp_alive_total_a")) or 0.0,
            _float_or_none(row.get("hp_alive_total_b")) or 0.0,
        ),
        cash_totals=(
            _float_or_none(row.get("cash_total_a")),
            _float_or_none(row.get("cash_total_b")),
        ),
        loadout_totals=(
            _float_or_none(row.get("loadout_est_total_a")),
            _float_or_none(row.get("loadout_est_total_b")),
        ),
        armor_totals=(
            _float_or_none(row.get("armor_alive_total_a")),
            _float_or_none(row.get("armor_alive_total_b")),
        ),
        loadout_source=row.get("loadout_source"),
        bomb_phase_time_remaining={
            "is_bomb_planted": bomb_planted,
            "round_phase": round_phase,
            "round_time_remaining": _float_or_none(row.get("round_time_remaining_s")),
        },
        round_time_remaining_s=_float_or_none(row.get("round_time_remaining_s")),
        round_time_remaining_was_ms=bool(row.get("round_time_remaining_was_ms")),
        round_time_remaining_was_out_of_range=bool(row.get("round_time_remaining_was_out_of_range")),
        round_time_remaining_was_missing=bool(row.get("round_time_remaining_was_missing")),
        map_index=_int_or_none(row.get("map_index")) or 0,
        series_score=(
            _int_or_none(row.get("series_score_a")) or 0,
            _int_or_none(row.get("series_score_b")) or 0,
        ),
        map_name=str(row.get("map_name") or ""),
        series_fmt=str(row.get("series_fmt") or ""),
        a_side=row.get("a_side"),
        team_one_id=_int_or_none(row.get("team_one_id")),
        team_two_id=_int_or_none(row.get("team_two_id")),
        team_one_provider_id=None if row.get("team_one_provider_id") is None else str(row.get("team_one_provider_id")),
        team_two_provider_id=None if row.get("team_two_provider_id") is None else str(row.get("team_two_provider_id")),
    )


def get_row_exclusion_reason(row: dict[str, Any], *, primary_match_id: int | None) -> str | None:
    if row.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        return "schema_version_mismatch"
    if row.get("live_source") != "BO3":
        return "non_bo3_live_source"
    if primary_match_id is None:
        return "primary_match_unresolved"
    if _int_or_none(row.get("match_id")) != primary_match_id:
        return "non_primary_match"
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
        if _float_or_none(row.get(field)) is None:
            return f"missing_{field}"
    rail_low = _float_or_none(row.get("rail_low"))
    rail_high = _float_or_none(row.get("rail_high"))
    p_hat = _float_or_none(row.get("p_hat"))
    if rail_low is None or rail_high is None or p_hat is None or rail_high < rail_low:
        return "invalid_rail_bounds"
    if not (rail_low <= p_hat <= rail_high):
        return "p_hat_outside_rails"
    return None


def _signal_active(row: dict[str, Any]) -> bool:
    alive_delta = abs((_int_or_none(row.get("alive_count_a")) or 0) - (_int_or_none(row.get("alive_count_b")) or 0))
    hp_delta = abs((_float_or_none(row.get("hp_alive_total_a")) or 0.0) - (_float_or_none(row.get("hp_alive_total_b")) or 0.0))
    loadout_delta = abs(
        (_float_or_none(row.get("loadout_est_total_a")) or 0.0) - (_float_or_none(row.get("loadout_est_total_b")) or 0.0)
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
        "signal_active": _signal_active(row),
        "timer_direction_reason_code": v2.get("timer_direction_reason_code"),
        "hard_boundary_reason_code": v2.get("hard_boundary_reason_code"),
        "weight_profile": v2.get("weight_profile"),
    }
    compared["mismatch_class"] = classify_mismatch(compared)
    return compared


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * percentile
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(sorted_values[lower])
    frac = idx - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


def _summarize_deltas(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    values_sorted = sorted(values)
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "p90": _percentile(values_sorted, 0.90),
        "max": float(max(values)),
    }


def collapse_distinct_raw_events(compared_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in compared_rows:
        raw_provider_event_id = row.get("raw_provider_event_id")
        key = str(raw_provider_event_id) if raw_provider_event_id else (
            f"fallback:{row.get('match_id')}:{row.get('game_number')}:{row.get('round_number')}:{row.get('capture_ts_iso')}"
        )
        grouped.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        representative = dict(rows[0])
        representative["duplicate_tick_count"] = len(rows)
        for metric in ("current_q", "v2_q", "q_delta", "abs_q_delta", "current_p_hat", "v2_p_hat", "p_hat_delta", "abs_p_hat_delta"):
            representative[metric] = float(statistics.median(float(row[metric]) for row in rows))
        representative["signal_active"] = any(bool(row["signal_active"]) for row in rows)
        representative["mismatch_class"] = classify_mismatch(representative)
        collapsed.append(representative)
    return collapsed


def decide_report(distinct_event_rows: list[dict[str, Any]], per_tick_compared_rows: list[dict[str, Any]], excluded_counts: Counter) -> tuple[str, list[str]]:
    focus_rows = [row for row in distinct_event_rows if row["signal_active"]]
    distinct_count = len(distinct_event_rows)
    focus_count = len(focus_rows)
    if distinct_count < 30:
        return DECISION_INCONCLUSIVE, [
            f"only {distinct_count} distinct raw events were truthfully comparable",
            f"per-tick compared rows totaled {len(per_tick_compared_rows)}, but duplicate ticks are not independent evidence",
        ]
    if focus_count < 20:
        return DECISION_INCONCLUSIVE, [
            f"only {focus_count} distinct comparable raw events had active live-side signal",
            f"per-tick compared rows totaled {len(per_tick_compared_rows)}, but duplicate ticks are not independent evidence",
        ]

    abs_q_values = sorted(float(row["abs_q_delta"]) for row in focus_rows)
    median_abs_q = float(statistics.median(abs_q_values))
    p90_abs_q = float(_percentile(abs_q_values, 0.90) or 0.0)
    large_gap_count = sum(1 for row in focus_rows if row["mismatch_class"] in {"large_gap", "loadout_sensitive_large_gap", "sign_flip_large_gap"})
    sign_flip_count = sum(1 for row in focus_rows if row["mismatch_class"] == "sign_flip_large_gap")
    large_gap_ratio = large_gap_count / float(focus_count)

    if large_gap_ratio >= 0.35 and median_abs_q >= 0.10:
        return DECISION_MATERIALLY_WRONG, [
            f"{large_gap_count}/{focus_count} distinct active raw events showed large bounded V2 q divergence",
            f"median absolute q delta on distinct active raw events was {median_abs_q:.3f}",
            "the mismatch pattern is too strong to call this lane close on the captured match",
        ]
    if sign_flip_count == 0 and large_gap_ratio <= 0.10 and p90_abs_q <= 0.08:
        return DECISION_MATERIALLY_CLOSE, [
            f"large-gap distinct raw events stayed limited to {large_gap_count}/{focus_count}",
            f"90th percentile absolute q delta on distinct active raw events stayed at {p90_abs_q:.3f}",
            "the bounded V2 comparison target stayed close on most active distinct raw events",
        ]

    reasons = [
        f"{large_gap_count}/{focus_count} distinct active raw events showed large bounded V2 divergence",
        f"median absolute q delta on distinct active raw events was {median_abs_q:.3f}",
    ]
    if excluded_counts:
        reasons.append(f"explicit exclusions remained present: {dict(excluded_counts)}")
    reasons.append("results were mixed enough that a directional close/wrong call would overclaim")
    return DECISION_INCONCLUSIVE, reasons


def build_backend_bo3_live_parity_diagnostic_report(capture_path: Path) -> dict[str, Any]:
    rows = load_capture_rows(capture_path)
    primary_match_id, match_counts = _choose_primary_match_id(rows)
    excluded_counts: Counter = Counter()
    compared_rows: list[dict[str, Any]] = []

    for row in rows:
        exclusion_reason = get_row_exclusion_reason(row, primary_match_id=primary_match_id)
        if exclusion_reason is not None:
            excluded_counts[exclusion_reason] += 1
            continue
        compared_rows.append(compare_row_to_v2(row))

    distinct_event_rows = collapse_distinct_raw_events(compared_rows)
    distinct_active_rows = [row for row in distinct_event_rows if row["signal_active"]]
    mismatch_class_counts = Counter(row["mismatch_class"] for row in distinct_active_rows)
    decision, reasons = decide_report(distinct_event_rows, compared_rows, excluded_counts)
    top_mismatches = sorted(distinct_event_rows, key=lambda row: row["abs_q_delta"], reverse=True)[:10]

    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "decision_basis": "bounded_real_runtime_bo3_vs_v2_reference_distinct_raw_events",
        "reasons": reasons,
        "input_artifact": {
            "path": str(capture_path),
            "schema_version_expected": CAPTURE_SCHEMA_VERSION,
            "total_rows": len(rows),
            "match_id_counts": _counter_to_dict(match_counts),
            "primary_match_id": primary_match_id,
            "primary_match_row_count": int(match_counts.get(primary_match_id, 0)) if primary_match_id is not None else 0,
        },
        "compared_rows": {
            "eligible_compared_row_count": len(compared_rows),
            "signal_active_row_count": len([row for row in compared_rows if row["signal_active"]]),
            "distinct_raw_event_count": len(distinct_event_rows),
            "distinct_signal_active_raw_event_count": len(distinct_active_rows),
            "excluded_row_count": int(sum(excluded_counts.values())),
            "exclusion_reasons": _counter_to_dict(excluded_counts),
        },
        "summary_deltas": {
            "all_compared_per_tick": {
                "abs_q_delta": _summarize_deltas(compared_rows, "abs_q_delta"),
                "abs_p_hat_delta": _summarize_deltas(compared_rows, "abs_p_hat_delta"),
            },
            "distinct_raw_event_basis": {
                "abs_q_delta": _summarize_deltas(distinct_event_rows, "abs_q_delta"),
                "abs_p_hat_delta": _summarize_deltas(distinct_event_rows, "abs_p_hat_delta"),
            },
            "distinct_signal_active_raw_event_basis": {
                "abs_q_delta": _summarize_deltas(distinct_active_rows, "abs_q_delta"),
                "abs_p_hat_delta": _summarize_deltas(distinct_active_rows, "abs_p_hat_delta"),
            },
        },
        "mismatch_class_counts": _counter_to_dict(mismatch_class_counts),
        "top_mismatch_rows": top_mismatches,
        "truth_boundary": {
            "bounded_reference_target": "engine.compute.midround_v2_cs2.apply_cs2_midround_adjustment_v2_mixture",
            "decision_uses_distinct_raw_events": True,
            "per_tick_rows_remain_visible_but_not_independent_evidence": True,
            "not_live_parity_implementation": True,
            "not_broad_representativeness": True,
            "not_replay_live_decision_surface": True,
        },
    }
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"unexpected decision {decision!r}")
    return report


def write_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded BO3 live parity diagnostic on the versioned backend capture evidence snapshot.")
    parser.add_argument(
        "--capture-path",
        default=str(DEFAULT_CAPTURE_PATH),
        help="Path to the backend BO3 capture contract JSONL evidence snapshot.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write the machine-readable diagnostic report.",
    )
    args = parser.parse_args()

    capture_path = Path(args.capture_path)
    output_path = Path(args.output)
    report = build_backend_bo3_live_parity_diagnostic_report(capture_path)
    write_report(report, output_path)
    print(json.dumps(
        {
            "decision": report["decision"],
            "primary_match_id": report["input_artifact"]["primary_match_id"],
            "eligible_compared_row_count": report["compared_rows"]["eligible_compared_row_count"],
            "distinct_raw_event_count": report["compared_rows"]["distinct_raw_event_count"],
            "distinct_signal_active_raw_event_count": report["compared_rows"]["distinct_signal_active_raw_event_count"],
            "output_path": str(output_path),
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



