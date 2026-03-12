from __future__ import annotations

import argparse
import sys
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.backend_bo3_capture_analysis_common import (
    CAPTURE_SCHEMA_VERSION,
    collapse_distinct_raw_events,
    compare_row_to_v2,
    counter_to_dict,
    int_or_none,
    load_capture_rows,
    get_row_readiness_exclusion_reason,
)

SCHEMA_VERSION = "backend_bo3_live_parity_diagnostic.v1"
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


def _choose_primary_match_id(rows: list[dict[str, Any]]) -> tuple[int | None, Counter]:
    counts: Counter = Counter()
    for row in rows:
        match_id = int_or_none(row.get("match_id"))
        if match_id is not None:
            counts[match_id] += 1
    if not counts:
        return None, counts
    [(match_id, top_count)] = counts.most_common(1)
    tied = [candidate for candidate, count in counts.items() if count == top_count]
    if len(tied) != 1:
        return None, counts
    return match_id, counts


def get_row_exclusion_reason(row: dict[str, Any], *, primary_match_id: int | None) -> str | None:
    if primary_match_id is None:
        return "primary_match_unresolved"
    if int_or_none(row.get("match_id")) != primary_match_id:
        return "non_primary_match"
    return get_row_readiness_exclusion_reason(row)


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
            "match_id_counts": counter_to_dict(match_counts),
            "primary_match_id": primary_match_id,
            "primary_match_row_count": int(match_counts.get(primary_match_id, 0)) if primary_match_id is not None else 0,
        },
        "compared_rows": {
            "eligible_compared_row_count": len(compared_rows),
            "signal_active_row_count": len([row for row in compared_rows if row["signal_active"]]),
            "distinct_raw_event_count": len(distinct_event_rows),
            "distinct_signal_active_raw_event_count": len(distinct_active_rows),
            "excluded_row_count": int(sum(excluded_counts.values())),
            "exclusion_reasons": counter_to_dict(excluded_counts),
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
        "mismatch_class_counts": counter_to_dict(mismatch_class_counts),
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
    parser = argparse.ArgumentParser(description="Run a bounded BO3 live parity diagnostic on a frozen backend capture snapshot cut from the persistent BO3 corpus.")
    parser.add_argument(
        "--capture-path",
        default=str(DEFAULT_CAPTURE_PATH),
        help="Path to the backend BO3 capture contract JSONL frozen snapshot.",
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

