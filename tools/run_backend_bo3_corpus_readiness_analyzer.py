from __future__ import annotations

import argparse
import sys
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.bo3_capture_contract import default_bo3_backend_capture_path
from tools.backend_bo3_capture_analysis_common import (
    CAPTURE_SCHEMA_VERSION,
    collapse_distinct_raw_events,
    compare_row_to_v2,
    counter_to_dict,
    get_row_readiness_exclusion_reason,
    load_capture_rows,
)

SCHEMA_VERSION = "backend_bo3_corpus_readiness_report.v1"
DEFAULT_CAPTURE_PATH = Path(default_bo3_backend_capture_path())
DEFAULT_REPORT_PATH = Path("automation/reports/backend_bo3_corpus_readiness_report.json")
MATERIALLY_USABLE_DISTINCT_RAW_EVENT_THRESHOLD = 30
MATERIALLY_USABLE_SIGNAL_ACTIVE_THRESHOLD = 20

READINESS_NO_USABLE = "no_usable_evidence"
READINESS_WEAK = "weak_usable_evidence"
READINESS_MATERIAL = "materially_usable_evidence"

BLOCKAGE_NOT_EASING = "not_easing"
BLOCKAGE_EASING_CONCENTRATED = "easing_but_overly_concentrated"
BLOCKAGE_EASING_MULTI_MATCH = "easing_multi_match"


def classify_match_readiness(*, distinct_raw_event_count: int, distinct_signal_active_raw_event_count: int) -> str:
    if distinct_raw_event_count <= 0:
        return READINESS_NO_USABLE
    if (
        distinct_raw_event_count >= MATERIALLY_USABLE_DISTINCT_RAW_EVENT_THRESHOLD
        and distinct_signal_active_raw_event_count >= MATERIALLY_USABLE_SIGNAL_ACTIVE_THRESHOLD
    ):
        return READINESS_MATERIAL
    return READINESS_WEAK


def _build_match_summary(match_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded_counts: Counter = Counter()
    compared_rows: list[dict[str, Any]] = []

    for row in rows:
        exclusion_reason = get_row_readiness_exclusion_reason(row)
        if exclusion_reason is not None:
            excluded_counts[exclusion_reason] += 1
            continue
        compared_rows.append(compare_row_to_v2(row))

    distinct_event_rows = collapse_distinct_raw_events(compared_rows)
    distinct_signal_active_rows = [row for row in distinct_event_rows if row["signal_active"]]
    readiness_class = classify_match_readiness(
        distinct_raw_event_count=len(distinct_event_rows),
        distinct_signal_active_raw_event_count=len(distinct_signal_active_rows),
    )
    return {
        "match_id": match_id,
        "row_count": len(rows),
        "eligible_compared_row_count": len(compared_rows),
        "signal_active_row_count": len([row for row in compared_rows if row["signal_active"]]),
        "distinct_raw_event_count": len(distinct_event_rows),
        "distinct_signal_active_raw_event_count": len(distinct_signal_active_rows),
        "excluded_row_count": int(sum(excluded_counts.values())),
        "exclusion_reasons": counter_to_dict(excluded_counts),
        "readiness_contribution_class": readiness_class,
    }


def _build_top_blockers(match_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reason_totals: Counter = Counter()
    match_counts_by_reason: dict[str, dict[str, int]] = {}

    for summary in match_summaries:
        match_id = str(summary["match_id"])
        for reason, count in summary["exclusion_reasons"].items():
            reason_totals[reason] += int(count)
            match_counts_by_reason.setdefault(reason, {})[match_id] = int(count)

    blockers: list[dict[str, Any]] = []
    for reason, count in reason_totals.most_common(10):
        affected_matches = match_counts_by_reason.get(reason, {})
        blockers.append(
            {
                "reason": reason,
                "count": int(count),
                "affected_match_count": len(affected_matches),
                "match_counts": dict(sorted(affected_matches.items(), key=lambda item: int(item[1]), reverse=True)),
            }
        )
    return blockers


def _build_readiness_summary(match_summaries: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    total_matches = len(match_summaries)
    materially_usable_matches = [summary for summary in match_summaries if summary["readiness_contribution_class"] == READINESS_MATERIAL]
    weak_matches = [summary for summary in match_summaries if summary["readiness_contribution_class"] == READINESS_WEAK]
    no_usable_matches = [summary for summary in match_summaries if summary["readiness_contribution_class"] == READINESS_NO_USABLE]
    total_distinct_signal_active = sum(int(summary["distinct_signal_active_raw_event_count"]) for summary in match_summaries)
    top_signal_match = max(match_summaries, key=lambda summary: int(summary["distinct_signal_active_raw_event_count"]), default=None)
    top_signal_share = None
    if top_signal_match is not None and total_distinct_signal_active > 0:
        top_signal_share = round(
            float(top_signal_match["distinct_signal_active_raw_event_count"]) / float(total_distinct_signal_active),
            4,
        )

    reasons: list[str] = []
    if total_distinct_signal_active == 0:
        status = BLOCKAGE_NOT_EASING
        reasons.append("no distinct signal-active comparable raw events were found anywhere in the corpus")
    elif not materially_usable_matches:
        status = BLOCKAGE_NOT_EASING
        reasons.append(
            f"no match cleared the materially usable threshold of {MATERIALLY_USABLE_DISTINCT_RAW_EVENT_THRESHOLD} distinct comparable raw events and {MATERIALLY_USABLE_SIGNAL_ACTIVE_THRESHOLD} distinct signal-active raw events"
        )
    elif len(materially_usable_matches) == 1 and total_matches > 1:
        status = BLOCKAGE_EASING_CONCENTRATED
        reasons.append(f"only 1/{total_matches} matches currently contributes materially usable evidence")
    else:
        status = BLOCKAGE_EASING_MULTI_MATCH
        reasons.append(f"{len(materially_usable_matches)}/{total_matches} matches currently contribute materially usable evidence")

    if top_signal_match is not None and top_signal_share is not None:
        reasons.append(
            f"match {top_signal_match['match_id']} carries {top_signal_match['distinct_signal_active_raw_event_count']}/{total_distinct_signal_active} distinct signal-active raw events"
        )
        if top_signal_share >= 0.75 and total_matches > 1:
            reasons.append("usable evidence is still heavily concentrated in one match")
    if blockers:
        reasons.append(f"top remaining blocker is {blockers[0]['reason']} ({blockers[0]['count']} rows)")

    return {
        "blockage_assessment": status,
        "materially_usable_match_count": len(materially_usable_matches),
        "weak_usable_match_count": len(weak_matches),
        "no_usable_match_count": len(no_usable_matches),
        "top_signal_match_id": None if top_signal_match is None else int(top_signal_match["match_id"]),
        "top_signal_match_share": top_signal_share,
        "reasons": reasons,
    }


def build_backend_bo3_corpus_readiness_report(capture_path: Path) -> dict[str, Any]:
    rows = load_capture_rows(capture_path)
    rows_by_match: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        match_id = row.get("match_id")
        if match_id is None:
            continue
        rows_by_match.setdefault(int(match_id), []).append(row)

    match_summaries = [
        _build_match_summary(match_id, match_rows)
        for match_id, match_rows in sorted(rows_by_match.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    corpus_exclusion_counts: Counter = Counter()
    for summary in match_summaries:
        for reason, count in summary["exclusion_reasons"].items():
            corpus_exclusion_counts[reason] += int(count)

    blockers = _build_top_blockers(match_summaries)
    readiness_summary = _build_readiness_summary(match_summaries, blockers)
    rows_per_match = {str(summary["match_id"]): int(summary["row_count"]) for summary in match_summaries}

    report = {
        "schema_version": SCHEMA_VERSION,
        "input_artifact": {
            "path": str(capture_path),
            "schema_version_expected": CAPTURE_SCHEMA_VERSION,
            "total_rows": len(rows),
            "distinct_match_count": len(match_summaries),
            "rows_per_match": rows_per_match,
        },
        "corpus_summary": {
            "total_rows": len(rows),
            "distinct_match_count": len(match_summaries),
            "rows_per_match": rows_per_match,
            "eligible_compared_row_count": int(sum(summary["eligible_compared_row_count"] for summary in match_summaries)),
            "signal_active_row_count": int(sum(summary["signal_active_row_count"] for summary in match_summaries)),
            "distinct_raw_event_count": int(sum(summary["distinct_raw_event_count"] for summary in match_summaries)),
            "distinct_signal_active_raw_event_count": int(sum(summary["distinct_signal_active_raw_event_count"] for summary in match_summaries)),
            "excluded_row_count": int(sum(summary["excluded_row_count"] for summary in match_summaries)),
            "exclusion_reasons": counter_to_dict(corpus_exclusion_counts),
        },
        "match_summaries": match_summaries,
        "top_blockers": blockers,
        "readiness_summary": readiness_summary,
        "truth_boundary": {
            "separate_from_bounded_single_match_diagnostic": True,
            "preserves_bounded_single_match_tool": True,
            "not_calibration_implementation": True,
            "not_live_parity_implementation": True,
            "not_dashboard_surface": True,
        },
    }
    return report


def write_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a corpus-level BO3 evidence readiness analyzer on the active continuity-protected backend capture corpus."
    )
    parser.add_argument(
        "--capture-path",
        default=str(DEFAULT_CAPTURE_PATH),
        help="Path to the active continuity-protected backend BO3 capture corpus JSONL.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write the machine-readable corpus readiness report.",
    )
    args = parser.parse_args()

    capture_path = Path(args.capture_path)
    output_path = Path(args.output)
    report = build_backend_bo3_corpus_readiness_report(capture_path)
    write_report(report, output_path)
    print(
        json.dumps(
            {
                "blockage_assessment": report["readiness_summary"]["blockage_assessment"],
                "distinct_match_count": report["corpus_summary"]["distinct_match_count"],
                "eligible_compared_row_count": report["corpus_summary"]["eligible_compared_row_count"],
                "distinct_raw_event_count": report["corpus_summary"]["distinct_raw_event_count"],
                "distinct_signal_active_raw_event_count": report["corpus_summary"]["distinct_signal_active_raw_event_count"],
                "output_path": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
