from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.bo3_capture_contract import default_bo3_backend_capture_path

CAPTURE_SCHEMA_VERSION = "backend_bo3_live_capture_contract.v1"
EVIDENCE_SCHEMA_VERSION = "backend_bo3_live_round_calibration_evidence_v1"
REPORT_SCHEMA_VERSION = "backend_bo3_live_round_calibration_evidence_report_v1"
DEFAULT_CAPTURE_PATH = Path(default_bo3_backend_capture_path())
DEFAULT_HISTORY_PATH = Path("logs/history_points.jsonl")
DEFAULT_EVIDENCE_OUTPUT = Path("automation/reports/backend_bo3_live_round_calibration_evidence_v1.json")
DEFAULT_REPORT_OUTPUT = Path("automation/reports/backend_bo3_live_round_calibration_evidence_report_v1.json")


@dataclass(frozen=True)
class JsonlRow:
    lineno: int
    data: dict[str, Any]


@dataclass(frozen=True)
class MalformedRow:
    lineno: int
    reason: str
    detail: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


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


def _parse_capture_ts_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _read_jsonl_rows(path: Path, malformed_reason: str) -> tuple[list[JsonlRow], list[MalformedRow]]:
    rows: list[JsonlRow] = []
    malformed: list[MalformedRow] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            malformed.append(MalformedRow(lineno=lineno, reason=malformed_reason, detail=f"json_decode_error:{exc.msg}"))
            continue
        if not isinstance(data, dict):
            malformed.append(MalformedRow(lineno=lineno, reason=malformed_reason, detail="json_not_object"))
            continue
        rows.append(JsonlRow(lineno=lineno, data=data))
    return rows, malformed


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _sample_malformed(rows: list[MalformedRow]) -> list[dict[str, Any]]:
    return [
        {"lineno": int(row.lineno), "reason": row.reason, "detail": row.detail}
        for row in rows[:5]
    ]


def _capture_exclusion_reason(row: dict[str, Any]) -> str | None:
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
        return "missing_q_intra_total"
    if _float_or_none(row.get("q_intra_total")) is None:
        return "invalid_q_intra_total"
    if row.get("a_side") not in ("T", "CT"):
        return "missing_a_side"
    if _int_or_none(row.get("match_id")) is None:
        return "missing_match_id"
    if any(_int_or_none(row.get(field)) is None for field in ("game_number", "map_index", "round_number")):
        return "missing_round_identity"
    if any(_int_or_none(row.get(field)) is None for field in ("team_one_id", "team_two_id")):
        return "missing_team_identity"
    if row.get("team_a_is_team_one") is None:
        return "missing_team_identity"
    if _parse_capture_ts_iso(row.get("capture_ts_iso")) is None:
        return "invalid_capture_timestamp"
    return None


def _duplicate_bucket_key(row: dict[str, Any]) -> tuple[Any, ...]:
    raw_provider_event_id = row.get("raw_provider_event_id")
    match_id = _int_or_none(row.get("match_id"))
    if raw_provider_event_id not in (None, ""):
        return ("raw_provider_event_id", match_id, str(raw_provider_event_id))
    return (
        "fallback_exact_capture_ts",
        match_id,
        _int_or_none(row.get("game_number")),
        _int_or_none(row.get("map_index")),
        _int_or_none(row.get("round_number")),
        row.get("capture_ts_iso"),
    )


def _identity_payload(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _int_or_none(row.get("match_id")),
        _int_or_none(row.get("game_number")),
        _int_or_none(row.get("map_index")),
        _int_or_none(row.get("round_number")),
        _int_or_none(row.get("team_one_id")),
        _int_or_none(row.get("team_two_id")),
        bool(row.get("team_a_is_team_one")),
        row.get("a_side"),
        None if row.get("raw_provider_event_id") in (None, "") else str(row.get("raw_provider_event_id")),
    )


def _collapse_candidate_group(rows: list[JsonlRow]) -> tuple[dict[str, Any] | None, str | None]:
    if not rows:
        return None, None
    identity_payloads = {_identity_payload(row.data) for row in rows}
    if len(identity_payloads) != 1:
        return None, "duplicate_identity_conflicting_payload"

    q_values = [_float_or_none(row.data.get("q_intra_total")) for row in rows]
    q_values = [value for value in q_values if value is not None]
    if not q_values:
        return None, "invalid_q_intra_total"

    rows_sorted = sorted(rows, key=lambda item: _parse_capture_ts_iso(item.data.get("capture_ts_iso")) or -1.0)
    representative = dict(rows_sorted[0].data)
    representative["_source_capture_linenos"] = [int(row.lineno) for row in rows_sorted]
    representative["source_capture_lineno"] = int(rows_sorted[0].lineno)
    representative["capture_ts_iso_first"] = rows_sorted[0].data["capture_ts_iso"]
    representative["capture_ts_iso_last"] = rows_sorted[-1].data["capture_ts_iso"]
    representative["capture_ts_epoch_last"] = _parse_capture_ts_iso(rows_sorted[-1].data["capture_ts_iso"])
    representative["duplicate_tick_count"] = len(rows_sorted)
    representative["q_intra_total"] = float(statistics.median(q_values))
    return representative, None


def _extract_valid_round_result_labels(
    rows: list[JsonlRow],
) -> tuple[dict[tuple[int, int, int, int], list[dict[str, Any]]], Counter[str], int]:
    labels_by_key: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    bad_data_counts: Counter[str] = Counter()
    seen_round_result_events = 0

    for row in rows:
        event = row.data.get("event")
        if not isinstance(event, dict) or event.get("event_type") != "round_result":
            continue
        seen_round_result_events += 1

        match_id = _int_or_none(row.data.get("match_id"))
        if match_id is None:
            bad_data_counts["invalid_history_match_id"] += 1
            continue
        game_number = _int_or_none(row.data.get("game_number") if row.data.get("game_number") is not None else event.get("game_number"))
        map_index = _int_or_none(row.data.get("map_index") if row.data.get("map_index") is not None else event.get("map_index"))
        round_number = _int_or_none(row.data.get("round_number") if row.data.get("round_number") is not None else event.get("round_number"))
        if game_number is None or map_index is None or round_number is None:
            bad_data_counts["invalid_history_round_identity"] += 1
            continue
        event_time = _float_or_none(row.data.get("t"))
        if event_time is None:
            bad_data_counts["invalid_history_event_time"] += 1
            continue

        label = {
            "source_label_lineno": int(row.lineno),
            "match_id": match_id,
            "game_number": game_number,
            "map_index": map_index,
            "round_number": round_number,
            "team_one_id": _int_or_none(row.data.get("team_one_id")),
            "team_two_id": _int_or_none(row.data.get("team_two_id")),
            "team_a_is_team_one": row.data.get("team_a_is_team_one"),
            "label_event_time": event_time,
            "round_winner_team_id": _int_or_none(event.get("round_winner_team_id")),
            "round_winner_is_team_a": event.get("round_winner_is_team_a"),
        }
        labels_by_key[(match_id, game_number, map_index, round_number)].append(label)

    return labels_by_key, bad_data_counts, seen_round_result_events


def _team_identity_matches(capture_row: dict[str, Any], label: dict[str, Any]) -> bool:
    return (
        _int_or_none(capture_row.get("team_one_id")) == _int_or_none(label.get("team_one_id"))
        and _int_or_none(capture_row.get("team_two_id")) == _int_or_none(label.get("team_two_id"))
        and bool(capture_row.get("team_a_is_team_one")) == bool(label.get("team_a_is_team_one"))
    )


def _build_join_key(capture_row: dict[str, Any]) -> dict[str, int]:
    return {
        "match_id": int(capture_row["match_id"]),
        "game_number": int(capture_row["game_number"]),
        "map_index": int(capture_row["map_index"]),
        "round_number": int(capture_row["round_number"]),
    }


def _join_collapsed_candidate(
    capture_row: dict[str, Any],
    labels_by_key: dict[tuple[int, int, int, int], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    key = (
        int(capture_row["match_id"]),
        int(capture_row["game_number"]),
        int(capture_row["map_index"]),
        int(capture_row["round_number"]),
    )
    labels = labels_by_key.get(key, [])
    if not labels:
        return None, "no_round_result_found"

    capture_ts_epoch_last = capture_row.get("capture_ts_epoch_last")
    later_labels = [label for label in labels if float(label["label_event_time"]) > float(capture_ts_epoch_last)]
    if not later_labels:
        return None, "label_event_not_after_capture"

    identity_matching_labels = [label for label in later_labels if _team_identity_matches(capture_row, label)]
    if not identity_matching_labels:
        return None, "team_identity_mismatch"

    if len(identity_matching_labels) > 1:
        winner_outcomes = {
            (
                _int_or_none(label.get("round_winner_team_id")),
                label.get("round_winner_is_team_a"),
            )
            for label in identity_matching_labels
        }
        if len(winner_outcomes) > 1:
            return None, "conflicting_round_result_outcomes_same_round"
        return None, "multiple_round_result_events_same_round"

    label = identity_matching_labels[0]
    if _int_or_none(label.get("round_winner_team_id")) is None or label.get("round_winner_is_team_a") is None:
        return None, "label_missing_round_winner"

    record = {
        "source_capture_lineno": int(capture_row["source_capture_lineno"]),
        "source_capture_linenos": list(capture_row["_source_capture_linenos"]),
        "source_label_lineno": int(label["source_label_lineno"]),
        "match_id": int(capture_row["match_id"]),
        "game_number": int(capture_row["game_number"]),
        "map_index": int(capture_row["map_index"]),
        "round_number": int(capture_row["round_number"]),
        "raw_provider_event_id": capture_row.get("raw_provider_event_id"),
        "capture_ts_iso_first": str(capture_row["capture_ts_iso_first"]),
        "capture_ts_iso_last": str(capture_row["capture_ts_iso_last"]),
        "duplicate_tick_count": int(capture_row["duplicate_tick_count"]),
        "team_one_id": _int_or_none(capture_row.get("team_one_id")),
        "team_two_id": _int_or_none(capture_row.get("team_two_id")),
        "team_a_is_team_one": bool(capture_row.get("team_a_is_team_one")),
        "a_side": str(capture_row.get("a_side")),
        "q_intra_total": float(capture_row["q_intra_total"]),
        "label_event_time": float(label["label_event_time"]),
        "round_winner_team_id": int(label["round_winner_team_id"]),
        "round_winner_is_team_a": bool(label["round_winner_is_team_a"]),
        "join_key": _build_join_key(capture_row),
        "join_rule": "match_id+game_number+map_index+round_number+strict_later_than",
    }
    return record, None


def build_backend_bo3_live_round_calibration_evidence(
    capture_path: Path,
    history_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_rows, malformed_capture_rows = _read_jsonl_rows(capture_path, "malformed_capture_row")
    history_rows, malformed_history_rows = _read_jsonl_rows(history_path, "malformed_history_row")

    labels_by_key, invalid_history_counts, history_round_result_event_count = _extract_valid_round_result_labels(history_rows)

    not_eligible_counts: Counter[str] = Counter()
    candidate_groups: dict[tuple[Any, ...], list[JsonlRow]] = defaultdict(list)
    for row in capture_rows:
        exclusion_reason = _capture_exclusion_reason(row.data)
        if exclusion_reason is not None:
            not_eligible_counts[exclusion_reason] += 1
            continue
        candidate_groups[_duplicate_bucket_key(row.data)].append(row)

    collapsed_candidates: list[dict[str, Any]] = []
    join_ambiguity_counts: Counter[str] = Counter()
    unlabeled_counts: Counter[str] = Counter()
    labeled_records: list[dict[str, Any]] = []

    for group_rows in candidate_groups.values():
        collapsed, collapse_reason = _collapse_candidate_group(group_rows)
        if collapse_reason is not None:
            join_ambiguity_counts[collapse_reason] += 1
            continue
        collapsed_candidates.append(collapsed)
        labeled_record, join_reason = _join_collapsed_candidate(collapsed, labels_by_key)
        if join_reason is not None:
            if join_reason in {
                "multiple_round_result_events_same_round",
                "conflicting_round_result_outcomes_same_round",
                "team_identity_mismatch",
                "duplicate_identity_conflicting_payload",
            }:
                join_ambiguity_counts[join_reason] += 1
            else:
                unlabeled_counts[join_reason] += 1
            continue
        labeled_records.append(labeled_record)

    bad_data_counts: Counter[str] = Counter()
    if malformed_capture_rows:
        bad_data_counts["malformed_capture_row"] += len(malformed_capture_rows)
    if malformed_history_rows:
        bad_data_counts["malformed_history_row"] += len(malformed_history_rows)
    bad_data_counts.update(invalid_history_counts)

    exclusion_counts: Counter[str] = Counter()
    exclusion_counts.update(not_eligible_counts)
    exclusion_counts.update(unlabeled_counts)
    exclusion_counts.update(join_ambiguity_counts)
    exclusion_counts.update(bad_data_counts)

    duplicate_tick_counts = [int(record["duplicate_tick_count"]) for record in labeled_records]
    duplicate_tick_stats = {
        "max_duplicate_tick_count": 0 if not duplicate_tick_counts else int(max(duplicate_tick_counts)),
        "median_duplicate_tick_count": 0.0 if not duplicate_tick_counts else float(statistics.median(duplicate_tick_counts)),
        "records_with_duplicate_ticks": int(sum(1 for count in duplicate_tick_counts if count > 1)),
    }
    labeled_records_per_match_id = Counter(str(record["match_id"]) for record in labeled_records)

    report_summary = {
        "total_capture_rows_scanned": len(capture_rows),
        "candidate_prediction_row_count": int(sum(len(group_rows) for group_rows in candidate_groups.values())),
        "collapsed_candidate_record_count": len(collapsed_candidates),
        "labeled_record_count": len(labeled_records),
        "unlabeled_eligible_count": int(sum(unlabeled_counts.values())),
        "not_eligible_count": int(sum(not_eligible_counts.values())),
        "join_ambiguity_count": int(sum(join_ambiguity_counts.values())),
        "bad_data_count": int(sum(bad_data_counts.values())),
        "counts_by_exclusion_reason": _counter_to_dict(exclusion_counts),
        "exclusion_reason_units": {
            **{reason: "capture_row" for reason in not_eligible_counts},
            **{reason: "collapsed_candidate_record" for reason in unlabeled_counts},
            **{reason: "collapsed_candidate_record" for reason in join_ambiguity_counts},
            **{reason: "jsonl_row" for reason in bad_data_counts},
        },
        "labeled_records_per_match_id": dict(sorted(labeled_records_per_match_id.items())),
        "duplicate_tick_stats": duplicate_tick_stats,
        "history_round_result_event_count": int(history_round_result_event_count),
        "bad_data_samples": {
            "capture": _sample_malformed(malformed_capture_rows),
            "history": _sample_malformed(malformed_history_rows),
        },
        "truth_boundary": {
            "round_level_q_intra_total_only": True,
            "label_event_type": "round_result",
            "strict_later_than_required": True,
            "requires_match_id_join": True,
            "not_calibration_tuning": True,
            "not_engine_math_change": True,
            "not_runtime_redesign": True,
        },
    }

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "prediction_target": "q_intra_total",
        "label_event_type": "round_result",
        "input_capture_path": str(capture_path),
        "input_history_path": str(history_path),
        "join_contract": {
            "fields": ["match_id", "game_number", "map_index", "round_number"],
            "event_type_required": "round_result",
            "strict_later_than_capture": True,
            "team_identity_checks": ["team_one_id", "team_two_id", "team_a_is_team_one"],
            "duplicate_policy": {
                "primary_bucket": ["match_id", "raw_provider_event_id"],
                "fallback_bucket": ["match_id", "game_number", "map_index", "round_number", "capture_ts_iso"],
                "q_collapse": "median",
                "capture_time_for_leakage": "latest_capture_ts_in_bucket",
            },
        },
        "truth_boundary": {
            "what_this_is": "live_round_level_labeled_calibration_evidence_export",
            "what_this_is_not": [
                "calibration_quality_verdict",
                "parameter_tuning",
                "segment_level_export",
                "runtime_redesign",
            ],
        },
        "summary": report_summary,
        "labeled_prediction_records": labeled_records,
    }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": evidence["generated_at"],
        "prediction_target": "q_intra_total",
        "label_event_type": "round_result",
        "input_capture_path": str(capture_path),
        "input_history_path": str(history_path),
        **report_summary,
    }
    return evidence, report


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export BO3 live round-level labeled calibration evidence for q_intra_total vs round_result."
    )
    parser.add_argument("--capture-path", default=str(DEFAULT_CAPTURE_PATH), help="Path to the active BO3 capture corpus JSONL.")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Path to persisted history_points.jsonl.")
    parser.add_argument("--evidence-output", default=str(DEFAULT_EVIDENCE_OUTPUT), help="Path to write the detailed evidence artifact.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT), help="Path to write the summary report artifact.")
    args = parser.parse_args()

    capture_path = Path(args.capture_path)
    history_path = Path(args.history_path)
    evidence_output = Path(args.evidence_output)
    report_output = Path(args.report_output)

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)
    write_json(evidence_output, evidence)
    write_json(report_output, report)
    print(
        json.dumps(
            {
                "labeled_record_count": report["labeled_record_count"],
                "candidate_prediction_row_count": report["candidate_prediction_row_count"],
                "collapsed_candidate_record_count": report["collapsed_candidate_record_count"],
                "join_ambiguity_count": report["join_ambiguity_count"],
                "bad_data_count": report["bad_data_count"],
                "evidence_output": str(evidence_output),
                "report_output": str(report_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
