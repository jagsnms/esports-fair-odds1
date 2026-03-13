from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.export_backend_bo3_live_round_calibration_evidence import (
    build_backend_bo3_live_round_calibration_evidence,
    write_json,
)


def _write_jsonl(path: Path, rows: list[str | dict]) -> None:
    path.write_text(
        "".join(
            (row if isinstance(row, str) else json.dumps(row)) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _capture_row(
    *,
    match_id: int = 501,
    capture_ts_iso: str = "2026-03-12T10:00:00.000Z",
    raw_provider_event_id: str | None = "event-1",
    q_intra_total: float = 0.61,
    game_number: int = 2,
    map_index: int = 1,
    round_number: int = 9,
    team_one_id: int = 101,
    team_two_id: int = 202,
    team_a_is_team_one: bool = True,
    a_side: str = "CT",
) -> dict:
    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "capture_ts_iso": capture_ts_iso,
        "match_id": match_id,
        "team_a_is_team_one": team_a_is_team_one,
        "raw_provider_event_id": raw_provider_event_id,
        "raw_seq_index": 11,
        "game_number": game_number,
        "map_index": map_index,
        "round_number": round_number,
        "round_phase": "IN_PROGRESS",
        "team_one_id": team_one_id,
        "team_two_id": team_two_id,
        "a_side": a_side,
        "bo3_snapshot_status": "live",
        "bo3_health": "GOOD",
        "clamp_reason": "ok",
        "q_intra_total": q_intra_total,
    }


def _round_result_row(
    *,
    t: float = 1773370800.0,
    match_id: int = 501,
    game_number: int = 2,
    map_index: int = 1,
    round_number: int = 9,
    team_one_id: int = 101,
    team_two_id: int = 202,
    team_a_is_team_one: bool = True,
    round_winner_team_id: int = 101,
    round_winner_is_team_a: bool = True,
) -> dict:
    return {
        "t": t,
        "p": 0.5,
        "lo": 0.01,
        "hi": 0.99,
        "match_id": match_id,
        "game_number": game_number,
        "map_index": map_index,
        "round_number": round_number,
        "team_one_id": team_one_id,
        "team_two_id": team_two_id,
        "team_a_is_team_one": team_a_is_team_one,
        "event": {
            "event_type": "round_result",
            "game_number": game_number,
            "map_index": map_index,
            "round_number": round_number,
            "round_winner_team_id": round_winner_team_id,
            "round_winner_is_team_a": round_winner_is_team_a,
        },
    }


def test_happy_path_same_match_join(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, [_capture_row()])
    _write_jsonl(history_path, [_round_result_row()])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert report["labeled_record_count"] == 1
    assert report["join_ambiguity_count"] == 0
    record = evidence["labeled_prediction_records"][0]
    assert record["match_id"] == 501
    assert record["round_winner_is_team_a"] is True
    assert record["join_key"] == {"match_id": 501, "game_number": 2, "map_index": 1, "round_number": 9}


def test_wrong_match_later_label_prevented_by_match_id(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, [_capture_row(match_id=501)])
    _write_jsonl(history_path, [_round_result_row(match_id=999)])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert evidence["labeled_prediction_records"] == []
    assert report["labeled_record_count"] == 0
    assert report["counts_by_exclusion_reason"]["no_round_result_found"] == 1


def test_duplicate_collapse_uses_median_q_and_duplicate_count(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    capture_rows = [
        _capture_row(capture_ts_iso="2026-03-12T10:00:00.000Z", q_intra_total=0.20),
        _capture_row(capture_ts_iso="2026-03-12T10:00:05.000Z", q_intra_total=0.60),
        _capture_row(capture_ts_iso="2026-03-12T10:00:09.000Z", q_intra_total=0.90),
    ]
    _write_jsonl(capture_path, capture_rows)
    _write_jsonl(history_path, [_round_result_row()])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert report["collapsed_candidate_record_count"] == 1
    assert report["labeled_record_count"] == 1
    record = evidence["labeled_prediction_records"][0]
    assert record["duplicate_tick_count"] == 3
    assert abs(record["q_intra_total"] - 0.60) < 1e-9
    assert record["capture_ts_iso_last"] == "2026-03-12T10:00:09.000Z"


def test_strict_later_leakage_refusal(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, [_capture_row(capture_ts_iso="2026-03-12T10:00:00.000Z")])
    _write_jsonl(history_path, [_round_result_row(t=1000.0)])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert evidence["labeled_prediction_records"] == []
    assert report["counts_by_exclusion_reason"]["label_event_not_after_capture"] == 1


def test_conflicting_round_result_refusal(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, [_capture_row()])
    _write_jsonl(
        history_path,
        [
            _round_result_row(t=1773370800.0, round_winner_team_id=101, round_winner_is_team_a=True),
            _round_result_row(t=1773370801.0, round_winner_team_id=202, round_winner_is_team_a=False),
        ],
    )

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert evidence["labeled_prediction_records"] == []
    assert report["counts_by_exclusion_reason"]["conflicting_round_result_outcomes_same_round"] == 1


def test_malformed_capture_row_accounted_for(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, ['{"not valid"', _capture_row()])
    _write_jsonl(history_path, [_round_result_row()])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert report["bad_data_count"] == 1
    assert report["counts_by_exclusion_reason"]["malformed_capture_row"] == 1
    assert report["bad_data_samples"]["capture"][0]["lineno"] == 1
    assert len(evidence["labeled_prediction_records"]) == 1


def test_malformed_history_row_accounted_for(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    _write_jsonl(capture_path, [_capture_row()])
    _write_jsonl(history_path, ['{"broken"', _round_result_row()])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)

    assert report["bad_data_count"] == 1
    assert report["counts_by_exclusion_reason"]["malformed_history_row"] == 1
    assert report["bad_data_samples"]["history"][0]["lineno"] == 1
    assert len(evidence["labeled_prediction_records"]) == 1


def test_artifact_report_shape(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_path = tmp_path / "evidence.json"
    report_path = tmp_path / "report.json"
    _write_jsonl(capture_path, [_capture_row()])
    _write_jsonl(history_path, [_round_result_row()])

    evidence, report = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)
    write_json(evidence_path, evidence)
    write_json(report_path, report)

    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert evidence_payload["schema_version"] == "backend_bo3_live_round_calibration_evidence_v1"
    assert report_payload["schema_version"] == "backend_bo3_live_round_calibration_evidence_report_v1"
    assert "labeled_prediction_records" in evidence_payload
    assert "counts_by_exclusion_reason" in report_payload
    assert "truth_boundary" in report_payload
