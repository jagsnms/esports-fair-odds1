from __future__ import annotations

import json
from pathlib import Path

import tools.run_backend_bo3_live_q_intra_measurement as measurement_runner


def _capture_row(
    *,
    match_id: int,
    round_number: int,
    q_intra_total: float,
    capture_ts_iso: str,
    raw_provider_event_id: str,
) -> dict:
    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "capture_ts_iso": capture_ts_iso,
        "match_id": match_id,
        "team_a_is_team_one": True,
        "raw_provider_event_id": raw_provider_event_id,
        "raw_seq_index": round_number,
        "game_number": 1,
        "map_index": 0,
        "round_number": round_number,
        "round_phase": "IN_PROGRESS",
        "team_one_id": 101,
        "team_two_id": 202,
        "a_side": "CT",
        "bo3_snapshot_status": "live",
        "bo3_health": "GOOD",
        "clamp_reason": "ok",
        "q_intra_total": q_intra_total,
    }


def _round_result_row(
    *,
    match_id: int,
    round_number: int,
    event_time: float,
    round_winner_is_team_a: bool,
) -> dict:
    return {
        "t": event_time,
        "p": 0.5,
        "lo": 0.01,
        "hi": 0.99,
        "match_id": match_id,
        "game_number": 1,
        "map_index": 0,
        "round_number": round_number,
        "team_one_id": 101,
        "team_two_id": 202,
        "team_a_is_team_one": True,
        "event": {
            "event_type": "round_result",
            "game_number": 1,
            "map_index": 0,
            "round_number": round_number,
            "round_winner_team_id": 101 if round_winner_is_team_a else 202,
            "round_winner_is_team_a": round_winner_is_team_a,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_happy_path_runs_exporter_then_gate_and_reports_success(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_output = tmp_path / "evidence.json"
    report_output = tmp_path / "report.json"
    gate_output = tmp_path / "gate.json"

    capture_rows: list[dict] = []
    history_rows: list[dict] = []
    probabilities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] * 10
    for idx, q_value in enumerate(probabilities, start=1):
        capture_rows.append(
            _capture_row(
                match_id=501,
                round_number=idx,
                q_intra_total=q_value,
                capture_ts_iso=f"2026-03-12T10:{idx // 60:02d}:{idx % 60:02d}.000Z",
                raw_provider_event_id=f"event-{idx}",
            )
        )
        history_rows.append(
            _round_result_row(
                match_id=501,
                round_number=idx,
                event_time=1773370800.0 + idx,
                round_winner_is_team_a=(idx % 2 == 0),
            )
        )

    _write_jsonl(capture_path, capture_rows)
    _write_jsonl(history_path, history_rows)

    result = measurement_runner.run_backend_bo3_live_q_intra_measurement(
        capture_path=str(capture_path),
        history_path=str(history_path),
        evidence_output=str(evidence_output),
        report_output=str(report_output),
        gate_output=str(gate_output),
        generated_at="2026-03-19T00:00:00Z",
    )

    assert result.exit_code == 0
    assert result.payload == {
        "exporter_evidence_output_path": str(evidence_output),
        "exporter_report_output_path": str(report_output),
        "gate_output_path": str(gate_output),
        "gate_status": "sufficient_evidence",
        "insufficiency_reasons": [],
        "labeled_record_count": 100,
    }
    assert evidence_output.exists()
    assert report_output.exists()
    assert gate_output.exists()


def test_insufficiency_path_completes_and_reports_gate_status(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_output = tmp_path / "evidence.json"
    report_output = tmp_path / "report.json"
    gate_output = tmp_path / "gate.json"

    capture_rows = [
        _capture_row(
            match_id=501,
            round_number=idx,
            q_intra_total=q_value,
            capture_ts_iso=f"2026-03-12T10:00:0{idx}.000Z",
            raw_provider_event_id=f"event-{idx}",
        )
        for idx, q_value in enumerate([0.2, 0.25, 0.3, 0.35], start=1)
    ]
    history_rows = [
        _round_result_row(
            match_id=501,
            round_number=idx,
            event_time=1773370800.0 + idx,
            round_winner_is_team_a=(idx % 2 == 0),
        )
        for idx in range(1, 5)
    ]

    _write_jsonl(capture_path, capture_rows)
    _write_jsonl(history_path, history_rows)

    result = measurement_runner.run_backend_bo3_live_q_intra_measurement(
        capture_path=str(capture_path),
        history_path=str(history_path),
        evidence_output=str(evidence_output),
        report_output=str(report_output),
        gate_output=str(gate_output),
        generated_at="2026-03-19T00:00:00Z",
    )

    assert result.exit_code == 0
    assert result.payload is not None
    assert result.payload["gate_status"] == "insufficient_evidence"
    assert result.payload["insufficiency_reasons"] == [
        "labeled_record_count_below_threshold",
        "non_empty_bin_count_below_threshold",
    ]
    assert result.payload["labeled_record_count"] == 4


def test_exporter_failure_propagates_nonzero(tmp_path: Path) -> None:
    result = measurement_runner.run_backend_bo3_live_q_intra_measurement(
        capture_path=str(tmp_path / "missing_capture.jsonl"),
        history_path=str(tmp_path / "missing_history.jsonl"),
        evidence_output=str(tmp_path / "evidence.json"),
        report_output=str(tmp_path / "report.json"),
        gate_output=str(tmp_path / "gate.json"),
    )

    assert result.exit_code == 1
    assert result.payload is None
    assert result.message.startswith("exporter_failed:")


def test_gate_failure_propagates_nonzero(monkeypatch, tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_output = tmp_path / "evidence.json"
    report_output = tmp_path / "report.json"
    gate_output = tmp_path / "gate.json"
    _write_jsonl(
        capture_path,
        [
            _capture_row(
                match_id=501,
                round_number=1,
                q_intra_total=0.5,
                capture_ts_iso="2026-03-12T10:00:00.000Z",
                raw_provider_event_id="event-1",
            )
        ],
    )
    _write_jsonl(
        history_path,
        [
            _round_result_row(
                match_id=501,
                round_number=1,
                event_time=1773370801.0,
                round_winner_is_team_a=True,
            )
        ],
    )

    class _FakeGateResult:
        exit_code = 1
        output_path = None
        message = "forced gate failure"
        artifact = None

    monkeypatch.setattr(
        measurement_runner,
        "run_backend_bo3_live_q_intra_reliability_gate",
        lambda **_: _FakeGateResult(),
    )

    result = measurement_runner.run_backend_bo3_live_q_intra_measurement(
        capture_path=str(capture_path),
        history_path=str(history_path),
        evidence_output=str(evidence_output),
        report_output=str(report_output),
        gate_output=str(gate_output),
    )

    assert result.exit_code == 1
    assert result.payload is None
    assert result.message == "gate_failed: forced gate failure"


def test_runner_writes_no_artifact_beyond_exporter_and_gate_outputs(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_output = tmp_path / "evidence.json"
    report_output = tmp_path / "report.json"
    gate_output = tmp_path / "gate.json"

    _write_jsonl(
        capture_path,
        [
            _capture_row(
                match_id=501,
                round_number=1,
                q_intra_total=0.5,
                capture_ts_iso="2026-03-12T10:00:00.000Z",
                raw_provider_event_id="event-1",
            )
        ],
    )
    _write_jsonl(
        history_path,
        [
            _round_result_row(
                match_id=501,
                round_number=1,
                event_time=1773370801.0,
                round_winner_is_team_a=True,
            )
        ],
    )

    result = measurement_runner.run_backend_bo3_live_q_intra_measurement(
        capture_path=str(capture_path),
        history_path=str(history_path),
        evidence_output=str(evidence_output),
        report_output=str(report_output),
        gate_output=str(gate_output),
        generated_at="2026-03-19T00:00:00Z",
    )

    assert result.exit_code == 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "capture.jsonl",
        "history.jsonl",
        "evidence.json",
        "report.json",
        "gate.json",
    }
