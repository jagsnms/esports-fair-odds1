from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tools.export_backend_bo3_live_round_calibration_evidence import (
    build_backend_bo3_live_round_calibration_evidence,
)
from tools.run_backend_bo3_live_q_intra_reliability_gate import (
    SCHEMA_VERSION,
    run_backend_bo3_live_q_intra_reliability_gate,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools" / "schemas" / "backend_bo3_live_q_intra_reliability_gate.schema.json"


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "null":
        return value is None
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return False


def _assert_schema_conformance(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "const" in schema:
        assert value == schema["const"], f"{path} must equal const {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{path} must be one of {schema['enum']!r}"

    expected_type = schema.get("type")
    if expected_type is not None:
        if isinstance(expected_type, list):
            assert any(_type_matches(value, t) for t in expected_type), f"{path} must match {expected_type!r}"
        else:
            assert _type_matches(value, expected_type), f"{path} must be of type {expected_type!r}"

    if isinstance(value, str) and "minLength" in schema:
        assert len(value) >= int(schema["minLength"]), f"{path} below minLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            assert value >= schema["minimum"], f"{path} below minimum"
        if "maximum" in schema:
            assert value <= schema["maximum"], f"{path} above maximum"
    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= int(schema["minItems"]), f"{path} below minItems"
        if "maxItems" in schema:
            assert len(value) <= int(schema["maxItems"]), f"{path} above maxItems"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _assert_schema_conformance(item, item_schema, f"{path}[{idx}]")
        contains_schema = schema.get("contains")
        if isinstance(contains_schema, dict):
            assert any(
                not _schema_assertion_error(item, contains_schema) for item in value
            ), f"{path} missing required contains element"
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            assert key in value, f"{path} missing required key {key!r}"
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item_value in value.items():
            if key in properties:
                _assert_schema_conformance(item_value, properties[key], f"{path}.{key}")
            elif additional is False:
                raise AssertionError(f"{path} has unexpected key {key!r}")


def _schema_assertion_error(value: Any, schema: dict[str, Any]) -> bool:
    try:
        _assert_schema_conformance(value, schema)
    except AssertionError:
        return True
    return False


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _evidence_payload_from_probs(probabilities: list[float]) -> dict[str, Any]:
    return {
        "schema_version": "backend_bo3_live_round_calibration_evidence_v1",
        "generated_at": "2026-03-19T00:00:00Z",
        "prediction_target": "q_intra_total",
        "label_event_type": "round_result",
        "join_contract": {},
        "truth_boundary": {},
        "summary": {"labeled_record_count": len(probabilities)},
        "labeled_prediction_records": [
            {
                "match_id": 501,
                "game_number": 1,
                "map_index": 0,
                "round_number": idx + 1,
                "team_one_id": 1,
                "team_two_id": 2,
                "team_a_is_team_one": True,
                "a_side": "CT",
                "q_intra_total": prob,
                "label_event_time": float(idx + 1000),
                "round_winner_team_id": 1 if idx % 2 == 0 else 2,
                "round_winner_is_team_a": idx % 2 == 0,
            }
            for idx, prob in enumerate(probabilities)
        ],
    }


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
) -> dict[str, Any]:
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
    t: float,
    match_id: int = 501,
    game_number: int = 2,
    map_index: int = 1,
    round_number: int = 9,
    team_one_id: int = 101,
    team_two_id: int = 202,
    team_a_is_team_one: bool = True,
    round_winner_team_id: int = 101,
    round_winner_is_team_a: bool = True,
) -> dict[str, Any]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_happy_path_emits_schema_counts_metrics_and_ten_bins(tmp_path: Path) -> None:
    schema = _load_schema()
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "gate.json"
    probabilities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] * 10
    _write_json(evidence_path, _evidence_payload_from_probs(probabilities))

    prev_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = run_backend_bo3_live_q_intra_reliability_gate(
            source_evidence_input="evidence.json",
            output_path=str(output_path),
            generated_at="2026-03-19T00:00:00Z",
        )
    finally:
        os.chdir(prev_cwd)

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source_evidence_path"] == "evidence.json"
    assert payload["sample_counts"] == {
        "labeled_record_count": 100,
        "non_empty_bin_count": 10,
        "empty_bin_count": 0,
    }
    assert payload["gate_status"] == "sufficient_evidence"
    assert payload["insufficiency_reasons"] == []
    assert isinstance(payload["metrics"]["brier_score"], float)
    assert isinstance(payload["metrics"]["log_loss"], float)
    assert len(payload["metrics"]["reliability_curve_bins"]) == 10
    assert payload["truth_boundary"]["what_this_is_not"] == [
        "parameter_tuning",
        "calibration_quality_certification",
        "p_hat_calibration",
        "rails_calibration",
        "engine_math_change",
        "replay_canonical_matching",
        "bo3_upstream_coarse_progression_work",
    ]
    _assert_schema_conformance(payload, schema)


def test_insufficiency_emits_artifact_with_reasons(tmp_path: Path) -> None:
    schema = _load_schema()
    evidence_path = tmp_path / "thin_evidence.json"
    output_path = tmp_path / "gate.json"
    _write_json(evidence_path, _evidence_payload_from_probs([0.2, 0.25, 0.3, 0.35]))

    result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=str(evidence_path),
        output_path=str(output_path),
        generated_at="2026-03-19T00:00:00Z",
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "insufficient_evidence"
    assert payload["insufficiency_reasons"] == [
        "labeled_record_count_below_threshold",
        "non_empty_bin_count_below_threshold",
    ]
    assert payload["sample_counts"]["labeled_record_count"] == 4
    _assert_schema_conformance(payload, schema)


def test_zero_record_policy_emits_null_metrics_and_zero_bins(tmp_path: Path) -> None:
    schema = _load_schema()
    evidence_path = tmp_path / "zero.json"
    output_path = tmp_path / "gate.json"
    payload = _evidence_payload_from_probs([])
    _write_json(evidence_path, payload)

    result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=str(evidence_path),
        output_path=str(output_path),
        generated_at="2026-03-19T00:00:00Z",
    )

    assert result.exit_code == 0
    gate = json.loads(output_path.read_text(encoding="utf-8"))
    assert gate["gate_status"] == "insufficient_evidence"
    assert gate["insufficiency_reasons"] == [
        "no_labeled_records",
        "labeled_record_count_below_threshold",
        "non_empty_bin_count_below_threshold",
    ]
    assert gate["metrics"]["brier_score"] is None
    assert gate["metrics"]["log_loss"] is None
    assert len(gate["metrics"]["reliability_curve_bins"]) == 10
    assert all(row["count"] == 0 for row in gate["metrics"]["reliability_curve_bins"])
    _assert_schema_conformance(gate, schema)


def test_malformed_evidence_refusal_emits_no_artifact(tmp_path: Path) -> None:
    evidence_path = tmp_path / "bad.json"
    output_path = tmp_path / "gate.json"
    bad_payload = _evidence_payload_from_probs([0.5])
    bad_payload["schema_version"] = "wrong_schema"
    _write_json(evidence_path, bad_payload)

    result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=str(evidence_path),
        output_path=str(output_path),
    )

    assert result.exit_code == 1
    assert result.output_path is None
    assert "schema_version" in result.message
    assert not output_path.exists()


def test_out_of_range_q_intra_refusal_emits_no_artifact(tmp_path: Path) -> None:
    evidence_path = tmp_path / "bad_q.json"
    output_path = tmp_path / "gate.json"
    bad_payload = _evidence_payload_from_probs([0.5])
    bad_payload["labeled_prediction_records"][0]["q_intra_total"] = 1.2
    _write_json(evidence_path, bad_payload)

    result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=str(evidence_path),
        output_path=str(output_path),
    )

    assert result.exit_code == 1
    assert result.output_path is None
    assert "within [0,1]" in result.message
    assert not output_path.exists()


def test_compatibility_with_current_exporter_shaped_payload_contract(tmp_path: Path) -> None:
    schema = _load_schema()
    capture_path = tmp_path / "capture.jsonl"
    history_path = tmp_path / "history.jsonl"
    evidence_path = tmp_path / "exported_evidence.json"
    output_path = tmp_path / "gate.json"
    _write_jsonl(capture_path, [_capture_row()])
    _write_jsonl(history_path, [_round_result_row(t=1773370800.0)])

    evidence_payload, _ = build_backend_bo3_live_round_calibration_evidence(capture_path, history_path)
    _write_json(evidence_path, evidence_payload)

    prev_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = run_backend_bo3_live_q_intra_reliability_gate(
            source_evidence_input="exported_evidence.json",
            output_path=str(output_path),
            generated_at="2026-03-19T00:00:00Z",
        )
    finally:
        os.chdir(prev_cwd)

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_evidence_path"] == "exported_evidence.json"
    assert payload["source_evidence_schema_version"] == "backend_bo3_live_round_calibration_evidence_v1"
    assert payload["prediction_target"] == "q_intra_total"
    assert payload["label_event_type"] == "round_result"
    assert payload["sample_counts"]["labeled_record_count"] == 1
    assert payload["metrics"]["brier_score"] is not None
    _assert_schema_conformance(payload, schema)
