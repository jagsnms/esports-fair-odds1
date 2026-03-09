from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.calibration_reliability_evidence_gate import build_calibration_reliability_summary
from tools.run_calibration_reliability_evidence_gate import run_evidence_gate_runner


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools" / "schemas" / "calibration_reliability_evidence.schema.json"


def _bins(a: float, b: float) -> list[dict[str, Any]]:
    return [
        {"bin_index": 0, "count": 5, "mean_prediction": a, "empirical_rate": b},
        {"bin_index": 1, "count": 7, "mean_prediction": a + 0.1, "empirical_rate": b + 0.1},
    ]


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
            assert any(_type_matches(value, t) for t in expected_type), (
                f"{path} must match one of types {expected_type!r}"
            )
        else:
            assert _type_matches(value, expected_type), f"{path} must be of type {expected_type!r}"

    if isinstance(value, str) and "minLength" in schema:
        assert len(value) >= int(schema["minLength"]), f"{path} below minLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        assert value >= schema["minimum"], f"{path} below minimum"
    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= int(schema["minItems"]), f"{path} below minItems"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _assert_schema_conformance(item, item_schema, f"{path}[{idx}]")
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


def _load_schema() -> dict[str, Any]:
    assert SCHEMA_PATH.exists()
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_exists_and_pass_incomplete_fail_payloads_conform() -> None:
    schema = _load_schema()

    pass_records = [
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "current",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.21,
            "log_loss": 0.62,
            "reliability_curve_bins": _bins(0.52, 0.50),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.24,
            "log_loss": 0.66,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.23,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.50, 0.49),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            "log_loss": 0.63,
            "reliability_curve_bins": _bins(0.54, 0.52),
        },
    ]
    incomplete_records = [
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.24,
            "log_loss": 0.66,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            "reliability_curve_bins": _bins(0.54, 0.52),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": None,
            "brier_score": 0.23,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.50, 0.49),
        },
    ]
    fail_records = [
        {
            "evidence_source": "replayish",
            "dataset_id": "bad_record",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": "0.24",
            "log_loss": 0.66,
            "reliability_curve_bins": "not-an-array",
        }
    ]

    pass_payload = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=pass_records,
        generated_at="2026-03-08T00:00:00Z",
    )
    incomplete_payload = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=incomplete_records,
        generated_at="2026-03-08T00:00:00Z",
    )
    fail_payload = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=fail_records,
        generated_at="2026-03-08T00:00:00Z",
    )

    assert pass_payload["gate_status"] == "pass"
    assert incomplete_payload["gate_status"] == "incomplete_evidence"
    assert fail_payload["gate_status"] == "fail"

    _assert_schema_conformance(pass_payload, schema)
    _assert_schema_conformance(incomplete_payload, schema)
    _assert_schema_conformance(fail_payload, schema)


def test_runner_generated_pass_artifact_conforms_to_schema(tmp_path: Path) -> None:
    schema = _load_schema()
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"

    replay_records = [
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.25,
            "log_loss": 0.65,
            "reliability_curve_bins": _bins(0.52, 0.49),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            "log_loss": 0.62,
            "reliability_curve_bins": _bins(0.53, 0.51),
        },
    ]
    simulation_records = [
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.24,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "current",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.23,
            "log_loss": 0.63,
            "reliability_curve_bins": _bins(0.52, 0.50),
        },
    ]
    replay_path.write_text(json.dumps(replay_records, sort_keys=True), encoding="utf-8")
    simulation_path.write_text(json.dumps(simulation_records, sort_keys=True), encoding="utf-8")

    result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="schema_conformance_run",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )

    assert result.exit_code == 0
    assert result.output_path is not None
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    _assert_schema_conformance(payload, schema)
