from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import tools.validate_engine_spec_diagnostics_parity_output as parity_output_validator
from tools.validate_engine_spec_diagnostics_parity_output import (
    validate_engine_spec_diagnostics_parity_output,
)


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "validate_engine_spec_diagnostics_parity_output.py"
DEFAULT_SCHEMA = ROOT / "tools" / "schemas" / "engine_spec_diagnostics_parity.schema.json"
OUTPUT_SCHEMA = ROOT / "tools" / "schemas" / "engine_spec_diagnostics_parity_output_validator.schema.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _valid_parity_payload() -> dict:
    return {
        "schema_version": "engine_spec_diagnostics_parity.v1",
        "status": "pass",
        "validated_at": "2026-03-09T20:00:00Z",
        "parity_mode": "exact_name_path_only",
        "source_of_truth": {
            "engine_spec_path": "/workspace/docs/ENGINE_SPEC.json",
            "engine_spec_field_path": "invariants.diagnostics_payload_required_fields",
        },
        "assessment_source": {
            "artifact_path_or_stream": "/tmp/assessment.json",
            "assessment_field_path": "contract_diagnostics_spec_required_keys",
        },
        "spec_required_fields": ["q", "rail_low"],
        "assessment_required_fields": ["q", "rail_low"],
        "matched_fields": ["q", "rail_low"],
        "missing_fields": [],
        "extra_assessment_fields": [],
        "counts": {
            "spec_required_count": 2,
            "assessment_required_count": 2,
            "matched_count": 2,
            "missing_count": 0,
            "extra_count": 0,
        },
        "errors": [],
    }


def _assert_common_shape(payload: dict) -> None:
    required_keys = {
        "schema_version",
        "status",
        "artifact_path",
        "schema_path",
        "validated_at",
        "errors",
        "violations",
        "artifact_metadata",
    }
    assert set(payload.keys()) == required_keys
    assert payload["schema_version"] == "engine_spec_diagnostics_parity_output_validator.v1"
    assert payload["status"] in {"pass", "fail", "blocked"}
    assert isinstance(payload["artifact_path"], str)
    assert isinstance(payload["schema_path"], str)
    assert isinstance(payload["validated_at"], str)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["violations"], list)
    assert isinstance(payload["artifact_metadata"], dict)
    assert set(payload["artifact_metadata"].keys()) == {
        "schema_version",
        "status",
        "parity_mode",
    }


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
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
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


def test_output_schema_exists_and_validates_pass_fail_blocked_payloads(tmp_path: Path) -> None:
    assert OUTPUT_SCHEMA.exists()
    schema = json.loads(OUTPUT_SCHEMA.read_text(encoding="utf-8"))

    pass_artifact = tmp_path / "valid_parity.json"
    fail_artifact = tmp_path / "invalid_parity.json"
    missing_artifact = tmp_path / "missing_parity.json"
    _write_json(pass_artifact, _valid_parity_payload())
    fail_payload_input = _valid_parity_payload()
    del fail_payload_input["counts"]
    _write_json(fail_artifact, fail_payload_input)

    pass_exit, pass_payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=pass_artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    fail_exit, fail_payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=fail_artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    blocked_exit, blocked_payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=missing_artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )

    assert pass_exit == 0
    assert fail_exit == 2
    assert blocked_exit == 1
    _assert_schema_conformance(pass_payload, schema)
    _assert_schema_conformance(fail_payload, schema)
    _assert_schema_conformance(blocked_payload, schema)


def test_pass_with_valid_parity_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["errors"] == []
    assert payload["violations"] == []
    assert payload["artifact_metadata"] == {
        "schema_version": "engine_spec_diagnostics_parity.v1",
        "status": "pass",
        "parity_mode": "exact_name_path_only",
    }
    _assert_common_shape(payload)


def test_fail_with_schema_violating_parity_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_parity.json"
    payload = _valid_parity_payload()
    del payload["counts"]
    _write_json(artifact, payload)
    exit_code, result = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 2
    assert result["status"] == "fail"
    assert result["errors"] == []
    assert len(result["violations"]) > 0
    _assert_common_shape(result)


def test_blocked_on_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "does_not_exist.json"
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("artifact unreadable" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_invalid_artifact_json(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_json.json"
    artifact.write_text("{not valid json", encoding="utf-8")
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("artifact invalid JSON" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_missing_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    missing_schema = tmp_path / "does_not_exist_schema.json"
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=missing_schema,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("schema unreadable" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_invalid_schema_json(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    schema = tmp_path / "invalid_schema.json"
    schema.write_text("{not valid json", encoding="utf-8")
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=schema,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("schema invalid JSON" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_unavailable_validator_engine(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    monkeypatch.setattr(parity_output_validator, "Draft202012Validator", None)
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("schema validator unavailable" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_invalid_required_args_validated_at_blank(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    exit_code, payload = validate_engine_spec_diagnostics_parity_output(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("invalid validated_at" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_deterministic_output_shape_and_stdout(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_parity.json"
    _write_json(artifact, _valid_parity_payload())
    cmd = [
        sys.executable,
        str(TOOL),
        "--artifact-path",
        str(artifact),
        "--validated-at",
        "2026-03-09T20:00:00Z",
    ]
    first = subprocess.run(cmd, check=False, capture_output=True, text=True)
    second = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    _assert_common_shape(payload)
    assert list(payload.keys()) == sorted(payload.keys())
