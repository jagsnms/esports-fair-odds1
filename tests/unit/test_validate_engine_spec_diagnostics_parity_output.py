from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tools.validate_engine_spec_diagnostics_parity_output as parity_output_validator
from tools.validate_engine_spec_diagnostics_parity_output import (
    validate_engine_spec_diagnostics_parity_output,
)


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "validate_engine_spec_diagnostics_parity_output.py"
DEFAULT_SCHEMA = ROOT / "tools" / "schemas" / "engine_spec_diagnostics_parity.schema.json"


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
