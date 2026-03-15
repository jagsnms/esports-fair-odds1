from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.validate_replay_validation_summary import validate_replay_validation_summary


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "validate_replay_validation_summary.py"
DEFAULT_SCHEMA = ROOT / "tools" / "schemas" / "replay_validation_summary.schema.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _valid_replay_summary_payload() -> dict:
    return {
        "schema_version": "replay_validation_summary.v1",
        "fixture_class": "raw_replay_sample",
        "replay_path": "tools/fixtures/raw_replay_sample.jsonl",
        "replay_contract_policy": "reject_point_like",
        "replay_point_transition_enabled": False,
        "direct_load_payload_count": 3,
        "replay_payload_count_loaded": 3,
        "total_points_captured": 3,
        "raw_contract_points": 3,
        "point_passthrough_points": 0,
        "rail_input_contract_policy_counts": {"v2_strict": 3},
        "rail_input_active_endpoint_semantics_counts": {"v1": 3},
        "rail_input_reason_code_counts": {"V2_REQUIRED_FIELDS_MISSING": 3},
        "rail_input_v2_activated_points": 0,
        "rail_input_v1_fallback_points": 3,
        "point_like_inputs_seen": 0,
        "point_like_inputs_rejected": 0,
        "point_like_inputs_transition_passthrough": 0,
        "point_like_reject_reason_counts": {},
        "points_with_contract_diagnostics": 3,
        "contract_diagnostics_spec_required_keys": [
            "round_state_vector",
            "q",
            "rail_low",
        ],
        "contract_diagnostics_required_keys": [
            "alive_counts",
            "hp_totals",
            "loadout_totals",
        ],
        "contract_diagnostics_key_presence_counts": {
            "alive_counts": 3,
            "hp_totals": 3,
            "loadout_totals": 3,
        },
        "contract_diagnostics_missing_key_counts": {
            "alive_counts": 0,
            "hp_totals": 0,
            "loadout_totals": 0,
        },
        "contract_diagnostics_key_presence_rates": {
            "alive_counts": 1.0,
            "hp_totals": 1.0,
            "loadout_totals": 1.0,
        },
        "structural_violations_total": 0,
        "behavioral_violations_total": 0,
        "invariant_violations_total": 0,
    }


def _valid_replay_point_source_payload() -> dict:
    return {
        "records": [
            {
                "p_hat": 0.5,
                "rail_low": 0.4,
                "rail_high": 0.6,
                "game_number": None,
                "map_index": 0,
                "round_number": None,
                "time": None,
            }
        ]
    }


def _valid_common_point_source_basis_payload() -> dict:
    return {
        "contract_id": "common_point_source_basis.v1",
        "shared_fields": [
            "p_hat",
            "rail_low",
            "rail_high",
            "game_number",
            "map_index",
            "round_number",
        ],
        "excluded_fields_by_source": {
            "replay_point_source": [
                "time",
                "point.event",
                "derived",
                "derived.debug",
            ],
            "canonical_phase2_trace": [
                "point_index_in_round",
                "label_scope",
                "round_winner_team_id",
                "round_winner_is_team_a",
            ],
        },
        "contract_limits": {
            "shared_field_subset_only": True,
            "record_matching_implied": False,
            "alignment_implied": False,
            "scoring_or_selection_implied": False,
        },
    }


def _valid_common_point_source_projection_payload() -> dict:
    return {
        "contract_id": "common_point_source_projection.v1",
        "source_surface": "replay_point_source",
        "shared_fields": [
            "p_hat",
            "rail_low",
            "rail_high",
            "game_number",
            "map_index",
            "round_number",
        ],
        "projection_limits": {
            "side_local_projection_only": True,
            "record_matching_implied": False,
            "alignment_implied": False,
            "scoring_or_selection_implied": False,
        },
        "records": [
            {
                "p_hat": 0.5,
                "rail_low": 0.4,
                "rail_high": 0.6,
                "game_number": None,
                "map_index": 0,
                "round_number": None,
            }
        ],
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
        "summary_metadata",
    }
    assert set(payload.keys()) == required_keys
    assert payload["schema_version"] == "replay_validation_summary_validator.v1"
    assert payload["status"] in {"pass", "fail", "blocked"}
    assert isinstance(payload["artifact_path"], str)
    assert isinstance(payload["schema_path"], str)
    assert isinstance(payload["validated_at"], str)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["violations"], list)
    assert isinstance(payload["summary_metadata"], dict)
    assert set(payload["summary_metadata"].keys()) == {"schema_version", "fixture_class", "replay_path"}


def test_pass_with_valid_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_summary.json"
    _write_json(artifact, _valid_replay_summary_payload())
    exit_code, payload = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["errors"] == []
    assert payload["violations"] == []
    _assert_common_shape(payload)


def test_pass_with_valid_artifact_including_replay_point_source(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_summary_with_point_source.json"
    payload = _valid_replay_summary_payload()
    payload["common_point_source_basis"] = _valid_common_point_source_basis_payload()
    payload["common_point_source_projection"] = _valid_common_point_source_projection_payload()
    payload["replay_point_source"] = _valid_replay_point_source_payload()
    _write_json(artifact, payload)
    exit_code, result = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 0
    assert result["status"] == "pass"
    assert result["errors"] == []
    assert result["violations"] == []
    _assert_common_shape(result)


def test_fail_with_out_of_scope_field_in_replay_point_source_record(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_summary_with_point_source.json"
    payload = _valid_replay_summary_payload()
    payload["common_point_source_basis"] = _valid_common_point_source_basis_payload()
    payload["common_point_source_projection"] = _valid_common_point_source_projection_payload()
    payload["replay_point_source"] = _valid_replay_point_source_payload()
    payload["replay_point_source"]["records"][0]["event"] = None
    _write_json(artifact, payload)
    exit_code, result = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 2
    assert result["status"] == "fail"
    assert result["errors"] == []
    assert any("$.replay_point_source.records[0]" in violation for violation in result["violations"])
    _assert_common_shape(result)


def test_fail_with_invalid_common_point_source_basis_shared_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_common_basis_summary.json"
    payload = _valid_replay_summary_payload()
    payload["common_point_source_basis"] = _valid_common_point_source_basis_payload()
    payload["common_point_source_projection"] = _valid_common_point_source_projection_payload()
    payload["common_point_source_basis"]["shared_fields"].append("time")
    payload["replay_point_source"] = _valid_replay_point_source_payload()
    _write_json(artifact, payload)
    exit_code, result = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 2
    assert result["status"] == "fail"
    assert result["errors"] == []
    assert any("$.common_point_source_basis.shared_fields" in violation for violation in result["violations"])
    _assert_common_shape(result)


def test_fail_with_out_of_scope_field_in_common_point_source_projection_record(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_projection_summary.json"
    payload = _valid_replay_summary_payload()
    payload["common_point_source_basis"] = _valid_common_point_source_basis_payload()
    payload["common_point_source_projection"] = _valid_common_point_source_projection_payload()
    payload["common_point_source_projection"]["records"][0]["time"] = None
    payload["replay_point_source"] = _valid_replay_point_source_payload()
    _write_json(artifact, payload)
    exit_code, result = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 2
    assert result["status"] == "fail"
    assert result["errors"] == []
    assert any("$.common_point_source_projection.records[0]" in violation for violation in result["violations"])
    _assert_common_shape(result)

def test_fail_with_schema_violating_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid_summary.json"
    payload = _valid_replay_summary_payload()
    del payload["replay_path"]
    _write_json(artifact, payload)
    exit_code, result = validate_replay_validation_summary(
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
    exit_code, payload = validate_replay_validation_summary(
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
    exit_code, payload = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=DEFAULT_SCHEMA,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("artifact invalid JSON" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_missing_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_summary.json"
    _write_json(artifact, _valid_replay_summary_payload())
    missing_schema = tmp_path / "does_not_exist_schema.json"
    exit_code, payload = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=missing_schema,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("schema unreadable" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_blocked_on_invalid_schema_json(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_summary.json"
    _write_json(artifact, _valid_replay_summary_payload())
    schema = tmp_path / "invalid_schema.json"
    schema.write_text("{not valid json", encoding="utf-8")
    exit_code, payload = validate_replay_validation_summary(
        artifact_path=artifact,
        schema_path=schema,
        validated_at="2026-03-09T20:00:00Z",
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("schema invalid JSON" in err for err in payload["errors"])
    _assert_common_shape(payload)


def test_deterministic_output_shape_and_stdout(tmp_path: Path) -> None:
    artifact = tmp_path / "valid_summary.json"
    _write_json(artifact, _valid_replay_summary_payload())
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
