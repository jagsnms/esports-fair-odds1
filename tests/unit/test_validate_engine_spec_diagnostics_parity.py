from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.validate_engine_spec_diagnostics_parity import validate_diagnostics_parity


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "tools" / "validate_engine_spec_diagnostics_parity.py"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _spec_payload(fields: list[str]) -> dict:
    return {
        "invariants": {
            "diagnostics_payload_required_fields": fields,
        }
    }


def _assessment_payload(fields: list[str], *, legacy_fields: list[str] | None = None) -> dict:
    return {
        "contract_diagnostics_spec_required_keys": fields,
        "contract_diagnostics_required_keys": fields if legacy_fields is None else legacy_fields,
    }


def test_pass_case_exact_parity_with_trimmed_spec_and_deterministic_order(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload([" q ", "rail_low", "rail_low", "timer_state"]))
    _write_json(assessment, _assessment_payload(["q", "rail_low", "timer_state", "extra_field"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert "validated_at" in payload
    assert payload["validated_at"] is None
    assert payload["parity_mode"] == "exact_name_path_only"
    assert payload["spec_required_fields"] == ["q", "rail_low", "timer_state"]
    assert payload["assessment_required_fields"] == ["q", "rail_low", "timer_state", "extra_field"]
    assert payload["matched_fields"] == ["q", "rail_low", "timer_state"]
    assert payload["missing_fields"] == []
    assert payload["extra_assessment_fields"] == ["extra_field"]
    assert payload["counts"] == {
        "spec_required_count": 3,
        "assessment_required_count": 4,
        "matched_count": 3,
        "missing_count": 0,
        "extra_count": 1,
    }
    assert payload["errors"] == []


def test_fail_case_with_one_missing_field(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["q", "rail_low", "timer_state"]))
    _write_json(assessment, _assessment_payload(["q", "rail_low"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 2
    assert payload["status"] == "fail"
    assert "validated_at" in payload
    assert payload["validated_at"] is None
    assert payload["matched_fields"] == ["q", "rail_low"]
    assert payload["missing_fields"] == [
        {
            "field": "timer_state",
            "expected_in": "contract_diagnostics_spec_required_keys",
            "match_mode": "exact_name_path_only",
            "reason": "missing_in_assessment_required_keys",
        }
    ]
    assert payload["counts"]["missing_count"] == 1


def test_fail_case_with_many_missing_fields(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["round_state_vector", "q", "p_hat", "timer_state"]))
    _write_json(assessment, _assessment_payload(["q"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 2
    assert payload["status"] == "fail"
    assert [entry["field"] for entry in payload["missing_fields"]] == [
        "round_state_vector",
        "p_hat",
        "timer_state",
    ]
    assert payload["counts"]["missing_count"] == 3
    assert payload["counts"]["matched_count"] == 1


def test_blocked_when_spec_missing_or_unreadable(tmp_path: Path) -> None:
    spec = tmp_path / "missing_ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(assessment, _assessment_payload(["q"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert "validated_at" in payload
    assert payload["validated_at"] is None
    assert any("engine spec unreadable" in err for err in payload["errors"])


def test_blocked_when_spec_invalid_json(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    spec.write_text("{not valid json", encoding="utf-8")
    _write_json(assessment, _assessment_payload(["q"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("engine spec invalid JSON" in err for err in payload["errors"])


def test_blocked_when_assessment_missing_or_unreadable(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "missing_assessment.json"
    _write_json(spec, _spec_payload(["q"]))

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("assessment artifact unreadable" in err for err in payload["errors"])


def test_blocked_when_assessment_invalid_json(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["q"]))
    assessment.write_text("{not valid json", encoding="utf-8")

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any("assessment artifact invalid JSON" in err for err in payload["errors"])


def test_blocked_when_new_assessment_field_missing_even_if_legacy_field_exists(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["q"]))
    _write_json(assessment, {"contract_diagnostics_required_keys": ["q"]})

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any(
        "missing/wrong-type path: contract_diagnostics_spec_required_keys" in err
        for err in payload["errors"]
    )


def test_blocked_when_new_assessment_field_wrong_type(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["q"]))
    _write_json(
        assessment,
        {
            "contract_diagnostics_spec_required_keys": "q",
            "contract_diagnostics_required_keys": ["q"],
        },
    )

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=spec,
        assessment_artifact_path=assessment,
    )
    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert any(
        "missing/wrong-type path: contract_diagnostics_spec_required_keys" in err
        for err in payload["errors"]
    )


def test_deterministic_stdout_output_ordering(tmp_path: Path) -> None:
    spec = tmp_path / "ENGINE_SPEC.json"
    assessment = tmp_path / "assessment.json"
    _write_json(spec, _spec_payload(["q", "rail_low", "timer_state"]))
    _write_json(assessment, _assessment_payload(["q", "rail_low", "timer_state"]))

    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--engine-spec-path",
        str(spec),
        "--assessment-artifact-path",
        str(assessment),
        "--validated-at",
        "2026-03-09T03:00:00Z",
    ]
    first = subprocess.run(cmd, check=False, capture_output=True, text=True)
    second = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    assert payload["schema_version"] == "engine_spec_diagnostics_parity.v1"
    assert payload["status"] == "pass"
    assert payload["validated_at"] == "2026-03-09T03:00:00Z"
    assert list(payload.keys()) == sorted(payload.keys())
