from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tools.replay_verification_assess import run_assessment


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools" / "schemas" / "replay_validation_summary.schema.json"
FIXTURE_PATH = ROOT / "tools" / "fixtures" / "replay_multimatch_small_v1.jsonl"


def _assert_schema_conformance(summary: dict, schema: dict) -> None:
    required = schema.get("required", [])
    for key in required:
        assert key in summary, f"missing required summary key: {key}"

    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in summary:
            continue
        value = summary[key]
        expected_type = spec.get("type")
        if expected_type == "integer":
            assert isinstance(value, int), f"{key} must be integer"
            assert value >= spec.get("minimum", value), f"{key} below minimum"
        elif expected_type == "string":
            assert isinstance(value, str), f"{key} must be string"
            if "minLength" in spec:
                assert len(value) >= int(spec["minLength"]), f"{key} below minLength"
            if "const" in spec:
                assert value == spec["const"], f"{key} must equal {spec['const']}"
        elif expected_type == "boolean":
            assert isinstance(value, bool), f"{key} must be boolean"
        elif expected_type == "object":
            assert isinstance(value, dict), f"{key} must be object"
        elif expected_type == "array":
            assert isinstance(value, list), f"{key} must be array"
        elif isinstance(expected_type, list):
            allowed = tuple(
                float if t == "number" else type(None) if t == "null" else object
                for t in expected_type
            )
            assert any(isinstance(value, t) for t in allowed), f"{key} must match one of {expected_type}"


def test_replay_verification_assess_stage1_deterministic_and_schema_conformant() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    first = asyncio.run(run_assessment(str(FIXTURE_PATH)))
    second = asyncio.run(run_assessment(str(FIXTURE_PATH)))

    _assert_schema_conformance(first, schema)
    _assert_schema_conformance(second, schema)

    assert first == second, "stage1 replay summary must be deterministic for same fixture"
    assert first["fixture_class"] == "replay_multimatch_small_v1"
    assert first["replay_contract_policy"] == "reject_point_like"
    assert first["replay_point_transition_enabled"] is False
    assert first["direct_load_payload_count"] == 6
    assert first["raw_contract_points"] == first["total_points_captured"]
    assert first["point_passthrough_points"] == 0
    assert first["point_like_inputs_seen"] == 0
    assert first["point_like_inputs_rejected"] == 0
    assert first["point_like_inputs_transition_passthrough"] == 0
    assert first["point_like_reject_reason_counts"] == {}
    assert first["points_with_contract_diagnostics"] == first["total_points_captured"]
    required_keys = first["contract_diagnostics_required_keys"]
    assert isinstance(required_keys, list)
    assert len(required_keys) > 0
    for key in required_keys:
        assert first["contract_diagnostics_key_presence_counts"][key] == first["points_with_contract_diagnostics"]
        assert first["contract_diagnostics_missing_key_counts"][key] == 0
        assert first["contract_diagnostics_key_presence_rates"][key] == 1.0
