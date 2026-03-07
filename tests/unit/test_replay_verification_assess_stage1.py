from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tools.replay_verification_assess import run_assessment
from engine.compute.rails_cs2 import (
    RAIL_INPUT_POLICY_V2_STRICT,
    V2_ACTIVATED,
    V2_FALLBACK_REQUIRED_MISSING,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools" / "schemas" / "replay_validation_summary.schema.json"
SPARSE_MULTIMATCH_FIXTURE_PATH = ROOT / "tools" / "fixtures" / "replay_multimatch_small_v1.jsonl"
SPARSE_RAW_FIXTURE_PATH = ROOT / "tools" / "fixtures" / "raw_replay_sample.jsonl"
CARRYOVER_COMPLETE_FIXTURE_PATH = ROOT / "tools" / "fixtures" / "replay_carryover_complete_v1.jsonl"
POINT_LIKE_FIXTURE_PATH = ROOT / "logs" / "history_points.jsonl"


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


def _run_deterministic(summary_path: Path, *, prematch_map: float | None = None) -> tuple[dict, dict]:
    first = asyncio.run(run_assessment(str(summary_path), prematch_map=prematch_map))
    second = asyncio.run(run_assessment(str(summary_path), prematch_map=prematch_map))
    return first, second


def test_sparse_fallback_classes_remain_deterministic_and_fallback_only() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    for fixture in (SPARSE_MULTIMATCH_FIXTURE_PATH, SPARSE_RAW_FIXTURE_PATH):
        first, second = _run_deterministic(fixture)
        _assert_schema_conformance(first, schema)
        _assert_schema_conformance(second, schema)

        assert first == second, "replay summary must be deterministic for the same sparse fixture"
        assert first["fixture_class"] == fixture.stem
        assert first["assessment_prematch_map"] is None
        assert first["replay_contract_policy"] == "reject_point_like"
        assert first["replay_point_transition_enabled"] is False
        assert first["raw_contract_points"] == first["total_points_captured"]
        assert first["point_passthrough_points"] == 0
        assert first["rail_input_contract_policy_counts"].get(RAIL_INPUT_POLICY_V2_STRICT, 0) == first["total_points_captured"]
        assert first["rail_input_v2_activated_points"] == 0
        assert first["rail_input_v1_fallback_points"] == first["total_points_captured"]
        assert first["rail_input_reason_code_counts"] == {V2_FALLBACK_REQUIRED_MISSING: first["total_points_captured"]}
        assert first["rail_input_active_endpoint_semantics_counts"].get("v1", 0) == first["total_points_captured"]
        assert first["rail_input_active_endpoint_semantics_counts"].get("v2", 0) == 0
        assert first["point_like_inputs_seen"] == 0
        assert first["point_like_inputs_rejected"] == 0
        assert first["point_like_inputs_transition_passthrough"] == 0
        assert first["point_like_reject_reason_counts"] == {}
        assert first["points_with_contract_diagnostics"] == first["total_points_captured"]
        for key in first["contract_diagnostics_required_keys"]:
            assert first["contract_diagnostics_key_presence_counts"][key] == first["points_with_contract_diagnostics"]
            assert first["contract_diagnostics_missing_key_counts"][key] == 0
            assert first["contract_diagnostics_key_presence_rates"][key] == 1.0


def test_carryover_complete_fixture_activates_v2_with_valid_prematch_map() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    first, second = _run_deterministic(CARRYOVER_COMPLETE_FIXTURE_PATH, prematch_map=0.55)
    _assert_schema_conformance(first, schema)
    _assert_schema_conformance(second, schema)

    assert first == second, "carryover-complete fixture summary must be deterministic"
    assert first["fixture_class"] == "replay_carryover_complete_v1"
    assert first["assessment_prematch_map"] == 0.55
    assert first["raw_contract_points"] == first["total_points_captured"]
    assert first["point_passthrough_points"] == 0
    assert first["total_points_captured"] > 0
    assert first["rail_input_v2_activated_points"] > 0
    assert first["rail_input_v1_fallback_points"] == 0
    assert first["rail_input_reason_code_counts"] == {V2_ACTIVATED: first["total_points_captured"]}
    assert first["rail_input_active_endpoint_semantics_counts"].get("v2", 0) == first["total_points_captured"]
    assert first["rail_input_active_endpoint_semantics_counts"].get("v1", 0) == 0
    assert first["structural_violations_total"] == 0
    assert first["invariant_violations_total"] == 0
    assert first["behavioral_violations_total"] == 0


def test_point_like_replay_is_rejected_and_excluded_from_activation_denominator() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    first, second = _run_deterministic(POINT_LIKE_FIXTURE_PATH)
    _assert_schema_conformance(first, schema)
    _assert_schema_conformance(second, schema)

    assert first == second, "point-like replay summary must be deterministic"
    assert first["fixture_class"] == "history_points"
    assert first["total_points_captured"] == 0
    assert first["raw_contract_points"] == 0
    assert first["point_passthrough_points"] == 0
    assert first["rail_input_v2_activated_points"] == 0
    assert first["rail_input_v1_fallback_points"] == 0
    assert first["rail_input_reason_code_counts"] == {}
    assert first["point_like_inputs_seen"] > 0
    assert first["point_like_inputs_rejected"] == first["point_like_inputs_seen"]
    assert first["point_like_inputs_transition_passthrough"] == 0
    assert first["points_with_contract_diagnostics"] == 0


def test_carryover_evidence_by_source_class_present_and_structured() -> None:
    """Assessment output includes carryover_evidence_by_source_class with activation/fallback/completeness."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    first, _ = _run_deterministic(SPARSE_RAW_FIXTURE_PATH)
    _assert_schema_conformance(first, schema)
    assert "carryover_evidence_by_source_class" in first
    assert "carryover_completeness_required_fields" in first
    assert isinstance(first["carryover_completeness_required_fields"], list)
    assert len(first["carryover_completeness_required_fields"]) > 0
    by_src = first["carryover_evidence_by_source_class"]
    assert isinstance(by_src, dict)
    # Sparse raw fixture should have REPLAY_raw with fallback only
    key = "REPLAY_raw"
    assert key in by_src, f"expected key {key} in carryover_evidence_by_source_class"
    rec = by_src[key]
    assert rec["points"] == first["total_points_captured"]
    assert rec["v1_fallback_points"] == rec["points"]
    assert rec["v2_activated_points"] == 0
    assert rec["required_incomplete_points"] == rec["points"]
    assert rec["required_complete_points"] == 0


def test_carryover_complete_fixture_source_class_shows_activation() -> None:
    """Carryover-complete fixture has REPLAY_raw with v2 activated and required_complete."""
    first, _ = _run_deterministic(CARRYOVER_COMPLETE_FIXTURE_PATH, prematch_map=0.55)
    assert "carryover_evidence_by_source_class" in first
    by_src = first["carryover_evidence_by_source_class"]
    key = "REPLAY_raw"
    assert key in by_src
    rec = by_src[key]
    assert rec["points"] == first["total_points_captured"]
    assert rec["v2_activated_points"] == rec["points"]
    assert rec["v1_fallback_points"] == 0
    assert rec["required_complete_points"] == rec["points"]
    assert rec["required_incomplete_points"] == 0
