from __future__ import annotations

import json
from pathlib import Path

from engine.simulation import phase2
from tools.compare_phase2_sources import SCHEMA_VERSION, build_phase2_source_comparison

SEED = 20260310


def test_phase2_trace_export_is_deterministic() -> None:
    first = phase2.generate_phase2_summary(SEED)["trace_export"]
    second = phase2.generate_phase2_summary(SEED)["trace_export"]
    assert first == second


def test_phase2_trace_export_contract_and_pairing_truthfulness() -> None:
    trace = phase2.generate_phase2_summary(SEED)["trace_export"]

    assert trace["schema_version"] == phase2.SIMULATION_PHASE2_TRACE_SCHEMA_VERSION
    assert trace["seed"] == SEED
    assert trace["policy_profile"] == phase2.PHASE2_STAGE1_POLICY_PROFILE
    assert trace["round_count"] == phase2.PHASE2_STAGE1_ROUNDS
    assert trace["ticks_per_round"] == phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    assert trace["canonical_source_contract"] == phase2.PHASE2_TRACE_SOURCE_CONTRACT
    assert trace["pairing_rule"] == {
        "id": phase2.PHASE2_TRACE_PAIRING_RULE_ID,
        "label_event_type": phase2.PHASE2_TRACE_LABEL_EVENT_TYPE,
        "join_keys": list(phase2.PHASE2_TRACE_JOIN_KEYS),
        "export_condition": phase2.PHASE2_TRACE_EXPORT_CONDITION,
    }
    assert trace["common_point_source_basis"] == phase2.common_point_source_basis_descriptor()
    assert trace["common_point_source_basis"]["shared_fields"] == [
        "p_hat",
        "rail_low",
        "rail_high",
        "game_number",
        "map_index",
        "round_number",
    ]
    assert trace["common_point_source_basis"]["contract_limits"] == {
        "shared_field_subset_only": True,
        "record_matching_implied": False,
        "alignment_implied": False,
        "scoring_or_selection_implied": False,
    }
    assert "point_index_in_round" not in trace["common_point_source_basis"]["shared_fields"]
    assert "label_scope" not in trace["common_point_source_basis"]["shared_fields"]
    assert "round_winner_team_id" not in trace["common_point_source_basis"]["shared_fields"]
    assert "round_winner_is_team_a" not in trace["common_point_source_basis"]["shared_fields"]

    assert trace["total_prediction_points_seen"] == (
        phase2.PHASE2_STAGE1_ROUNDS * phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    )
    assert trace["round_result_event_count"] == phase2.PHASE2_STAGE1_ROUNDS - 1
    assert trace["labeled_prediction_record_count"] == (
        (phase2.PHASE2_STAGE1_ROUNDS - 1) * phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    )
    assert trace["unlabeled_prediction_points_excluded"] == phase2.PHASE2_STAGE1_TICKS_PER_ROUND

    records = trace["trace_records"]
    assert len(records) == trace["labeled_prediction_record_count"]
    assert {record["label_scope"] for record in records} == {phase2.PHASE2_TRACE_LABEL_SCOPE}
    assert {record["round_number"] for record in records} == set(range(1, phase2.PHASE2_STAGE1_ROUNDS))

    per_round_labels: dict[tuple[int, int, int], tuple[int, bool]] = {}
    per_round_indexes: dict[tuple[int, int, int], list[int]] = {}
    for record in records:
        key = (record["game_number"], record["map_index"], record["round_number"])
        assert record["game_number"] == 1
        assert record["map_index"] == 0
        assert 0 <= record["point_index_in_round"] < phase2.PHASE2_STAGE1_TICKS_PER_ROUND
        assert 0.0 <= record["p_hat"] <= 1.0
        assert 0.0 <= record["rail_low"] <= record["rail_high"] <= 1.0
        label = (record["round_winner_team_id"], record["round_winner_is_team_a"])
        if key in per_round_labels:
            assert per_round_labels[key] == label
        else:
            per_round_labels[key] = label
        per_round_indexes.setdefault(key, []).append(record["point_index_in_round"])

    for indexes in per_round_indexes.values():
        assert sorted(indexes) == list(range(phase2.PHASE2_STAGE1_TICKS_PER_ROUND))


def test_eco_bias_trace_export_preserves_same_truthful_rules() -> None:
    trace = phase2.generate_phase2_summary(
        SEED,
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
    )["trace_export"]

    assert trace["schema_version"] == phase2.SIMULATION_PHASE2_TRACE_SCHEMA_VERSION
    assert trace["seed"] == SEED
    assert trace["policy_profile"] == phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE
    assert trace["round_count"] == phase2.PHASE2_STAGE1_ROUNDS
    assert trace["ticks_per_round"] == phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    assert trace["canonical_source_contract"] == phase2.phase2_trace_source_contract(
        phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE
    )
    assert trace["pairing_rule"] == {
        "id": phase2.PHASE2_TRACE_PAIRING_RULE_ID,
        "label_event_type": phase2.PHASE2_TRACE_LABEL_EVENT_TYPE,
        "join_keys": list(phase2.PHASE2_TRACE_JOIN_KEYS),
        "export_condition": phase2.PHASE2_TRACE_EXPORT_CONDITION,
    }
    assert trace["common_point_source_basis"] == phase2.common_point_source_basis_descriptor()
    assert trace["total_prediction_points_seen"] == (
        phase2.PHASE2_STAGE1_ROUNDS * phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    )
    assert trace["round_result_event_count"] == phase2.PHASE2_STAGE1_ROUNDS - 1
    assert trace["labeled_prediction_record_count"] == (
        (phase2.PHASE2_STAGE1_ROUNDS - 1) * phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    )
    assert trace["unlabeled_prediction_points_excluded"] == phase2.PHASE2_STAGE1_TICKS_PER_ROUND


def test_phase2_source_comparison_artifact_is_explicit_and_machine_readable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        repo_root
        / "automation"
        / "reports"
        / "phase2_source_comparison_balanced_v1_vs_eco_bias_v1_seed20260310.json"
    )
    result = build_phase2_source_comparison(
        seed=SEED,
        generated_at="2026-03-10T00:00:00Z",
        output_path=output_path,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    artifact = result.artifact
    assert artifact == written
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["comparison_basis"] == {
        "type": "canonical_phase2_source_vs_source",
        "seed": SEED,
        "round_count": phase2.PHASE2_STAGE1_ROUNDS,
        "ticks_per_round": phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        "left_policy_profile": phase2.PHASE2_STAGE1_POLICY_PROFILE,
        "right_policy_profile": phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
    }
    assert artifact["left_source"]["policy_profile"] == phase2.PHASE2_STAGE1_POLICY_PROFILE
    assert artifact["right_source"]["policy_profile"] == phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE
    assert artifact["left_source"]["canonical_source_contract"] == phase2.phase2_source_contract(
        phase2.PHASE2_STAGE1_POLICY_PROFILE
    )
    assert artifact["right_source"]["canonical_source_contract"] == phase2.phase2_source_contract(
        phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE
    )

    comparison = artifact["comparison"]
    assert comparison["source_identity_explicit"] is True
    assert comparison["source_vs_source_not_baseline_current"] is True
    assert comparison["same_seed"] is True
    assert comparison["same_shape"] is True
    assert comparison["same_trace_export_rule"] is True
    assert comparison["same_labeled_point_only_rule"] is True
    assert comparison["same_carryover_complete_floor"] is True
    assert comparison["decision_pressure"]["family_distribution_changed"] is True
    assert comparison["decision_pressure"]["pressure_present"] is True
    assert any(
        delta != 0 for delta in comparison["policy_family_count_deltas_eco_minus_balanced"].values()
    )
