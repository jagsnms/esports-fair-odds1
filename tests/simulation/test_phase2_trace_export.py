from __future__ import annotations

from engine.simulation import phase2

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
