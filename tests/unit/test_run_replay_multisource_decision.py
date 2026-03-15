from __future__ import annotations

import json
from pathlib import Path

from engine.simulation import phase2
import tools.run_replay_multisource_decision as decision_tool
import tools.run_replay_simulation_validation_pilot as pilot


ROOT = Path(__file__).resolve().parents[2]


def _summary(
    *,
    total_points_captured: int = 10,
    raw_contract_points: int = 10,
    unknown_replay_mode_points: int = 0,
    invariant_violations_total: int = 0,
    behavioral_violations_total: int = 0,
    p_hat_min: float = 0.41,
    p_hat_max: float = 0.58,
    p_hat_count: int = 10,
    p_hat_median: float | None = None,
    fixture_class: str | None = None,
) -> dict[str, object]:
    median = float(p_hat_median) if p_hat_median is not None else float((p_hat_min + p_hat_max) / 2.0)
    payload: dict[str, object] = {
        "total_points_captured": total_points_captured,
        "raw_contract_points": raw_contract_points,
        "unknown_replay_mode_points": unknown_replay_mode_points,
        "invariant_violations_total": invariant_violations_total,
        "behavioral_violations_total": behavioral_violations_total,
        "p_hat_min": p_hat_min,
        "p_hat_max": p_hat_max,
        "p_hat_count": p_hat_count,
        "p_hat_median": median,
    }
    if fixture_class is not None:
        payload["fixture_class"] = fixture_class
    return payload


def _alignment_payload(
    *,
    target_replay_total_points: int = 10,
    candidate_totals: dict[int, int] | None = None,
    selected_rounds: int | None = phase2.PHASE2_STAGE1_ROUNDS,
) -> dict[str, object]:
    candidate_totals = candidate_totals or {32: 10}
    return {
        "target_replay_total_points": target_replay_total_points,
        "attempted_synthetic_rounds": list(candidate_totals.keys()),
        "attempt_results": [
            {
                "attempted_rounds": int(round_count),
                "synthetic_total_points": int(total_points),
                "abs_delta": abs(int(target_replay_total_points) - int(total_points)),
            }
            for round_count, total_points in candidate_totals.items()
        ],
        "selected_synthetic_rounds": selected_rounds,
        "alignment_achieved": selected_rounds is not None,
        "stop_reason": None if selected_rounds is not None else pilot.ALIGNMENT_STOP_REASON,
    }


def _pilot_artifact(
    *,
    policy_profile: str,
    replay_summary: dict[str, object],
    synthetic_summary: dict[str, object],
    selected_rounds: int | None = phase2.PHASE2_STAGE1_ROUNDS,
    alignment: dict[str, object] | None = None,
    force_inconclusive_reason: str | None = None,
) -> dict[str, object]:
    alignment_payload = alignment or _alignment_payload(selected_rounds=selected_rounds)
    return pilot.build_pilot_decision_artifact(
        run_id=f"artifact_{policy_profile}",
        replay_input_path="tools/fixtures/replay_carryover_complete_v1.jsonl",
        synthetic_seed=decision_tool.FIXED_SYNTHETIC_SEED,
        synthetic_policy_profile=policy_profile,
        synthetic_rounds=(selected_rounds if selected_rounds is not None else phase2.PHASE2_STAGE1_ROUNDS),
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-10T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
        synthetic_source_contract=phase2.phase2_source_contract(policy_profile),
        alignment=alignment_payload,
        force_inconclusive_reason=force_inconclusive_reason,
    )


def test_build_replay_multisource_decision_artifact_prefers_balanced_when_it_has_better_replay_fit() -> None:
    replay_summary = _summary(fixture_class="replay_fixture_class")
    balanced_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.42, p_hat_max=0.57),
    )
    eco_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.20, p_hat_max=0.70, p_hat_median=0.30),
    )

    artifact = decision_tool.build_replay_multisource_decision_artifact(
        run_id="decision_balanced_preferred",
        replay_input_path="tools/fixtures/replay_carryover_complete_v1.jsonl",
        generated_at="2026-03-10T00:00:00Z",
        replay_summary=replay_summary,
        balanced_artifact=balanced_artifact,
        eco_artifact=eco_artifact,
    )

    assert artifact["decision"]["decision"] == "balanced_preferred"
    assert artifact["decision"]["decision_basis"] == decision_tool.DECISION_BASIS
    assert artifact["replay_anchor"]["fixture_class"] == "replay_fixture_class"
    assert sorted(artifact["sources"].keys()) == ["balanced_v1", "eco_bias_v1"]
    assert artifact["sources"]["balanced_v1"]["refusal_class"] == pilot.REFUSAL_CLASS_NONE
    assert artifact["sources"]["balanced_v1"]["cross_surface_failed_checks"] == []
    assert artifact["sources"]["eco_bias_v1"]["cross_surface_failed_checks"] == [
        pilot.CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05,
        pilot.CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05,
        pilot.CHECK_ID_P_HAT_MEDIAN_ABS_DELTA_LTE_0_05,
    ]


def test_build_replay_multisource_decision_artifact_marks_no_material_difference_for_close_passes() -> None:
    replay_summary = _summary()
    balanced_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.42, p_hat_max=0.57, p_hat_median=0.495),
    )
    eco_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.421, p_hat_max=0.571, p_hat_median=0.496),
    )

    artifact = decision_tool.build_replay_multisource_decision_artifact(
        run_id="decision_no_material_difference",
        replay_input_path="tools/fixtures/replay_carryover_complete_v1.jsonl",
        generated_at="2026-03-10T00:00:00Z",
        replay_summary=replay_summary,
        balanced_artifact=balanced_artifact,
        eco_artifact=eco_artifact,
    )

    assert artifact["decision"]["decision"] == "no_material_difference"
    assert artifact["decision"]["no_material_difference_rule"] == decision_tool.NO_MATERIAL_DIFFERENCE_RULE
    assert artifact["decision"]["reasons"]


def test_equal_failed_check_count_with_different_failed_check_identities_does_not_collapse_to_no_material_difference() -> None:
    replay_summary = _summary()
    balanced_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.461, p_hat_max=0.628, p_hat_median=0.544),
    )
    eco_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(p_hat_min=0.459, p_hat_max=0.631, p_hat_median=0.544),
    )

    artifact = decision_tool.build_replay_multisource_decision_artifact(
        run_id="decision_incompatible_failed_checks",
        replay_input_path="tools/fixtures/replay_carryover_complete_v1.jsonl",
        generated_at="2026-03-10T00:00:00Z",
        replay_summary=replay_summary,
        balanced_artifact=balanced_artifact,
        eco_artifact=eco_artifact,
    )

    assert artifact["sources"]["balanced_v1"]["cross_surface_failed_checks"] == [
        pilot.CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05,
    ]
    assert artifact["sources"]["eco_bias_v1"]["cross_surface_failed_checks"] == [
        pilot.CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05,
    ]
    assert artifact["decision"]["decision"] == "eco_preferred"
    assert artifact["decision"]["decision"] != "no_material_difference"


def test_build_replay_multisource_decision_artifact_surfaces_specific_refusal_classes_for_unusable_sources() -> None:
    replay_summary = _summary(fixture_class="fixture_alpha")
    unaligned_alignment = _alignment_payload(
        target_replay_total_points=3,
        candidate_totals={32: 159, 31: 154, 33: 164, 30: 149, 34: 169},
        selected_rounds=None,
    )
    balanced_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(total_points_captured=159, raw_contract_points=159, p_hat_min=0.30, p_hat_max=0.64),
        selected_rounds=None,
        alignment=unaligned_alignment,
        force_inconclusive_reason=pilot.ALIGNMENT_STOP_REASON,
    )
    eco_artifact = _pilot_artifact(
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        replay_summary=replay_summary,
        synthetic_summary=_summary(total_points_captured=154, raw_contract_points=154, p_hat_min=0.31, p_hat_max=0.63),
        selected_rounds=None,
        alignment=unaligned_alignment,
        force_inconclusive_reason=pilot.ALIGNMENT_STOP_REASON,
    )

    artifact = decision_tool.build_replay_multisource_decision_artifact(
        run_id="decision_specific_refusal_classes",
        replay_input_path="tools/fixtures/replay_carryover_complete_v1.jsonl",
        generated_at="2026-03-10T00:00:00Z",
        replay_summary=replay_summary,
        balanced_artifact=balanced_artifact,
        eco_artifact=eco_artifact,
    )

    assert artifact["decision"]["decision"] == "inconclusive"
    assert artifact["sources"]["balanced_v1"]["refusal_class"] == (
        pilot.REFUSAL_CLASS_ALIGNMENT_NO_CANDIDATE_REPLAY_BELOW_CANDIDATE_FAMILY
    )
    assert artifact["sources"]["eco_bias_v1"]["refusal_class"] == (
        pilot.REFUSAL_CLASS_ALIGNMENT_NO_CANDIDATE_REPLAY_BELOW_CANDIDATE_FAMILY
    )
    assert artifact["decision"]["reasons"] == [
        "balanced_v1 source block is not replay-comparable enough for a two-source decision (alignment_no_candidate_replay_below_candidate_family)",
        "eco_bias_v1 source block is not replay-comparable enough for a two-source decision (alignment_no_candidate_replay_below_candidate_family)",
    ]


def test_runner_writes_stable_combined_two_source_artifact(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "multisource.json"

    replay_summary = _summary(fixture_class="fixture_alpha")
    calls: list[str] = []

    def _fake_load_replay_assessment_summary(replay_input_path: str) -> dict[str, object]:
        assert replay_input_path == str(replay_input)
        return replay_summary

    def _fake_evaluate_replay_against_canonical_phase2_source(**kwargs: object) -> pilot.PilotResult:
        policy_profile = str(kwargs["synthetic_policy_profile"])
        calls.append(policy_profile)
        synthetic_summary = (
            _summary(p_hat_min=0.42, p_hat_max=0.57, p_hat_median=0.495)
            if policy_profile == phase2.PHASE2_STAGE1_POLICY_PROFILE
            else _summary(p_hat_min=0.421, p_hat_max=0.571, p_hat_median=0.496)
        )
        artifact = _pilot_artifact(
            policy_profile=policy_profile,
            replay_summary=replay_summary,
            synthetic_summary=synthetic_summary,
        )
        return pilot.PilotResult(
            exit_code=0,
            decision=str(artifact["decision"]),
            output_path=None,
            message=f"ok {policy_profile}",
            artifact=artifact,
        )

    monkeypatch.setattr(decision_tool.pilot, "load_replay_assessment_summary", _fake_load_replay_assessment_summary)
    monkeypatch.setattr(
        decision_tool.pilot,
        "evaluate_replay_against_canonical_phase2_source",
        _fake_evaluate_replay_against_canonical_phase2_source,
    )

    kwargs = {
        "replay_input_path": str(replay_input),
        "run_id": "multisource_stable",
        "generated_at": "2026-03-10T00:00:00Z",
        "output_path": output_path,
    }
    first = decision_tool.run_replay_multisource_decision(**kwargs)
    first_text = output_path.read_text(encoding="utf-8")
    second = decision_tool.run_replay_multisource_decision(**kwargs)
    second_text = output_path.read_text(encoding="utf-8")

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.decision == "no_material_difference"
    assert second.decision == "no_material_difference"
    assert first_text == second_text
    assert calls == ["balanced_v1", "eco_bias_v1", "balanced_v1", "eco_bias_v1"]

    payload = json.loads(first_text)
    assert payload["decision"]["decision"] == "no_material_difference"
    assert payload["decision"]["decision_basis"] == decision_tool.DECISION_BASIS
    assert sorted(payload["sources"].keys()) == ["balanced_v1", "eco_bias_v1"]
    assert payload["sources"]["balanced_v1"]["synthetic_seed"] == decision_tool.FIXED_SYNTHETIC_SEED
    assert payload["sources"]["balanced_v1"]["refusal_class"] == pilot.REFUSAL_CLASS_NONE
    assert payload["sources"]["eco_bias_v1"]["synthetic_seed"] == decision_tool.FIXED_SYNTHETIC_SEED
    assert payload["sources"]["eco_bias_v1"]["refusal_class"] == pilot.REFUSAL_CLASS_NONE
    assert payload["replay_anchor"]["fixture_class"] == "fixture_alpha"
