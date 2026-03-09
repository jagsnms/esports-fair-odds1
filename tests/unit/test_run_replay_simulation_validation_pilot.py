from __future__ import annotations

import json
from pathlib import Path

import tools.run_replay_simulation_validation_pilot as pilot


def _summary(
    *,
    total_points_captured: int = 10,
    raw_contract_points: int = 10,
    unknown_replay_mode_points: int = 0,
    invariant_violations_total: int = 0,
    behavioral_violations_total: int = 0,
    p_hat_min: float = 0.41,
    p_hat_max: float = 0.58,
) -> dict[str, object]:
    return {
        "total_points_captured": total_points_captured,
        "raw_contract_points": raw_contract_points,
        "unknown_replay_mode_points": unknown_replay_mode_points,
        "invariant_violations_total": invariant_violations_total,
        "behavioral_violations_total": behavioral_violations_total,
        "p_hat_min": p_hat_min,
        "p_hat_max": p_hat_max,
    }


def test_build_pilot_decision_artifact_pass_when_local_and_cross_checks_hold() -> None:
    replay_summary = _summary()
    synthetic_summary = _summary(p_hat_min=0.42, p_hat_max=0.57)

    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_pass",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
    )

    assert artifact["decision"] == "pass"
    assert artifact["decision_reasons"] == []
    assert artifact["replay"]["local_sanity_pass"] is True
    assert artifact["synthetic"]["local_sanity_pass"] is True
    assert artifact["replay"]["raw_contract_coverage_rate"] == 1.0
    assert artifact["synthetic"]["raw_contract_coverage_rate"] == 1.0
    assert artifact["comparison"]["cross_surface_pass"] is True
    assert artifact["comparison"]["total_points_captured_abs_delta"] == 0
    assert artifact["comparison"]["failed_checks"] == []
    assert artifact["comparison"]["mismatch_class"] == "none"


def test_build_pilot_decision_artifact_mismatch_when_cross_surface_delta_exceeds_tolerance() -> None:
    replay_summary = _summary(p_hat_min=0.10, p_hat_max=0.50)
    synthetic_summary = _summary(p_hat_min=0.17, p_hat_max=0.62)

    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_mismatch",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
    )

    assert artifact["decision"] == "mismatch"
    assert artifact["replay"]["local_sanity_pass"] is True
    assert artifact["synthetic"]["local_sanity_pass"] is True
    assert artifact["comparison"]["cross_surface_pass"] is False
    assert any("abs(replay.p_hat_min - synthetic.p_hat_min)" in reason for reason in artifact["decision_reasons"])
    assert any("abs(replay.p_hat_max - synthetic.p_hat_max)" in reason for reason in artifact["decision_reasons"])
    assert artifact["comparison"]["total_points_captured_abs_delta"] == 0
    assert artifact["comparison"]["failed_checks"] == [
        "p_hat_min_abs_delta_lte_0_05",
        "p_hat_max_abs_delta_lte_0_05",
    ]
    assert artifact["comparison"]["mismatch_class"] == "cross_surface_behavioral_or_metric"


def test_build_pilot_decision_artifact_mismatch_when_total_points_captured_differs() -> None:
    replay_summary = _summary(total_points_captured=3, raw_contract_points=3)
    synthetic_summary = _summary(total_points_captured=49, raw_contract_points=49)

    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_volume_mismatch",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
    )

    assert artifact["decision"] == "mismatch"
    assert artifact["comparison"]["total_points_captured_abs_delta"] == 46
    assert artifact["comparison"]["failed_checks"] == ["total_points_captured_equality"]
    assert artifact["comparison"]["mismatch_class"] == "volume_alignment_only"
    assert any(
        "cross-surface mismatch: replay.total_points_captured must equal synthetic.total_points_captured exactly"
        == reason
        for reason in artifact["decision_reasons"]
    )


def test_build_pilot_decision_artifact_failed_checks_use_frozen_order_for_multiple_mismatches() -> None:
    replay_summary = _summary(
        total_points_captured=10,
        raw_contract_points=10,
        invariant_violations_total=0,
        behavioral_violations_total=0,
        p_hat_min=0.10,
        p_hat_max=0.20,
    )
    synthetic_summary = _summary(
        total_points_captured=12,
        raw_contract_points=12,
        invariant_violations_total=0,
        behavioral_violations_total=0,
        p_hat_min=0.25,
        p_hat_max=0.30,
    )

    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_multi_mismatch_order",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
    )

    assert artifact["decision"] == "mismatch"
    assert artifact["comparison"]["failed_checks"] == [
        "total_points_captured_equality",
        "p_hat_min_abs_delta_lte_0_05",
        "p_hat_max_abs_delta_lte_0_05",
    ]
    assert artifact["comparison"]["mismatch_class"] == "cross_surface_behavioral_or_metric"


def test_build_pilot_decision_artifact_inconclusive_when_required_field_unreadable() -> None:
    replay_summary = _summary(total_points_captured=0)
    synthetic_summary = _summary()

    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_inconclusive",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
    )

    assert artifact["decision"] == "inconclusive"
    assert artifact["replay"]["local_sanity_pass"] is False
    assert any("total_points_captured must be > 0" in reason for reason in artifact["decision_reasons"])
    assert artifact["comparison"]["failed_checks"] == []
    assert artifact["comparison"]["mismatch_class"] == "none"


def test_runner_writes_stable_artifact_for_fixed_inputs(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "pilot.json"

    def _fake_write_synthetic(path: Path, **_: object) -> int:
        path.write_text('{"dummy": true}\n', encoding="utf-8")
        return 1

    async def _fake_run_assessment(path: str, *, prematch_map: float | None = None) -> dict[str, object]:
        _ = prematch_map
        if "synthetic_raw_replay.jsonl" in path:
            return _summary(p_hat_min=0.43, p_hat_max=0.59)
        return _summary(p_hat_min=0.41, p_hat_max=0.58)

    monkeypatch.setattr(pilot, "write_synthetic_raw_replay_jsonl", _fake_write_synthetic)
    monkeypatch.setattr(pilot, "run_assessment", _fake_run_assessment)

    first = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(replay_input),
        run_id="pilot_stable",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        output_path=output_path,
    )
    second = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(replay_input),
        run_id="pilot_stable",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        output_path=output_path,
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.decision == "pass"
    assert second.decision == "pass"

    first_text = output_path.read_text(encoding="utf-8")
    second_text = output_path.read_text(encoding="utf-8")
    assert first_text == second_text

    payload = json.loads(first_text)
    assert payload["schema_version"] == "replay_simulation_validation_pilot.v1"
    assert payload["decision"] == "pass"
    assert payload["slice"]["replay_input_path"] == str(replay_input)
    assert payload["slice"]["synthetic_seed"] == 1337
    assert "total_points_captured_abs_delta" in payload["comparison"]
    assert "failed_checks" in payload["comparison"]
    assert "mismatch_class" in payload["comparison"]


def test_alignment_attempt_order_and_first_match_selection_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "pilot_alignment.json"

    def _fake_write_synthetic(path: Path, *, rounds: int, **_: object) -> int:
        path.write_text(json.dumps({"rounds": int(rounds)}), encoding="utf-8")
        return 1

    async def _fake_run_assessment(path: str, *, prematch_map: float | None = None) -> dict[str, object]:
        _ = prematch_map
        if "synthetic_raw_replay.jsonl" in path:
            rounds = int(json.loads(Path(path).read_text(encoding="utf-8"))["rounds"])
            synthetic_points_by_rounds = {
                1: 5,  # abs_delta=2
                2: 3,  # abs_delta=0 -> first acceptable candidate
                3: 8,
            }
            return _summary(
                total_points_captured=synthetic_points_by_rounds.get(rounds, 99),
                raw_contract_points=synthetic_points_by_rounds.get(rounds, 99),
            )
        return _summary(total_points_captured=3, raw_contract_points=3)

    monkeypatch.setattr(pilot, "write_synthetic_raw_replay_jsonl", _fake_write_synthetic)
    monkeypatch.setattr(pilot, "run_assessment", _fake_run_assessment)

    result = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(replay_input),
        run_id="pilot_alignment_order",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        output_path=output_path,
    )

    assert result.decision == "pass"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["alignment"]["target_replay_total_points"] == 3
    assert payload["alignment"]["attempted_synthetic_rounds"] == [1, 2]
    assert payload["alignment"]["selected_synthetic_rounds"] == 2
    assert payload["alignment"]["alignment_achieved"] is True
    assert payload["alignment"]["stop_reason"] is None
    assert payload["alignment"]["attempt_results"] == [
        {"attempted_rounds": 1, "synthetic_total_points": 5, "abs_delta": 2},
        {"attempted_rounds": 2, "synthetic_total_points": 3, "abs_delta": 0},
    ]


def test_alignment_stops_inconclusive_when_fixed_candidate_budget_cannot_align(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "pilot_alignment_inconclusive.json"

    def _fake_write_synthetic(path: Path, *, rounds: int, **_: object) -> int:
        path.write_text(json.dumps({"rounds": int(rounds)}), encoding="utf-8")
        return 1

    async def _fake_run_assessment(path: str, *, prematch_map: float | None = None) -> dict[str, object]:
        _ = prematch_map
        if "synthetic_raw_replay.jsonl" in path:
            rounds = int(json.loads(Path(path).read_text(encoding="utf-8"))["rounds"])
            synthetic_points_by_rounds = {
                2: 9,
                1: 8,
                3: 10,
            }
            return _summary(
                total_points_captured=synthetic_points_by_rounds.get(rounds, 99),
                raw_contract_points=synthetic_points_by_rounds.get(rounds, 99),
            )
        return _summary(total_points_captured=6, raw_contract_points=6)

    monkeypatch.setattr(pilot, "write_synthetic_raw_replay_jsonl", _fake_write_synthetic)
    monkeypatch.setattr(pilot, "run_assessment", _fake_run_assessment)

    result = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(replay_input),
        run_id="pilot_alignment_inconclusive",
        synthetic_seed=1337,
        synthetic_policy_profile="balanced_v1",
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
        output_path=output_path,
    )

    assert result.decision == "inconclusive"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "inconclusive"
    assert payload["decision_reasons"] == [pilot.ALIGNMENT_STOP_REASON]
    assert payload["comparison"]["failed_checks"] == []
    assert payload["comparison"]["mismatch_class"] == "none"
    assert payload["alignment"]["target_replay_total_points"] == 6
    assert payload["alignment"]["attempted_synthetic_rounds"] == [2, 1, 3]
    assert payload["alignment"]["selected_synthetic_rounds"] is None
    assert payload["alignment"]["alignment_achieved"] is False
    assert payload["alignment"]["stop_reason"] == pilot.ALIGNMENT_STOP_REASON
    assert payload["alignment"]["attempt_results"] == [
        {"attempted_rounds": 2, "synthetic_total_points": 9, "abs_delta": 3},
        {"attempted_rounds": 1, "synthetic_total_points": 8, "abs_delta": 2},
        {"attempted_rounds": 3, "synthetic_total_points": 10, "abs_delta": 4},
    ]
