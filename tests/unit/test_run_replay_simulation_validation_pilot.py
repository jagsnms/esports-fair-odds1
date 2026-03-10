from __future__ import annotations

import json
from pathlib import Path

from engine.simulation import phase2
import tools.run_replay_simulation_validation_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
FIRST_SLICE_FIXTURE = ROOT / "tools" / "fixtures" / "raw_replay_sample.jsonl"
SECOND_SLICE_FIXTURE = ROOT / "tools" / "fixtures" / "replay_carryover_complete_v1.jsonl"


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
) -> dict[str, object]:
    median = float(p_hat_median) if p_hat_median is not None else float((p_hat_min + p_hat_max) / 2.0)
    return {
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


def _canonical_phase2_artifact(*, seed: int, replay_summary: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": pilot.SIMULATION_PHASE2_SUMMARY_VERSION,
        "seed": int(seed),
        "policy_profile": phase2.PHASE2_STAGE1_POLICY_PROFILE,
        "round_count": phase2.PHASE2_STAGE1_ROUNDS,
        "ticks_per_round": phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        "replay_comparable_summary": replay_summary,
    }


def _assert_decision_contract_coherence(payload: dict[str, object]) -> None:
    decision = payload["decision"]
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    failed_checks = comparison["failed_checks"]
    mismatch_class = comparison["mismatch_class"]
    decision_reasons = payload["decision_reasons"]

    if decision == "pass":
        assert failed_checks == []
        assert mismatch_class == "none"
        assert decision_reasons == []
        return
    if decision == "mismatch":
        assert isinstance(failed_checks, list)
        assert len(failed_checks) > 0
        if failed_checks == ["total_points_captured_abs_delta_lte_1"]:
            assert mismatch_class == "volume_alignment_only"
        else:
            assert mismatch_class == "cross_surface_behavioral_or_metric"
        assert isinstance(decision_reasons, list) and len(decision_reasons) > 0
        return
    if decision == "inconclusive":
        assert failed_checks == []
        assert mismatch_class == "none"
        assert isinstance(decision_reasons, list) and len(decision_reasons) > 0
        return
    raise AssertionError(f"unexpected decision: {decision!r}")


def test_build_pilot_decision_artifact_pass_when_local_and_cross_checks_hold() -> None:
    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_pass",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=phase2.PHASE2_STAGE1_ROUNDS,
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=_summary(),
        synthetic_summary=_summary(p_hat_min=0.42, p_hat_max=0.57),
    )

    assert artifact["decision"] == "pass"
    assert artifact["decision_reasons"] == []
    assert artifact["comparison"]["cross_surface_pass"] is True
    assert artifact["comparison"]["failed_checks"] == []
    assert artifact["slice"]["synthetic_source_contract"] == pilot.CANONICAL_PHASE2_SOURCE_CONTRACT
    assert artifact["slice"]["synthetic_source_schema_version"] == pilot.SIMULATION_PHASE2_SUMMARY_VERSION


def test_build_pilot_decision_artifact_mismatch_when_cross_surface_delta_exceeds_tolerance() -> None:
    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_mismatch",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=phase2.PHASE2_STAGE1_ROUNDS,
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=_summary(p_hat_min=0.10, p_hat_max=0.50),
        synthetic_summary=_summary(p_hat_min=0.17, p_hat_max=0.62),
    )

    assert artifact["decision"] == "mismatch"
    assert artifact["comparison"]["failed_checks"] == [
        "p_hat_min_abs_delta_lte_0_05",
        "p_hat_max_abs_delta_lte_0_05",
        "p_hat_median_abs_delta_lte_0_05",
    ]
    assert artifact["comparison"]["mismatch_class"] == "cross_surface_behavioral_or_metric"


def test_build_pilot_decision_artifact_inconclusive_when_p_hat_count_is_insufficient_for_fingerprint() -> None:
    artifact = pilot.build_pilot_decision_artifact(
        run_id="pilot_p_hat_count_insufficient",
        replay_input_path="tools/fixtures/raw_replay_sample.jsonl",
        synthetic_seed=1337,
        synthetic_policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=phase2.PHASE2_STAGE1_ROUNDS,
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-09T00:00:00Z",
        replay_summary=_summary(p_hat_count=2),
        synthetic_summary=_summary(p_hat_count=10),
    )

    assert artifact["decision"] == "inconclusive"
    assert artifact["comparison"]["failed_checks"] == []
    assert artifact["decision_reasons"] == [pilot.TRAJECTORY_FINGERPRINT_INSUFFICIENT_P_HAT_COUNT_REASON]


def test_runner_writes_stable_artifact_for_canonical_phase2_inputs(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "pilot.json"

    seeds_seen: list[int] = []

    def _fake_generate_phase2_summary(seed: int) -> dict[str, object]:
        seeds_seen.append(int(seed))
        return _canonical_phase2_artifact(
            seed=seed,
            replay_summary=_summary(p_hat_min=0.43, p_hat_max=0.59),
        )

    async def _fake_run_assessment(path: str, *, prematch_map: float | None = None) -> dict[str, object]:
        _ = path
        _ = prematch_map
        return _summary(p_hat_min=0.41, p_hat_max=0.58)

    monkeypatch.setattr(pilot, "generate_phase2_summary", _fake_generate_phase2_summary)
    monkeypatch.setattr(pilot, "run_assessment", _fake_run_assessment)

    kwargs = {
        "replay_input_path": str(replay_input),
        "run_id": "pilot_stable",
        "synthetic_seed": 1337,
        "synthetic_policy_profile": phase2.PHASE2_STAGE1_POLICY_PROFILE,
        "synthetic_rounds": phase2.PHASE2_STAGE1_ROUNDS,
        "synthetic_ticks_per_round": phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        "generated_at": "2026-03-09T00:00:00Z",
        "output_path": output_path,
    }
    first = pilot.run_replay_simulation_validation_pilot(**kwargs)
    first_text = output_path.read_text(encoding="utf-8")
    second = pilot.run_replay_simulation_validation_pilot(**kwargs)
    second_text = output_path.read_text(encoding="utf-8")

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.decision == "pass"
    assert second.decision == "pass"
    assert first_text == second_text
    assert seeds_seen == [1337, 1337]

    payload = json.loads(first_text)
    _assert_decision_contract_coherence(payload)
    assert payload["slice"]["replay_input_path"] == str(replay_input)
    assert payload["slice"]["synthetic_seed"] == 1337
    assert payload["slice"]["synthetic_policy_profile"] == phase2.PHASE2_STAGE1_POLICY_PROFILE
    assert payload["slice"]["synthetic_rounds"] == phase2.PHASE2_STAGE1_ROUNDS
    assert payload["slice"]["synthetic_ticks_per_round"] == phase2.PHASE2_STAGE1_TICKS_PER_ROUND
    assert payload["slice"]["synthetic_source_contract"] == pilot.CANONICAL_PHASE2_SOURCE_CONTRACT
    assert payload["slice"]["synthetic_source_schema_version"] == pilot.SIMULATION_PHASE2_SUMMARY_VERSION
    assert payload["synthetic"]["p_hat_min"] == 0.43
    assert payload["synthetic"]["p_hat_max"] == 0.59
    assert payload["alignment"]["attempted_synthetic_rounds"] == [phase2.PHASE2_STAGE1_ROUNDS]
    assert payload["alignment"]["selected_synthetic_rounds"] == phase2.PHASE2_STAGE1_ROUNDS


def test_runner_rejects_noncanonical_profile() -> None:
    result = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(FIRST_SLICE_FIXTURE),
        run_id="pilot_bad_profile",
        synthetic_seed=1337,
        synthetic_policy_profile="eco_bias_v1",
        synthetic_rounds=phase2.PHASE2_STAGE1_ROUNDS,
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-09T00:00:00Z",
    )

    assert result.exit_code == 1
    assert result.decision == "inconclusive"
    assert "canonical Phase 2 binding only supports balanced_v1" in result.message


def test_runner_rejects_noncanonical_shape() -> None:
    result = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(FIRST_SLICE_FIXTURE),
        run_id="pilot_bad_shape",
        synthetic_seed=1337,
        synthetic_policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=10,
        synthetic_ticks_per_round=4,
        generated_at="2026-03-09T00:00:00Z",
    )

    assert result.exit_code == 1
    assert result.decision == "inconclusive"
    assert "canonical Phase 2 binding only supports rounds=32, ticks_per_round=4" in result.message


def test_runner_marks_unaligned_canonical_fixed_slice_inconclusive(tmp_path: Path, monkeypatch) -> None:
    replay_input = tmp_path / "replay.jsonl"
    replay_input.write_text('{"dummy": true}\n', encoding="utf-8")
    output_path = tmp_path / "pilot_unaligned.json"

    def _fake_generate_phase2_summary(seed: int) -> dict[str, object]:
        return _canonical_phase2_artifact(
            seed=seed,
            replay_summary=_summary(total_points_captured=159, raw_contract_points=159, p_hat_min=0.30, p_hat_max=0.64),
        )

    async def _fake_run_assessment(path: str, *, prematch_map: float | None = None) -> dict[str, object]:
        _ = path
        _ = prematch_map
        return _summary(total_points_captured=3, raw_contract_points=3, p_hat_min=0.51, p_hat_max=0.53)

    monkeypatch.setattr(pilot, "generate_phase2_summary", _fake_generate_phase2_summary)
    monkeypatch.setattr(pilot, "run_assessment", _fake_run_assessment)

    result = pilot.run_replay_simulation_validation_pilot(
        replay_input_path=str(replay_input),
        run_id="pilot_unaligned_fixed_slice",
        synthetic_seed=20260310,
        synthetic_policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=phase2.PHASE2_STAGE1_ROUNDS,
        synthetic_ticks_per_round=phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        generated_at="2026-03-09T00:00:00Z",
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.exit_code == 1
    assert result.decision == "inconclusive"
    assert payload["decision"] == "inconclusive"
    assert payload["decision_reasons"] == [pilot.CANONICAL_PHASE2_FIXED_SLICE_NOTE]
    assert payload["alignment"]["alignment_achieved"] is False
    assert payload["alignment"]["selected_synthetic_rounds"] is None
    assert payload["alignment"]["stop_reason"] == pilot.CANONICAL_PHASE2_FIXED_SLICE_NOTE
    assert payload["slice"]["synthetic_source_contract"] == pilot.CANONICAL_PHASE2_SOURCE_CONTRACT


def test_runner_uses_canonical_phase2_slice_for_real_fixture_deterministically(tmp_path: Path) -> None:
    output_path = tmp_path / "slice2_pilot.json"
    kwargs = {
        "replay_input_path": str(SECOND_SLICE_FIXTURE),
        "run_id": "replay_sim_pilot_slice2",
        "synthetic_seed": 20260310,
        "synthetic_policy_profile": phase2.PHASE2_STAGE1_POLICY_PROFILE,
        "synthetic_rounds": phase2.PHASE2_STAGE1_ROUNDS,
        "synthetic_ticks_per_round": phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        "generated_at": "2026-03-09T00:00:00Z",
        "output_path": output_path,
    }

    first = pilot.run_replay_simulation_validation_pilot(**kwargs)
    first_payload = json.loads(output_path.read_text(encoding="utf-8"))
    _assert_decision_contract_coherence(first_payload)

    second = pilot.run_replay_simulation_validation_pilot(**kwargs)
    second_payload = json.loads(output_path.read_text(encoding="utf-8"))
    _assert_decision_contract_coherence(second_payload)

    assert first.decision == second.decision
    assert first_payload == second_payload
    assert first_payload["slice"]["synthetic_seed"] == 20260310
    assert first_payload["slice"]["synthetic_policy_profile"] == phase2.PHASE2_STAGE1_POLICY_PROFILE
    assert first_payload["slice"]["synthetic_source_contract"] == pilot.CANONICAL_PHASE2_SOURCE_CONTRACT
    assert first_payload["slice"]["synthetic_source_schema_version"] == pilot.SIMULATION_PHASE2_SUMMARY_VERSION
    assert first_payload["alignment"]["attempted_synthetic_rounds"] == [phase2.PHASE2_STAGE1_ROUNDS]
    if first_payload["alignment"]["alignment_achieved"]:
        assert first_payload["alignment"]["selected_synthetic_rounds"] == phase2.PHASE2_STAGE1_ROUNDS
        assert first_payload["decision"] in ("pass", "mismatch")
    else:
        assert first_payload["alignment"]["selected_synthetic_rounds"] is None
        assert first_payload["decision"] == "inconclusive"
        assert first_payload["decision_reasons"] == [pilot.CANONICAL_PHASE2_FIXED_SLICE_NOTE]
        assert first_payload["alignment"]["stop_reason"] == pilot.CANONICAL_PHASE2_FIXED_SLICE_NOTE
    assert first_payload["synthetic"]["invariant_violations_total"] == 0
    assert first_payload["synthetic"]["behavioral_violations_total"] == 0

