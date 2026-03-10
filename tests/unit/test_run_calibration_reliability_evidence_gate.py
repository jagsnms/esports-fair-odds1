from __future__ import annotations

import json
from pathlib import Path

from tools.run_calibration_reliability_evidence_gate import (
    OUTPUT_PREFIX,
    run_evidence_gate_runner,
)


def _bins(a: float, b: float) -> list[dict]:
    return [
        {"bin_index": 0, "count": 4, "mean_prediction": a, "empirical_rate": b},
        {"bin_index": 1, "count": 6, "mean_prediction": a + 0.1, "empirical_rate": b + 0.1},
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _pass_replay_records() -> list[dict]:
    return [
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.25,
            "log_loss": 0.65,
            "reliability_curve_bins": _bins(0.52, 0.49),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            "log_loss": 0.62,
            "reliability_curve_bins": _bins(0.53, 0.51),
        },
    ]


def _pass_simulation_records() -> list[dict]:
    return [
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.24,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "current",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.23,
            "log_loss": 0.63,
            "reliability_curve_bins": _bins(0.52, 0.50),
        },
    ]


def test_pass_case_writes_artifact_and_returns_zero(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"
    _write_json(replay_path, _pass_replay_records())
    _write_json(simulation_path, _pass_simulation_records())

    result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_001",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )

    assert result.exit_code == 0
    assert result.gate_status == "pass"
    assert result.output_path == reports_dir / f"{OUTPUT_PREFIX}run_001.json"
    assert result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "pass"
    assert payload["generated_at"] == "2026-03-08T01:00:00Z"


def test_incomplete_evidence_writes_artifact_and_returns_nonzero(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"
    replay = _pass_replay_records()
    simulation = _pass_simulation_records()
    del simulation[1]["log_loss"]  # Missing required metric -> incomplete_evidence
    _write_json(replay_path, replay)
    _write_json(simulation_path, simulation)

    result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_002",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )

    assert result.exit_code != 0
    assert result.gate_status == "incomplete_evidence"
    assert result.output_path is not None and result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "incomplete_evidence"


def test_fail_case_writes_artifact_and_returns_nonzero(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"
    _write_json(
        replay_path,
        [
            {
                "evidence_source": "replayish",  # invalid enum -> fail
                "dataset_id": "bad",
                "evaluation_scope": "baseline",
                "seed": None,
                "segment": None,
                "brier_score": 0.24,
                "log_loss": 0.64,
                "reliability_curve_bins": _bins(0.5, 0.49),
            }
        ],
    )
    _write_json(simulation_path, [])

    result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_003",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )

    assert result.exit_code != 0
    assert result.gate_status == "fail"
    assert result.output_path is not None and result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "fail"


def test_missing_or_invalid_inputs_fail_process_without_artifact(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"

    missing_file_result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_004",
        reports_dir=reports_dir,
    )
    assert missing_file_result.exit_code == 1
    assert missing_file_result.output_path is None

    replay_path.write_text("{not-json", encoding="utf-8")
    simulation_path.write_text("[]", encoding="utf-8")
    invalid_json_result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_004",
        reports_dir=reports_dir,
    )
    assert invalid_json_result.exit_code == 1
    assert invalid_json_result.output_path is None

    replay_path.write_text("{}", encoding="utf-8")  # wrong top-level type
    invalid_top_level_result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_004",
        reports_dir=reports_dir,
    )
    assert invalid_top_level_result.exit_code == 1
    assert invalid_top_level_result.output_path is None

    assert not reports_dir.exists()


def test_generated_at_supplied_makes_output_deterministic(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"
    _write_json(replay_path, _pass_replay_records())
    _write_json(simulation_path, _pass_simulation_records())

    first = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_005a",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )
    second = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="run_005b",
        generated_at="2026-03-08T01:00:00Z",
        reports_dir=reports_dir,
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.output_path is not None and second.output_path is not None
    first_text = first.output_path.read_text(encoding="utf-8")
    second_text = second.output_path.read_text(encoding="utf-8")
    assert first_text == second_text


def test_invalid_run_id_is_rejected(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    simulation_path = tmp_path / "simulation.json"
    reports_dir = tmp_path / "reports"
    _write_json(replay_path, _pass_replay_records())
    _write_json(simulation_path, _pass_simulation_records())

    result = run_evidence_gate_runner(
        replay_input_path=str(replay_path),
        simulation_input_path=str(simulation_path),
        baseline_ref="baseline:sha1",
        current_ref="current:sha2",
        run_id="Run 005",
        reports_dir=reports_dir,
    )

    assert result.exit_code == 1
    assert result.output_path is None
    assert "invalid run_id" in result.message
