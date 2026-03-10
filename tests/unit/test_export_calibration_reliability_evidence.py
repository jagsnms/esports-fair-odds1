from __future__ import annotations

import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from engine.simulation import phase2
from tools.export_calibration_reliability_evidence import (
    SIMULATION_DATASET_ID,
    SIMULATION_EXPORT_STATUS,
    SIMULATION_POLICY_PROFILE,
    SIMULATION_REQUIRED_SEED,
    run_export,
)
from tools.run_calibration_reliability_evidence_gate import run_evidence_gate_runner

SEED = SIMULATION_REQUIRED_SEED


def _source_report() -> dict:
    bins_before = [
        {"bin": 0, "n": 4, "p_mean": 0.2, "y_rate": 0.25},
        {"bin": 1, "n": 6, "p_mean": 0.7, "y_rate": 0.66},
    ]
    bins_after = [
        {"bin": 0, "n": 5, "p_mean": 0.22, "y_rate": 0.2},
        {"bin": 1, "n": 5, "p_mean": 0.68, "y_rate": 0.6},
    ]
    return {
        "ran_at": "2026-03-10T00:00:00Z",
        "games": {
            "cs2": {
                "metrics_before": {"brier": 0.24, "logloss": 0.64},
                "metrics_after": {"brier": 0.22, "logloss": 0.61},
                "reliability_before": bins_before,
                "reliability_after": bins_after,
            },
        },
    }


@lru_cache(maxsize=1)
def _trace_summary() -> dict:
    return phase2.generate_phase2_summary(SEED)


def _write_trace_input(path: Path) -> None:
    path.write_text(json.dumps(_trace_summary(), sort_keys=True), encoding="utf-8")


class TestExportCalibrationReliabilityEvidence(unittest.TestCase):
    def test_exporter_emits_bounded_true_simulation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_report_path = root / "report.json"
            replay_output_path = root / "replay.json"
            simulation_output_path = root / "simulation.json"
            manifest_output_path = root / "manifest.json"
            baseline_trace_path = root / "baseline_trace.json"
            current_trace_path = root / "current_trace.json"
            source_report_path.write_text(json.dumps(_source_report(), sort_keys=True), encoding="utf-8")
            _write_trace_input(baseline_trace_path)
            _write_trace_input(current_trace_path)

            result = run_export(
                source_report_path=source_report_path,
                source_calibration_path=None,
                replay_output_path=replay_output_path,
                simulation_output_path=simulation_output_path,
                manifest_output_path=manifest_output_path,
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="export_bounded_true_sim",
                generated_at="2026-03-10T01:00:00Z",
                simulation_seed=SEED,
                simulation_baseline_trace_path=baseline_trace_path,
                simulation_current_trace_path=current_trace_path,
            )

            self.assertEqual(result.exit_code, 0)
            replay_records = json.loads(replay_output_path.read_text(encoding="utf-8"))
            simulation_records = json.loads(simulation_output_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_output_path.read_text(encoding="utf-8"))

            self.assertEqual(len(replay_records), 2)
            self.assertEqual({row["evidence_source"] for row in replay_records}, {"replay"})
            self.assertEqual(len(simulation_records), 2)
            self.assertEqual({row["evidence_source"] for row in simulation_records}, {"simulation"})
            self.assertEqual({row["dataset_id"] for row in simulation_records}, {SIMULATION_DATASET_ID})
            self.assertEqual({row["seed"] for row in simulation_records}, {SEED})
            self.assertEqual({row["segment"] for row in simulation_records}, {"global"})
            self.assertEqual(
                {row["evaluation_scope"] for row in simulation_records},
                {"baseline", "current"},
            )
            self.assertTrue(all(len(row["reliability_curve_bins"]) == 10 for row in simulation_records))
            self.assertEqual(
                manifest["extraction_parameters"]["simulation_export_status"],
                SIMULATION_EXPORT_STATUS,
            )
            self.assertEqual(
                manifest["extraction_parameters"]["simulation_policy_profile"],
                SIMULATION_POLICY_PROFILE,
            )
            self.assertEqual(manifest["outputs"]["simulation_record_count"], 2)
            self.assertEqual(
                manifest["simulation_trace_inputs"]["baseline_trace"]["unlabeled_prediction_points_excluded"],
                phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
            )
            self.assertEqual(
                manifest["simulation_trace_inputs"]["current_trace"]["labeled_prediction_record_count"],
                (phase2.PHASE2_STAGE1_ROUNDS - 1) * phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
            )

    def test_exported_bounded_simulation_records_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_report_path = root / "report.json"
            replay_output_path = root / "replay.json"
            simulation_output_path = root / "simulation.json"
            manifest_output_path = root / "manifest.json"
            reports_dir = root / "reports"
            baseline_trace_path = root / "baseline_trace.json"
            current_trace_path = root / "current_trace.json"
            source_report_path.write_text(json.dumps(_source_report(), sort_keys=True), encoding="utf-8")
            _write_trace_input(baseline_trace_path)
            _write_trace_input(current_trace_path)

            export_result = run_export(
                source_report_path=source_report_path,
                source_calibration_path=None,
                replay_output_path=replay_output_path,
                simulation_output_path=simulation_output_path,
                manifest_output_path=manifest_output_path,
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="export_then_gate_true_sim",
                generated_at="2026-03-10T01:00:00Z",
                simulation_seed=SEED,
                simulation_baseline_trace_path=baseline_trace_path,
                simulation_current_trace_path=current_trace_path,
            )
            self.assertEqual(export_result.exit_code, 0)

            gate_result = run_evidence_gate_runner(
                replay_input_path=str(replay_output_path),
                simulation_input_path=str(simulation_output_path),
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="gate_bounded_simulation",
                generated_at="2026-03-10T01:00:00Z",
                reports_dir=reports_dir,
            )

            self.assertEqual(gate_result.exit_code, 0)
            self.assertEqual(gate_result.gate_status, "pass")
            self.assertIsNotNone(gate_result.summary)
            self.assertEqual(gate_result.summary["incomplete_reasons"], [])


if __name__ == "__main__":
    unittest.main()
