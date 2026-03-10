from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.export_calibration_reliability_evidence import run_export
from tools.run_calibration_reliability_evidence_gate import run_evidence_gate_runner


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
            "valorant": {
                "metrics_before": {"brier": 0.13, "logloss": 0.44},
                "metrics_after": {"brier": 0.10, "logloss": 0.33},
                "reliability_before": bins_before,
                "reliability_after": bins_after,
            },
        },
    }


class TestExportCalibrationReliabilityEvidence(unittest.TestCase):
    def test_exporter_disables_fake_simulation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_report_path = root / "report.json"
            replay_output_path = root / "replay.json"
            simulation_output_path = root / "simulation.json"
            manifest_output_path = root / "manifest.json"
            source_report_path.write_text(json.dumps(_source_report(), sort_keys=True), encoding="utf-8")

            result = run_export(
                source_report_path=source_report_path,
                source_calibration_path=None,
                replay_output_path=replay_output_path,
                simulation_output_path=simulation_output_path,
                manifest_output_path=manifest_output_path,
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="export_disable_fake_sim",
                generated_at="2026-03-10T01:00:00Z",
                simulation_seed=1337,
            )

            self.assertEqual(result.exit_code, 0)
            replay_records = json.loads(replay_output_path.read_text(encoding="utf-8"))
            simulation_records = json.loads(simulation_output_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_output_path.read_text(encoding="utf-8"))

            self.assertEqual(len(replay_records), 2)
            self.assertEqual({row["evidence_source"] for row in replay_records}, {"replay"})
            self.assertEqual(simulation_records, [])
            self.assertEqual(manifest["extraction_parameters"]["game_to_surface_mapping"], {"cs2": "replay"})
            self.assertEqual(
                manifest["extraction_parameters"]["simulation_export_status"],
                "disabled_no_true_simulation_source",
            )
            self.assertEqual(manifest["outputs"]["simulation_record_count"], 0)

    def test_exported_absence_of_simulation_surfaces_as_incomplete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_report_path = root / "report.json"
            replay_output_path = root / "replay.json"
            simulation_output_path = root / "simulation.json"
            manifest_output_path = root / "manifest.json"
            reports_dir = root / "reports"
            source_report_path.write_text(json.dumps(_source_report(), sort_keys=True), encoding="utf-8")

            export_result = run_export(
                source_report_path=source_report_path,
                source_calibration_path=None,
                replay_output_path=replay_output_path,
                simulation_output_path=simulation_output_path,
                manifest_output_path=manifest_output_path,
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="export_then_gate",
                generated_at="2026-03-10T01:00:00Z",
                simulation_seed=1337,
            )
            self.assertEqual(export_result.exit_code, 0)

            gate_result = run_evidence_gate_runner(
                replay_input_path=str(replay_output_path),
                simulation_input_path=str(simulation_output_path),
                baseline_ref="baseline:report",
                current_ref="current:report",
                run_id="gate_missing_simulation",
                generated_at="2026-03-10T01:00:00Z",
                reports_dir=reports_dir,
            )

            self.assertNotEqual(gate_result.exit_code, 0)
            self.assertEqual(gate_result.gate_status, "incomplete_evidence")
            self.assertIsNotNone(gate_result.summary)
            reasons = set(gate_result.summary["incomplete_reasons"])
            self.assertIn("required slice missing: source=simulation, scope=baseline", reasons)
            self.assertIn("required slice missing: source=simulation, scope=current", reasons)


if __name__ == "__main__":
    unittest.main()