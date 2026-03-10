"""Contract tests for the seeded simulation Phase 1 harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.simulation import phase1

EXPECTED_FAMILIES = {
    "players_alive_trajectory",
    "loadout_regime_shift",
    "bomb_plant_countdown_transition",
    "near_zero_timer_pressure_ramp",
    "carryover_economy_transition",
}
SEED = 20260310


class TestSimulationPhase1Contract(unittest.TestCase):
    def test_seeded_summary_is_deterministic(self) -> None:
        summary_a = phase1.generate_phase1_summary(SEED)
        summary_b = phase1.generate_phase1_summary(SEED)
        self.assertEqual(summary_a, summary_b)

    def test_canonical_engine_path_is_executed(self) -> None:
        with patch("engine.simulation.phase1.compute_bounds", wraps=phase1.compute_bounds) as bounds_mock:
            with patch("engine.simulation.phase1.compute_rails", wraps=phase1.compute_rails) as rails_mock:
                with patch("engine.simulation.phase1.resolve_p_hat", wraps=phase1.resolve_p_hat) as resolve_mock:
                    summary = phase1.generate_phase1_summary(SEED)

        self.assertEqual(summary["canonical_engine_path"], phase1.CANONICAL_ENGINE_PATH)
        self.assertEqual(bounds_mock.call_count, summary["total_ticks_evaluated"])
        self.assertEqual(rails_mock.call_count, summary["total_ticks_evaluated"])
        self.assertEqual(resolve_mock.call_count, summary["total_ticks_evaluated"])

    def test_summary_artifact_shape_and_structural_safety(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        schema_path = repo_root / "tools" / "schemas" / "simulation_phase1_summary.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "simulation_phase1_summary.json"
            summary = phase1.emit_phase1_summary(SEED, output_path=output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary, written)
        for key in schema["required"]:
            self.assertIn(key, summary)

        self.assertEqual(summary["schema_version"], phase1.SIMULATION_PHASE1_SUMMARY_VERSION)
        self.assertEqual(set(summary["trajectory_family_counts"].keys()), EXPECTED_FAMILIES)
        self.assertEqual(set(summary["trajectory_tick_counts"].keys()), EXPECTED_FAMILIES)
        for family in EXPECTED_FAMILIES:
            self.assertEqual(summary["trajectory_family_counts"][family], 1)
            self.assertGreater(summary["trajectory_tick_counts"][family], 0)

        self.assertEqual(
            summary["total_ticks_evaluated"],
            sum(summary["trajectory_tick_counts"].values()),
        )
        self.assertEqual(summary["p_hat_count"], summary["total_ticks_evaluated"])
        self.assertEqual(summary["structural_violations_total"], 0)
        self.assertIsInstance(summary["p_hat_min"], float)
        self.assertIsInstance(summary["p_hat_max"], float)
        self.assertIsInstance(summary["rail_low_min"], float)
        self.assertIsInstance(summary["rail_low_max"], float)
        self.assertIsInstance(summary["rail_high_min"], float)
        self.assertIsInstance(summary["rail_high_max"], float)


if __name__ == "__main__":
    unittest.main()
