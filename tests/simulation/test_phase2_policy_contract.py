"""Contract tests for the bounded Phase 2 policy-driven simulation harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.resolve import resolve_p_hat
from engine.simulation import phase2

SEED = 20260310


class TestSimulationPhase2PolicyContract(unittest.TestCase):
    def test_seeded_summary_is_deterministic(self) -> None:
        summary_a = phase2.generate_phase2_summary(SEED)
        summary_b = phase2.generate_phase2_summary(SEED)
        self.assertEqual(summary_a, summary_b)

    def test_canonical_engine_path_is_executed(self) -> None:
        with patch("engine.compute.bounds.compute_bounds", wraps=compute_bounds) as bounds_mock:
            with patch("engine.compute.rails.compute_rails", wraps=compute_rails) as rails_mock:
                with patch("engine.compute.resolve.resolve_p_hat", wraps=resolve_p_hat) as resolve_mock:
                    summary = phase2.generate_phase2_summary(SEED)

        replay_summary = summary["replay_comparable_summary"]
        self.assertEqual(summary["canonical_engine_path"], phase2.CANONICAL_ENGINE_PATH)
        self.assertGreater(bounds_mock.call_count, 0)
        self.assertEqual(bounds_mock.call_count, summary["generated_payload_count"])
        self.assertEqual(rails_mock.call_count, summary["generated_payload_count"])
        self.assertEqual(resolve_mock.call_count, summary["generated_payload_count"])
        self.assertGreaterEqual(replay_summary["raw_contract_points"], summary["generated_payload_count"])

    def test_summary_artifact_shape_and_safety(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        schema_path = repo_root / "tools" / "schemas" / "simulation_phase2_policy_summary.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "simulation_phase2_policy_summary.json"
            summary = phase2.emit_phase2_summary(SEED, output_path=output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary, written)
        for key in schema["required"]:
            self.assertIn(key, summary)

        self.assertEqual(summary["schema_version"], phase2.SIMULATION_PHASE2_SUMMARY_VERSION)
        self.assertEqual(summary["policy_profile"], phase2.PHASE2_STAGE1_POLICY_PROFILE)
        self.assertEqual(summary["round_count"], phase2.PHASE2_STAGE1_ROUNDS)
        self.assertEqual(summary["ticks_per_round"], phase2.PHASE2_STAGE1_TICKS_PER_ROUND)
        self.assertEqual(
            summary["generated_payload_count"],
            phase2.PHASE2_STAGE1_ROUNDS * phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
        )
        self.assertEqual(set(summary["policy_families"]), set(phase2.POLICY_FAMILIES))
        self.assertEqual(set(summary["realized_family_counts"].keys()), set(phase2.POLICY_FAMILIES))
        self.assertEqual(summary["structural_violations_total"], 0)
        self.assertEqual(summary["behavioral_violations_total"], 0)
        self.assertEqual(summary["invariant_violations_total"], 0)

        replay_summary = summary["replay_comparable_summary"]
        self.assertEqual(replay_summary["schema_version"], "replay_validation_summary.v1")
        self.assertEqual(replay_summary["fixture_class"], phase2.PHASE2_STAGE1_FIXTURE_CLASS)
        self.assertEqual(replay_summary["assessment_prematch_map"], phase2.PHASE2_STAGE2_PREMATCH_MAP)
        self.assertEqual(replay_summary["replay_path"], f"synthetic://phase2/balanced_v1/seed/{SEED}")
        self.assertIs(replay_summary["replay_path_exists"], False)
        self.assertEqual(replay_summary["total_points_captured"], replay_summary["raw_contract_points"])
        self.assertEqual(replay_summary["unknown_replay_mode_points"], 0)
        self.assertEqual(replay_summary["point_passthrough_points"], 0)
        self.assertGreater(replay_summary["rail_input_v2_activated_points"], 0)
        self.assertEqual(replay_summary["rail_input_v1_fallback_points"], 0)
        self.assertNotIn("V2_REQUIRED_FIELDS_MISSING", replay_summary["rail_input_reason_code_counts"])
        self.assertEqual(
            replay_summary["rail_input_reason_code_counts"].get("V2_STRICT_ACTIVATED"),
            replay_summary["total_points_captured"],
        )
        carryover = replay_summary["carryover_evidence_by_source_class"]["REPLAY_raw"]
        self.assertGreater(carryover["required_complete_points"], 0)
        self.assertEqual(carryover["required_incomplete_points"], 0)
        self.assertEqual(carryover["v2_activated_points"], replay_summary["total_points_captured"])
        self.assertEqual(carryover["v1_fallback_points"], 0)
        self.assertEqual(replay_summary["structural_violations_total"], 0)
        self.assertEqual(replay_summary["behavioral_violations_total"], 0)
        self.assertEqual(replay_summary["invariant_violations_total"], 0)
        self.assertGreater(replay_summary["p_hat_count"], 0)


if __name__ == "__main__":
    unittest.main()

