#!/usr/bin/env python3
"""
Emit one thin machine-readable comparison between the two approved canonical Phase 2
sources: balanced_v1 and eco_bias_v1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.simulation import phase2

SCHEMA_VERSION = "phase2_source_comparison.v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "automation"
    / "reports"
    / "phase2_source_comparison_balanced_v1_vs_eco_bias_v1_seed20260310.json"
)


@dataclass(frozen=True)
class ComparisonResult:
    artifact: dict[str, Any]
    output_path: Path | None


def _now_iso_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_block(summary: dict[str, Any]) -> dict[str, Any]:
    replay = summary["replay_comparable_summary"]
    trace = summary["trace_export"]
    return {
        "policy_profile": summary["policy_profile"],
        "canonical_source_contract": summary["canonical_source_contract"],
        "trace_source_contract": trace["canonical_source_contract"],
        "seed": summary["seed"],
        "round_count": summary["round_count"],
        "ticks_per_round": summary["ticks_per_round"],
        "comparison_basis": {
            "assessment_path": "canonical replay_comparable_summary",
            "trace_path": "labeled-point-only trace_export",
            "label_scope": trace["pairing_rule"]["label_event_type"],
        },
        "safety_floor": {
            "structural_violations_total": replay["structural_violations_total"],
            "behavioral_violations_total": replay["behavioral_violations_total"],
            "invariant_violations_total": replay["invariant_violations_total"],
            "preserved": (
                replay["structural_violations_total"] == 0
                and replay["behavioral_violations_total"] == 0
                and replay["invariant_violations_total"] == 0
            ),
        },
        "policy_distribution": {
            "realized_family_counts": dict(summary["realized_family_counts"]),
            "family_sequence": list(summary["policy_distribution"]["family_sequence"]),
        },
        "trace_export": {
            "labeled_prediction_record_count": trace["labeled_prediction_record_count"],
            "unlabeled_prediction_points_excluded": trace["unlabeled_prediction_points_excluded"],
            "round_result_event_count": trace["round_result_event_count"],
        },
        "replay_comparable_summary": {
            "total_points_captured": replay["total_points_captured"],
            "raw_contract_points": replay["raw_contract_points"],
            "rail_input_v2_activated_points": replay["rail_input_v2_activated_points"],
            "rail_input_v1_fallback_points": replay["rail_input_v1_fallback_points"],
            "p_hat_min": replay["p_hat_min"],
            "p_hat_max": replay["p_hat_max"],
            "p_hat_median": replay["p_hat_median"],
        },
    }


def build_phase2_source_comparison(
    *,
    seed: int = 20260310,
    generated_at: str | None = None,
    output_path: str | Path | None = None,
) -> ComparisonResult:
    balanced = phase2.generate_phase2_summary(
        seed,
        policy_profile=phase2.PHASE2_STAGE1_POLICY_PROFILE,
    )
    eco_bias = phase2.generate_phase2_summary(
        seed,
        policy_profile=phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
    )

    balanced_block = _source_block(balanced)
    eco_bias_block = _source_block(eco_bias)
    balanced_counts = balanced_block["policy_distribution"]["realized_family_counts"]
    eco_counts = eco_bias_block["policy_distribution"]["realized_family_counts"]
    family_count_deltas = {
        family: int(eco_counts[family]) - int(balanced_counts[family])
        for family in phase2.POLICY_FAMILIES
    }
    comparison = {
        "source_identity_explicit": True,
        "source_vs_source_not_baseline_current": True,
        "same_seed": balanced["seed"] == eco_bias["seed"],
        "same_shape": (
            balanced["round_count"] == eco_bias["round_count"]
            and balanced["ticks_per_round"] == eco_bias["ticks_per_round"]
        ),
        "same_trace_export_rule": (
            balanced["trace_export"]["pairing_rule"] == eco_bias["trace_export"]["pairing_rule"]
        ),
        "same_labeled_point_only_rule": (
            balanced["trace_export"]["unlabeled_prediction_points_excluded"]
            == eco_bias["trace_export"]["unlabeled_prediction_points_excluded"]
        ),
        "same_carryover_complete_floor": (
            balanced_block["safety_floor"]["preserved"] and eco_bias_block["safety_floor"]["preserved"]
        ),
        "policy_family_count_deltas_eco_minus_balanced": family_count_deltas,
        "replay_metric_deltas_eco_minus_balanced": {
            "total_points_captured_delta": int(
                eco_bias_block["replay_comparable_summary"]["total_points_captured"]
            )
            - int(balanced_block["replay_comparable_summary"]["total_points_captured"]),
            "p_hat_min_delta": float(eco_bias_block["replay_comparable_summary"]["p_hat_min"])
            - float(balanced_block["replay_comparable_summary"]["p_hat_min"]),
            "p_hat_max_delta": float(eco_bias_block["replay_comparable_summary"]["p_hat_max"])
            - float(balanced_block["replay_comparable_summary"]["p_hat_max"]),
            "p_hat_median_delta": float(eco_bias_block["replay_comparable_summary"]["p_hat_median"])
            - float(balanced_block["replay_comparable_summary"]["p_hat_median"]),
            "labeled_prediction_record_count_delta": int(
                eco_bias_block["trace_export"]["labeled_prediction_record_count"]
            )
            - int(balanced_block["trace_export"]["labeled_prediction_record_count"]),
            "unlabeled_prediction_points_excluded_delta": int(
                eco_bias_block["trace_export"]["unlabeled_prediction_points_excluded"]
            )
            - int(balanced_block["trace_export"]["unlabeled_prediction_points_excluded"]),
        },
        "decision_pressure": {
            "basis": "same seed/shape and same truthful trace rule, but different canonical source definition",
            "family_distribution_changed": any(delta != 0 for delta in family_count_deltas.values()),
            "p_hat_median_abs_delta": abs(
                float(eco_bias_block["replay_comparable_summary"]["p_hat_median"])
                - float(balanced_block["replay_comparable_summary"]["p_hat_median"])
            ),
            "pressure_present": any(delta != 0 for delta in family_count_deltas.values()),
        },
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso_z(),
        "comparison_basis": {
            "type": "canonical_phase2_source_vs_source",
            "seed": int(seed),
            "round_count": phase2.PHASE2_STAGE1_ROUNDS,
            "ticks_per_round": phase2.PHASE2_STAGE1_TICKS_PER_ROUND,
            "left_policy_profile": phase2.PHASE2_STAGE1_POLICY_PROFILE,
            "right_policy_profile": phase2.PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        },
        "left_source": balanced_block,
        "right_source": eco_bias_block,
        "comparison": comparison,
    }

    destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ComparisonResult(artifact=artifact, output_path=destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a bounded machine-readable comparison between balanced_v1 and eco_bias_v1."
    )
    parser.add_argument("--seed", type=int, default=20260310, help="Explicit deterministic seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Optional JSON output path.",
    )
    parser.add_argument("--generated-at", type=str, default=None, help="Optional explicit generated_at timestamp.")
    args = parser.parse_args()

    result = build_phase2_source_comparison(
        seed=args.seed,
        generated_at=args.generated_at,
        output_path=args.output,
    )
    print(json.dumps(result.artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
