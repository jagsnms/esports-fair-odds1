#!/usr/bin/env python3
"""Run one bounded replay-anchored two-source decision contract."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from engine.simulation.phase2 import (
        PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        PHASE2_STAGE1_POLICY_PROFILE,
        PHASE2_STAGE1_ROUNDS,
        PHASE2_STAGE1_TICKS_PER_ROUND,
    )
    import tools.run_replay_simulation_validation_pilot as pilot
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from engine.simulation.phase2 import (  # type: ignore[no-redef]
        PHASE2_SECOND_SOURCE_POLICY_PROFILE,
        PHASE2_STAGE1_POLICY_PROFILE,
        PHASE2_STAGE1_ROUNDS,
        PHASE2_STAGE1_TICKS_PER_ROUND,
    )
    import run_replay_simulation_validation_pilot as pilot  # type: ignore[no-redef]


SCHEMA_VERSION = "replay_multisource_decision.v1"
DECISION_CONTRACT_VERSION = "replay_anchored_multi_source_decision.v1"
DECISION_BASIS = "replay_anchored_multi_source"
OUTPUT_PREFIX = "replay_multisource_decision_"
DEFAULT_REPORTS_DIR = ROOT / "automation" / "reports"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
FIXED_SYNTHETIC_SEED = 20260310
NO_MATERIAL_DIFFERENCE_DELTA_THRESHOLD = 0.01
NO_MATERIAL_DIFFERENCE_RULE = (
    "emit no_material_difference when both aligned source blocks have equal failed_check_count and each key replay-vs-source absolute delta differs by <= 0.01"
)
SOURCE_POLICY_PROFILES = (
    PHASE2_STAGE1_POLICY_PROFILE,
    PHASE2_SECOND_SOURCE_POLICY_PROFILE,
)
OUTPUT_FILENAME = (
    f"{OUTPUT_PREFIX}{PHASE2_STAGE1_POLICY_PROFILE}_vs_{PHASE2_SECOND_SOURCE_POLICY_PROFILE}_seed{FIXED_SYNTHETIC_SEED}.json"
)


@dataclass(frozen=True)
class MultiSourceDecisionResult:
    exit_code: int
    decision: str
    output_path: Path | None
    message: str
    artifact: dict[str, Any] | None = None


def _require_run_id(run_id: str) -> str:
    run_id_norm = str(run_id or "")
    if not run_id_norm or RUN_ID_PATTERN.fullmatch(run_id_norm) is None:
        raise ValueError("invalid run_id: must match [a-z0-9_-]+ and be non-empty")
    return run_id_norm


def _comparison_metrics(source_payload: dict[str, Any]) -> dict[str, float | None]:
    comparison = source_payload["comparison"]
    return {
        "total_points_captured_abs_delta": None if comparison["total_points_captured_abs_delta"] is None else float(comparison["total_points_captured_abs_delta"]),
        "p_hat_min_abs_delta": None if comparison["p_hat_min_abs_delta"] is None else float(comparison["p_hat_min_abs_delta"]),
        "p_hat_max_abs_delta": None if comparison["p_hat_max_abs_delta"] is None else float(comparison["p_hat_max_abs_delta"]),
        "p_hat_median_abs_delta": None if comparison["p_hat_median_abs_delta"] is None else float(comparison["p_hat_median_abs_delta"]),
    }


def _source_block(source_payload: dict[str, Any]) -> dict[str, Any]:
    slice_block = source_payload["slice"]
    comparison = source_payload["comparison"]
    replay = source_payload["replay"]
    synthetic = source_payload["synthetic"]
    alignment = source_payload["alignment"]
    return {
        "policy_profile": slice_block["synthetic_policy_profile"],
        "canonical_source_contract": slice_block["synthetic_source_contract"],
        "synthetic_seed": slice_block["synthetic_seed"],
        "fixed_shape_basis": {
            "requested_rounds": PHASE2_STAGE1_ROUNDS,
            "requested_ticks_per_round": PHASE2_STAGE1_TICKS_PER_ROUND,
            "round_candidates": list(pilot.CANONICAL_PHASE2_ROUND_CANDIDATES),
            "selected_rounds": alignment["selected_synthetic_rounds"],
        },
        "local_sanity_result": {
            "replay_pass": replay["local_sanity_pass"],
            "replay_reasons": replay["local_sanity_reasons"],
            "synthetic_pass": synthetic["local_sanity_pass"],
            "synthetic_reasons": synthetic["local_sanity_reasons"],
        },
        "alignment_result": alignment,
        "source_decision": source_payload["decision"],
        "source_decision_reasons": source_payload["decision_reasons"],
        "cross_surface_failed_checks": comparison["failed_checks"],
        "key_replay_vs_source_deltas": _comparison_metrics(source_payload),
        "cross_surface_pass": comparison["cross_surface_pass"],
        "cross_surface_reasons": comparison["cross_surface_reasons"],
    }


def _usable_source_block(source_block: dict[str, Any]) -> bool:
    local_sanity = source_block["local_sanity_result"]
    alignment = source_block["alignment_result"]
    return (
        local_sanity["replay_pass"] is True
        and local_sanity["synthetic_pass"] is True
        and alignment["alignment_achieved"] is True
        and source_block["source_decision"] in ("pass", "mismatch")
    )


def _severity_tuple(source_block: dict[str, Any]) -> tuple[float, float, float, float, float]:
    deltas = source_block["key_replay_vs_source_deltas"]
    return (
        float(len(source_block["cross_surface_failed_checks"])),
        float("inf") if deltas["p_hat_median_abs_delta"] is None else float(deltas["p_hat_median_abs_delta"]),
        float("inf") if deltas["p_hat_min_abs_delta"] is None else float(deltas["p_hat_min_abs_delta"]),
        float("inf") if deltas["p_hat_max_abs_delta"] is None else float(deltas["p_hat_max_abs_delta"]),
        float("inf") if deltas["total_points_captured_abs_delta"] is None else float(deltas["total_points_captured_abs_delta"]),
    )


def _all_metric_differences_within_tie_threshold(
    left_block: dict[str, Any],
    right_block: dict[str, Any],
) -> bool:
    left = left_block["key_replay_vs_source_deltas"]
    right = right_block["key_replay_vs_source_deltas"]
    keys = (
        "total_points_captured_abs_delta",
        "p_hat_min_abs_delta",
        "p_hat_max_abs_delta",
        "p_hat_median_abs_delta",
    )
    for key in keys:
        if left[key] is None or right[key] is None:
            return False
        if abs(float(left[key]) - float(right[key])) > NO_MATERIAL_DIFFERENCE_DELTA_THRESHOLD:
            return False
    return True


def build_replay_multisource_decision_artifact(
    *,
    run_id: str,
    replay_input_path: str,
    generated_at: str,
    replay_summary: Any,
    balanced_artifact: dict[str, Any],
    eco_artifact: dict[str, Any],
) -> dict[str, Any]:
    replay_anchor = {
        "replay_input_path": replay_input_path,
        "fixture_class": replay_summary.get("fixture_class") if isinstance(replay_summary, dict) else None,
        "generated_at": generated_at,
    }
    source_blocks = {
        PHASE2_STAGE1_POLICY_PROFILE: _source_block(balanced_artifact),
        PHASE2_SECOND_SOURCE_POLICY_PROFILE: _source_block(eco_artifact),
    }
    balanced_block = source_blocks[PHASE2_STAGE1_POLICY_PROFILE]
    eco_block = source_blocks[PHASE2_SECOND_SOURCE_POLICY_PROFILE]
    reasons: list[str] = []

    if not _usable_source_block(balanced_block) or not _usable_source_block(eco_block):
        decision = "inconclusive"
        if not _usable_source_block(balanced_block):
            reasons.append("balanced_v1 source block is not replay-comparable enough for a two-source decision")
        if not _usable_source_block(eco_block):
            reasons.append("eco_bias_v1 source block is not replay-comparable enough for a two-source decision")
    else:
        balanced_failed = len(balanced_block["cross_surface_failed_checks"])
        eco_failed = len(eco_block["cross_surface_failed_checks"])
        if balanced_failed == eco_failed and _all_metric_differences_within_tie_threshold(balanced_block, eco_block):
            decision = "no_material_difference"
            reasons.append(
                f"both sources have failed_check_count={balanced_failed} and all key replay-vs-source deltas stay within the no-material-difference threshold {NO_MATERIAL_DIFFERENCE_DELTA_THRESHOLD:.2f}"
            )
        else:
            balanced_severity = _severity_tuple(balanced_block)
            eco_severity = _severity_tuple(eco_block)
            if balanced_severity < eco_severity:
                decision = "balanced_preferred"
                reasons.append(
                    f"balanced_v1 outranks eco_bias_v1 on replay anchoring with severity tuple {balanced_severity} < {eco_severity}"
                )
            elif eco_severity < balanced_severity:
                decision = "eco_preferred"
                reasons.append(
                    f"eco_bias_v1 outranks balanced_v1 on replay anchoring with severity tuple {eco_severity} < {balanced_severity}"
                )
            else:
                decision = "no_material_difference"
                reasons.append("both sources produced identical replay-anchored severity tuples")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision_contract_version": DECISION_CONTRACT_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "replay_anchor": replay_anchor,
        "sources": source_blocks,
        "decision": {
            "decision": decision,
            "decision_basis": DECISION_BASIS,
            "reasons": reasons,
            "no_material_difference_rule": NO_MATERIAL_DIFFERENCE_RULE,
        },
    }


def run_replay_multisource_decision(
    *,
    replay_input_path: str,
    run_id: str,
    generated_at: str | None = None,
    output_path: Path | None = None,
) -> MultiSourceDecisionResult:
    try:
        run_id_norm = _require_run_id(run_id)
    except ValueError as exc:
        return MultiSourceDecisionResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=str(exc),
            artifact=None,
        )

    replay_input = Path(replay_input_path)
    if not replay_input.exists():
        return MultiSourceDecisionResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=f"replay input missing: {replay_input}",
            artifact=None,
        )

    generated_at_value = (
        generated_at if isinstance(generated_at, str) and generated_at.strip() else datetime.now(UTC).isoformat()
    )
    replay_summary = pilot.load_replay_assessment_summary(str(replay_input))
    source_artifacts: dict[str, dict[str, Any]] = {}
    for policy_profile in SOURCE_POLICY_PROFILES:
        source_result = pilot.evaluate_replay_against_canonical_phase2_source(
            replay_input_path=str(replay_input),
            run_id=f"{run_id_norm}_{policy_profile}",
            synthetic_seed=FIXED_SYNTHETIC_SEED,
            synthetic_policy_profile=policy_profile,
            synthetic_rounds=PHASE2_STAGE1_ROUNDS,
            synthetic_ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
            generated_at=generated_at_value,
            replay_summary=replay_summary,
            write_output=False,
        )
        if not isinstance(source_result.artifact, dict):
            return MultiSourceDecisionResult(
                exit_code=1,
                decision="inconclusive",
                output_path=None,
                message=f"missing source artifact for {policy_profile}",
                artifact=None,
            )
        source_artifacts[policy_profile] = source_result.artifact

    artifact = build_replay_multisource_decision_artifact(
        run_id=run_id_norm,
        replay_input_path=str(replay_input),
        generated_at=generated_at_value,
        replay_summary=replay_summary,
        balanced_artifact=source_artifacts[PHASE2_STAGE1_POLICY_PROFILE],
        eco_artifact=source_artifacts[PHASE2_SECOND_SOURCE_POLICY_PROFILE],
    )

    reports_dir = DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path or (reports_dir / OUTPUT_FILENAME)
    destination.write_text(json.dumps(artifact, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    decision = str(artifact["decision"]["decision"])
    exit_code = 0 if decision != "inconclusive" else 1
    return MultiSourceDecisionResult(
        exit_code=exit_code,
        decision=decision,
        output_path=destination,
        message=f"replay multisource decision {decision}: wrote {destination}",
        artifact=artifact,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded replay-anchored two-source decision contract and emit one decision artifact."
    )
    parser.add_argument("--replay-input", required=True, help="Path to replay fixture/input for replay assessment")
    parser.add_argument("--run-id", required=True, help="Run id ([a-z0-9_-]+)")
    parser.add_argument("--generated-at", default=None, help="Optional explicit generated_at timestamp")
    parser.add_argument("--output-path", default=None, help="Optional output artifact path")
    args = parser.parse_args()

    result = run_replay_multisource_decision(
        replay_input_path=args.replay_input,
        run_id=args.run_id,
        generated_at=args.generated_at,
        output_path=Path(args.output_path) if args.output_path else None,
    )
    payload = {
        "exit_code": result.exit_code,
        "decision": result.decision,
        "message": result.message,
        "output_path": str(result.output_path) if result.output_path else None,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
