#!/usr/bin/env python3
"""
Bounded replay-vs-simulation validation pilot runner.

Runs one replay assessment and one bounded canonical Phase 2 simulation assessment,
evaluates the frozen local-sanity + cross-surface comparison contract, and emits
one machine-readable decision artifact.
"""
from __future__ import annotations

import argparse
import asyncio
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
        PHASE2_STAGE1_POLICY_PROFILE,
        PHASE2_STAGE1_ROUNDS,
        PHASE2_STAGE1_TICKS_PER_ROUND,
        SIMULATION_PHASE2_SUMMARY_VERSION,
        generate_phase2_summary,
    )
    from tools.replay_verification_assess import run_assessment
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from engine.simulation.phase2 import (  # type: ignore[no-redef]
        PHASE2_STAGE1_POLICY_PROFILE,
        PHASE2_STAGE1_ROUNDS,
        PHASE2_STAGE1_TICKS_PER_ROUND,
        SIMULATION_PHASE2_SUMMARY_VERSION,
        generate_phase2_summary,
    )
    from replay_verification_assess import run_assessment  # type: ignore[no-redef]


SCHEMA_VERSION = "replay_simulation_validation_pilot.v1"
DECISION_VERSION = "replay_simulation_validation_pilot_decision.v1"
CANONICAL_PHASE2_SOURCE_CONTRACT = "canonical_phase2_balanced_v1"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
P_HAT_ABS_DELTA_TOLERANCE = 0.05
DEFAULT_REPORTS_DIR = ROOT / "automation" / "reports"
OUTPUT_PREFIX = "replay_simulation_validation_pilot_"
CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1 = "total_points_captured_abs_delta_lte_1"
CHECK_ID_RAW_CONTRACT_COVERAGE_RATE_EQUALITY = "raw_contract_coverage_rate_equality"
CHECK_ID_INVARIANT_VIOLATIONS_TOTAL_EQUALITY = "invariant_violations_total_equality"
CHECK_ID_BEHAVIORAL_VIOLATIONS_TOTAL_EQUALITY = "behavioral_violations_total_equality"
CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05 = "p_hat_min_abs_delta_lte_0_05"
CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05 = "p_hat_max_abs_delta_lte_0_05"
CHECK_ID_P_HAT_MEDIAN_ABS_DELTA_LTE_0_05 = "p_hat_median_abs_delta_lte_0_05"
FAILED_CHECK_ORDER = (
    CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1,
    CHECK_ID_RAW_CONTRACT_COVERAGE_RATE_EQUALITY,
    CHECK_ID_INVARIANT_VIOLATIONS_TOTAL_EQUALITY,
    CHECK_ID_BEHAVIORAL_VIOLATIONS_TOTAL_EQUALITY,
    CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05,
    CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05,
    CHECK_ID_P_HAT_MEDIAN_ABS_DELTA_LTE_0_05,
)
MISMATCH_CLASS_VOLUME_ALIGNMENT_ONLY = "volume_alignment_only"
MISMATCH_CLASS_CROSS_SURFACE_BEHAVIORAL_OR_METRIC = "cross_surface_behavioral_or_metric"
MISMATCH_CLASS_NONE = "none"
ALIGNMENT_ABS_DELTA_THRESHOLD = 1
ALIGNMENT_STOP_REASON = (
    "alignment inconclusive: no synthetic round candidate satisfied abs(replay.total_points_captured - synthetic.total_points_captured) <= 1"
)
TRAJECTORY_FINGERPRINT_INSUFFICIENT_P_HAT_COUNT_REASON = (
    "trajectory fingerprint unavailable: insufficient p_hat_count"
)
CANONICAL_PHASE2_FIXED_SLICE_NOTE = (
    "canonical Phase 2 slice is fixed; no round-alignment search was performed"
)


@dataclass(frozen=True)
class PilotResult:
    exit_code: int
    decision: str
    output_path: Path | None
    message: str
    artifact: dict[str, Any] | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_side_block(side_name: str, summary: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "total_points_captured": None,
        "raw_contract_points": None,
        "unknown_replay_mode_points": None,
        "invariant_violations_total": None,
        "behavioral_violations_total": None,
        "p_hat_min": None,
        "p_hat_max": None,
        "p_hat_count": None,
        "p_hat_median": None,
        "raw_contract_coverage_rate": None,
        "local_sanity_pass": False,
        "local_sanity_reasons": [],
    }
    reasons: list[str] = []
    if not isinstance(summary, dict):
        reasons.append(f"{side_name} summary unreadable: expected JSON object")
        block["local_sanity_reasons"] = reasons
        return block

    int_fields = (
        "total_points_captured",
        "raw_contract_points",
        "unknown_replay_mode_points",
        "invariant_violations_total",
        "behavioral_violations_total",
    )
    for key in int_fields:
        value = summary.get(key)
        if _is_int(value):
            block[key] = int(value)
        else:
            reasons.append(f"{side_name} missing/unreadable integer field: {key}")

    number_fields = ("p_hat_min", "p_hat_max")
    for key in number_fields:
        value = summary.get(key)
        if _is_number(value):
            block[key] = float(value)
        else:
            reasons.append(f"{side_name} missing/unreadable numeric field: {key}")

    p_hat_count = summary.get("p_hat_count")
    if _is_int(p_hat_count):
        block["p_hat_count"] = int(p_hat_count)

    p_hat_median = summary.get("p_hat_median")
    if _is_number(p_hat_median):
        block["p_hat_median"] = float(p_hat_median)

    if reasons:
        block["local_sanity_reasons"] = reasons
        return block

    total_points_captured = int(block["total_points_captured"])
    raw_contract_points = int(block["raw_contract_points"])
    unknown_replay_mode_points = int(block["unknown_replay_mode_points"])
    invariant_violations_total = int(block["invariant_violations_total"])
    behavioral_violations_total = int(block["behavioral_violations_total"])
    p_hat_min = float(block["p_hat_min"])
    p_hat_max = float(block["p_hat_max"])
    if total_points_captured > 0:
        block["raw_contract_coverage_rate"] = float(raw_contract_points) / float(total_points_captured)

    if total_points_captured <= 0:
        reasons.append(f"{side_name} local sanity failed: total_points_captured must be > 0")
    if raw_contract_points != total_points_captured:
        reasons.append(
            f"{side_name} local sanity failed: raw_contract_points must equal total_points_captured exactly"
        )
    if unknown_replay_mode_points != 0:
        reasons.append(f"{side_name} local sanity failed: unknown_replay_mode_points must equal 0")
    if invariant_violations_total != 0:
        reasons.append(f"{side_name} local sanity failed: invariant_violations_total must equal 0")
    if behavioral_violations_total != 0:
        reasons.append(f"{side_name} local sanity failed: behavioral_violations_total must equal 0")
    if not (0.0 <= p_hat_min <= p_hat_max <= 1.0):
        reasons.append(
            f"{side_name} local sanity failed: p_hat_min/p_hat_max must satisfy 0.0 <= min <= max <= 1.0"
        )

    block["local_sanity_reasons"] = reasons
    block["local_sanity_pass"] = len(reasons) == 0
    return block


def _empty_comparison_block() -> dict[str, Any]:
    return {
        "p_hat_min_abs_delta": None,
        "p_hat_max_abs_delta": None,
        "p_hat_median_abs_delta": None,
        "total_points_captured_abs_delta": None,
        "failed_checks": [],
        "mismatch_class": MISMATCH_CLASS_NONE,
        "cross_surface_pass": False,
        "cross_surface_reasons": [],
    }


def _extract_total_points(summary: Any) -> int | None:
    if isinstance(summary, dict):
        value = summary.get("total_points_captured")
        if _is_int(value):
            return int(value)
    return None


def _build_slice_metadata(
    *,
    replay_input_path: str,
    synthetic_seed: int,
    synthetic_policy_profile: str,
    synthetic_rounds: int,
    synthetic_ticks_per_round: int,
    synthetic_source_schema_version: str,
) -> dict[str, Any]:
    return {
        "replay_input_path": replay_input_path,
        "synthetic_seed": int(synthetic_seed),
        "synthetic_policy_profile": synthetic_policy_profile,
        "synthetic_rounds": int(synthetic_rounds),
        "synthetic_ticks_per_round": int(synthetic_ticks_per_round),
        "synthetic_source_contract": CANONICAL_PHASE2_SOURCE_CONTRACT,
        "synthetic_source_schema_version": synthetic_source_schema_version,
    }


def build_pilot_decision_artifact(
    *,
    run_id: str,
    replay_input_path: str,
    synthetic_seed: int,
    synthetic_policy_profile: str,
    synthetic_rounds: int,
    synthetic_ticks_per_round: int,
    generated_at: str,
    replay_summary: Any,
    synthetic_summary: Any,
    synthetic_source_schema_version: str = SIMULATION_PHASE2_SUMMARY_VERSION,
    alignment: dict[str, Any] | None = None,
    force_inconclusive_reason: str | None = None,
) -> dict[str, Any]:
    replay = _extract_side_block("replay", replay_summary)
    synthetic = _extract_side_block("synthetic", synthetic_summary)
    slice_metadata = _build_slice_metadata(
        replay_input_path=replay_input_path,
        synthetic_seed=synthetic_seed,
        synthetic_policy_profile=synthetic_policy_profile,
        synthetic_rounds=synthetic_rounds,
        synthetic_ticks_per_round=synthetic_ticks_per_round,
        synthetic_source_schema_version=synthetic_source_schema_version,
    )

    decision_reasons: list[str] = []
    failed_check_set: set[str] = set()
    comparison = _empty_comparison_block()
    comparison_reasons = comparison["cross_surface_reasons"]
    failed_checks = comparison["failed_checks"]

    if replay["local_sanity_pass"] is not True:
        decision_reasons.extend(replay["local_sanity_reasons"])
    if synthetic["local_sanity_pass"] is not True:
        decision_reasons.extend(synthetic["local_sanity_reasons"])

    if force_inconclusive_reason is not None:
        decision = "inconclusive"
        decision_reasons = [force_inconclusive_reason]
    elif decision_reasons:
        decision = "inconclusive"
    else:
        replay_p_hat_count = replay["p_hat_count"]
        synthetic_p_hat_count = synthetic["p_hat_count"]
        if (
            not _is_int(replay_p_hat_count)
            or not _is_int(synthetic_p_hat_count)
            or int(replay_p_hat_count) < 3
            or int(synthetic_p_hat_count) < 3
        ):
            decision = "inconclusive"
            decision_reasons = [TRAJECTORY_FINGERPRINT_INSUFFICIENT_P_HAT_COUNT_REASON]
            return {
                "schema_version": SCHEMA_VERSION,
                "decision_contract_version": DECISION_VERSION,
                "run_id": run_id,
                "generated_at": generated_at,
                "slice": slice_metadata,
                "alignment": alignment
                if isinstance(alignment, dict)
                else {
                    "target_replay_total_points": None,
                    "attempted_synthetic_rounds": [],
                    "attempt_results": [],
                    "selected_synthetic_rounds": None,
                    "alignment_achieved": False,
                    "stop_reason": None,
                },
                "replay": replay,
                "synthetic": synthetic,
                "comparison": comparison,
                "decision": decision,
                "decision_reasons": decision_reasons,
            }
        if replay["p_hat_median"] is None or synthetic["p_hat_median"] is None:
            decision = "inconclusive"
            decision_reasons = ["trajectory fingerprint unavailable: missing p_hat_median"]
            return {
                "schema_version": SCHEMA_VERSION,
                "decision_contract_version": DECISION_VERSION,
                "run_id": run_id,
                "generated_at": generated_at,
                "slice": slice_metadata,
                "alignment": alignment
                if isinstance(alignment, dict)
                else {
                    "target_replay_total_points": None,
                    "attempted_synthetic_rounds": [],
                    "attempt_results": [],
                    "selected_synthetic_rounds": None,
                    "alignment_achieved": False,
                    "stop_reason": None,
                },
                "replay": replay,
                "synthetic": synthetic,
                "comparison": comparison,
                "decision": decision,
                "decision_reasons": decision_reasons,
            }

        replay_coverage = replay["raw_contract_coverage_rate"]
        synthetic_coverage = synthetic["raw_contract_coverage_rate"]
        total_points_abs_delta = abs(
            int(replay["total_points_captured"]) - int(synthetic["total_points_captured"])
        )
        comparison["total_points_captured_abs_delta"] = total_points_abs_delta
        if total_points_abs_delta > ALIGNMENT_ABS_DELTA_THRESHOLD:
            failed_check_set.add(CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1)
            comparison_reasons.append(
                "cross-surface mismatch: abs(replay.total_points_captured - synthetic.total_points_captured) must be <= 1"
            )
        if replay_coverage != synthetic_coverage:
            failed_check_set.add(CHECK_ID_RAW_CONTRACT_COVERAGE_RATE_EQUALITY)
            comparison_reasons.append(
                "cross-surface mismatch: replay.raw_contract_coverage_rate must equal synthetic.raw_contract_coverage_rate exactly"
            )
        if replay["invariant_violations_total"] != synthetic["invariant_violations_total"]:
            failed_check_set.add(CHECK_ID_INVARIANT_VIOLATIONS_TOTAL_EQUALITY)
            comparison_reasons.append(
                "cross-surface mismatch: replay.invariant_violations_total must equal synthetic.invariant_violations_total exactly"
            )
        if replay["behavioral_violations_total"] != synthetic["behavioral_violations_total"]:
            failed_check_set.add(CHECK_ID_BEHAVIORAL_VIOLATIONS_TOTAL_EQUALITY)
            comparison_reasons.append(
                "cross-surface mismatch: replay.behavioral_violations_total must equal synthetic.behavioral_violations_total exactly"
            )

        p_hat_min_abs_delta = abs(float(replay["p_hat_min"]) - float(synthetic["p_hat_min"]))
        p_hat_max_abs_delta = abs(float(replay["p_hat_max"]) - float(synthetic["p_hat_max"]))
        comparison["p_hat_min_abs_delta"] = p_hat_min_abs_delta
        comparison["p_hat_max_abs_delta"] = p_hat_max_abs_delta
        if p_hat_min_abs_delta > P_HAT_ABS_DELTA_TOLERANCE:
            failed_check_set.add(CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05)
            comparison_reasons.append(
                "cross-surface mismatch: abs(replay.p_hat_min - synthetic.p_hat_min) must be <= 0.05"
            )
        if p_hat_max_abs_delta > P_HAT_ABS_DELTA_TOLERANCE:
            failed_check_set.add(CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05)
            comparison_reasons.append(
                "cross-surface mismatch: abs(replay.p_hat_max - synthetic.p_hat_max) must be <= 0.05"
            )
        p_hat_median_abs_delta = abs(float(replay["p_hat_median"]) - float(synthetic["p_hat_median"]))
        comparison["p_hat_median_abs_delta"] = p_hat_median_abs_delta
        if p_hat_median_abs_delta > P_HAT_ABS_DELTA_TOLERANCE:
            failed_check_set.add(CHECK_ID_P_HAT_MEDIAN_ABS_DELTA_LTE_0_05)
            comparison_reasons.append(
                "cross-surface mismatch: abs(replay.p_hat_median - synthetic.p_hat_median) must be <= 0.05"
            )

        failed_checks.extend([check_id for check_id in FAILED_CHECK_ORDER if check_id in failed_check_set])
        if comparison_reasons:
            decision = "mismatch"
            if failed_checks == [CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1]:
                comparison["mismatch_class"] = MISMATCH_CLASS_VOLUME_ALIGNMENT_ONLY
            else:
                comparison["mismatch_class"] = MISMATCH_CLASS_CROSS_SURFACE_BEHAVIORAL_OR_METRIC
            decision_reasons.extend(comparison_reasons)
        else:
            decision = "pass"
            comparison["cross_surface_pass"] = True

    return {
        "schema_version": SCHEMA_VERSION,
        "decision_contract_version": DECISION_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "slice": slice_metadata,
        "alignment": alignment
        if isinstance(alignment, dict)
        else {
            "target_replay_total_points": None,
            "attempted_synthetic_rounds": [],
            "attempt_results": [],
            "selected_synthetic_rounds": None,
            "alignment_achieved": False,
            "stop_reason": None,
        },
        "replay": replay,
        "synthetic": synthetic,
        "comparison": comparison,
        "decision": decision,
        "decision_reasons": decision_reasons,
    }


def run_replay_simulation_validation_pilot(
    *,
    replay_input_path: str,
    run_id: str,
    synthetic_seed: int = 1337,
    synthetic_policy_profile: str = PHASE2_STAGE1_POLICY_PROFILE,
    synthetic_rounds: int = PHASE2_STAGE1_ROUNDS,
    synthetic_ticks_per_round: int = PHASE2_STAGE1_TICKS_PER_ROUND,
    generated_at: str | None = None,
    output_path: Path | None = None,
) -> PilotResult:
    run_id_norm = str(run_id or "")
    if not run_id_norm or RUN_ID_PATTERN.fullmatch(run_id_norm) is None:
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message="invalid run_id: must match [a-z0-9_-]+ and be non-empty",
            artifact=None,
        )
    replay_input = Path(replay_input_path)
    if not replay_input.exists():
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=f"replay input missing: {replay_input}",
            artifact=None,
        )
    if synthetic_policy_profile != PHASE2_STAGE1_POLICY_PROFILE:
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=(
                f"invalid synthetic_policy_profile: canonical Phase 2 binding only supports {PHASE2_STAGE1_POLICY_PROFILE}"
            ),
            artifact=None,
        )
    if synthetic_rounds != PHASE2_STAGE1_ROUNDS or synthetic_ticks_per_round != PHASE2_STAGE1_TICKS_PER_ROUND:
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=(
                f"invalid synthetic shape: canonical Phase 2 binding only supports rounds={PHASE2_STAGE1_ROUNDS}, ticks_per_round={PHASE2_STAGE1_TICKS_PER_ROUND}"
            ),
            artifact=None,
        )

    generated_at_value = (
        generated_at if isinstance(generated_at, str) and generated_at.strip() else datetime.now(UTC).isoformat()
    )

    replay_summary: Any
    synthetic_summary: Any
    try:
        replay_summary = asyncio.run(run_assessment(str(replay_input)))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        replay_summary = {"_error": f"replay assessment failed: {exc}"}

    replay_total_points = _extract_total_points(replay_summary)

    canonical_phase2_summary: Any
    try:
        canonical_phase2_summary = generate_phase2_summary(int(synthetic_seed))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        canonical_phase2_summary = {"_error": f"canonical phase2 summary failed: {exc}"}

    synthetic_summary = (
        canonical_phase2_summary.get("replay_comparable_summary")
        if isinstance(canonical_phase2_summary, dict)
        else None
    )
    if not isinstance(synthetic_summary, dict):
        synthetic_summary = {"_error": "canonical phase2 replay_comparable_summary unavailable"}

    synthetic_total_points = _extract_total_points(synthetic_summary)
    abs_delta = (
        abs(int(replay_total_points) - int(synthetic_total_points))
        if replay_total_points is not None and synthetic_total_points is not None
        else None
    )
    synthetic_source_schema_version = (
        str(canonical_phase2_summary.get("schema_version"))
        if isinstance(canonical_phase2_summary, dict) and isinstance(canonical_phase2_summary.get("schema_version"), str)
        else SIMULATION_PHASE2_SUMMARY_VERSION
    )

    alignment_payload: dict[str, Any] = {
        "target_replay_total_points": replay_total_points,
        "attempted_synthetic_rounds": [PHASE2_STAGE1_ROUNDS],
        "attempt_results": [
            {
                "attempted_rounds": int(PHASE2_STAGE1_ROUNDS),
                "synthetic_total_points": synthetic_total_points,
                "abs_delta": abs_delta,
            }
        ],
        "selected_synthetic_rounds": (
            int(PHASE2_STAGE1_ROUNDS)
            if abs_delta is not None and abs_delta <= ALIGNMENT_ABS_DELTA_THRESHOLD
            else None
        ),
        "alignment_achieved": abs_delta is not None and abs_delta <= ALIGNMENT_ABS_DELTA_THRESHOLD,
        "stop_reason": (
            None
            if abs_delta is not None and abs_delta <= ALIGNMENT_ABS_DELTA_THRESHOLD
            else CANONICAL_PHASE2_FIXED_SLICE_NOTE
        ),
    }

    artifact = build_pilot_decision_artifact(
        run_id=run_id_norm,
        replay_input_path=str(replay_input),
        synthetic_seed=int(synthetic_seed),
        synthetic_policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
        synthetic_rounds=int(PHASE2_STAGE1_ROUNDS),
        synthetic_ticks_per_round=int(PHASE2_STAGE1_TICKS_PER_ROUND),
        generated_at=generated_at_value,
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
        synthetic_source_schema_version=synthetic_source_schema_version,
        alignment=alignment_payload,
        force_inconclusive_reason=(
            None
            if abs_delta is not None and abs_delta <= ALIGNMENT_ABS_DELTA_THRESHOLD
            else CANONICAL_PHASE2_FIXED_SLICE_NOTE
        ),
    )

    reports_dir = DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path or (reports_dir / f"{OUTPUT_PREFIX}{run_id_norm}.json")
    destination.write_text(json.dumps(artifact, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    decision = str(artifact["decision"])
    exit_code = 0 if decision == "pass" else 2 if decision == "mismatch" else 1
    return PilotResult(
        exit_code=exit_code,
        decision=decision,
        output_path=destination,
        message=f"replay-simulation pilot {decision}: wrote {destination}",
        artifact=artifact,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded replay-vs-simulation validation pilot and emit one decision artifact."
    )
    parser.add_argument("--replay-input", required=True, help="Path to replay fixture/input for replay assessment")
    parser.add_argument("--run-id", required=True, help="Run id ([a-z0-9_-]+)")
    parser.add_argument("--synthetic-seed", type=int, default=1337)
    parser.add_argument(
        "--synthetic-policy-profile",
        default=PHASE2_STAGE1_POLICY_PROFILE,
        choices=[PHASE2_STAGE1_POLICY_PROFILE],
    )
    parser.add_argument("--synthetic-rounds", type=int, default=PHASE2_STAGE1_ROUNDS)
    parser.add_argument("--synthetic-ticks-per-round", type=int, default=PHASE2_STAGE1_TICKS_PER_ROUND)
    parser.add_argument("--generated-at", default=None, help="Optional explicit generated_at timestamp")
    parser.add_argument("--output-path", default=None, help="Optional output artifact path")
    args = parser.parse_args()

    result = run_replay_simulation_validation_pilot(
        replay_input_path=args.replay_input,
        run_id=args.run_id,
        synthetic_seed=args.synthetic_seed,
        synthetic_policy_profile=args.synthetic_policy_profile,
        synthetic_rounds=args.synthetic_rounds,
        synthetic_ticks_per_round=args.synthetic_ticks_per_round,
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




