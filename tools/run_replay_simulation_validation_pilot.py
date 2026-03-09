#!/usr/bin/env python3
"""
Bounded replay-vs-simulation validation pilot runner.

Runs one replay assessment and one seeded synthetic assessment, evaluates the
frozen local-sanity + cross-surface comparison contract, and emits one
machine-readable decision artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.replay_verification_assess import run_assessment
    from tools.synthetic_state_generator import POLICY_PROFILES, write_synthetic_raw_replay_jsonl
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from replay_verification_assess import run_assessment  # type: ignore[no-redef]
    from synthetic_state_generator import POLICY_PROFILES, write_synthetic_raw_replay_jsonl  # type: ignore[no-redef]


SCHEMA_VERSION = "replay_simulation_validation_pilot.v1"
DECISION_VERSION = "replay_simulation_validation_pilot_decision.v1"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
P_HAT_ABS_DELTA_TOLERANCE = 0.05
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = ROOT / "automation" / "reports"
OUTPUT_PREFIX = "replay_simulation_validation_pilot_"
CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1 = "total_points_captured_abs_delta_lte_1"
CHECK_ID_RAW_CONTRACT_COVERAGE_RATE_EQUALITY = "raw_contract_coverage_rate_equality"
CHECK_ID_INVARIANT_VIOLATIONS_TOTAL_EQUALITY = "invariant_violations_total_equality"
CHECK_ID_BEHAVIORAL_VIOLATIONS_TOTAL_EQUALITY = "behavioral_violations_total_equality"
CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05 = "p_hat_min_abs_delta_lte_0_05"
CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05 = "p_hat_max_abs_delta_lte_0_05"
FAILED_CHECK_ORDER = (
    CHECK_ID_TOTAL_POINTS_CAPTURED_ABS_DELTA_LTE_1,
    CHECK_ID_RAW_CONTRACT_COVERAGE_RATE_EQUALITY,
    CHECK_ID_INVARIANT_VIOLATIONS_TOTAL_EQUALITY,
    CHECK_ID_BEHAVIORAL_VIOLATIONS_TOTAL_EQUALITY,
    CHECK_ID_P_HAT_MIN_ABS_DELTA_LTE_0_05,
    CHECK_ID_P_HAT_MAX_ABS_DELTA_LTE_0_05,
)
MISMATCH_CLASS_VOLUME_ALIGNMENT_ONLY = "volume_alignment_only"
MISMATCH_CLASS_CROSS_SURFACE_BEHAVIORAL_OR_METRIC = "cross_surface_behavioral_or_metric"
MISMATCH_CLASS_NONE = "none"
ALIGNMENT_ABS_DELTA_THRESHOLD = 1
ALIGNMENT_STOP_REASON = (
    "alignment inconclusive: no synthetic round candidate satisfied abs(replay.total_points_captured - synthetic.total_points_captured) <= 1"
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
    alignment: dict[str, Any] | None = None,
    force_inconclusive_reason: str | None = None,
) -> dict[str, Any]:
    replay = _extract_side_block("replay", replay_summary)
    synthetic = _extract_side_block("synthetic", synthetic_summary)

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
        "slice": {
            "replay_input_path": replay_input_path,
            "synthetic_seed": int(synthetic_seed),
            "synthetic_policy_profile": synthetic_policy_profile,
            "synthetic_rounds": int(synthetic_rounds),
            "synthetic_ticks_per_round": int(synthetic_ticks_per_round),
            "synthetic_raw_contract_mode": "IN_PROGRESS",
        },
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
    synthetic_policy_profile: str = "balanced_v1",
    synthetic_rounds: int = 10,
    synthetic_ticks_per_round: int = 4,
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
    if synthetic_policy_profile not in POLICY_PROFILES:
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message=f"invalid synthetic_policy_profile: must be one of {', '.join(POLICY_PROFILES)}",
            artifact=None,
        )
    if synthetic_rounds <= 0 or synthetic_ticks_per_round <= 0:
        return PilotResult(
            exit_code=1,
            decision="inconclusive",
            output_path=None,
            message="invalid synthetic shape: rounds and ticks_per_round must be > 0",
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
    rounds_base = int(synthetic_rounds)
    if replay_total_points is not None:
        rounds_base = max(1, int(math.ceil(float(replay_total_points) / float(int(synthetic_ticks_per_round)))))
    raw_candidates = [rounds_base, rounds_base - 1, rounds_base + 1]
    candidate_rounds: list[int] = []
    for rounds in raw_candidates:
        if rounds > 0 and rounds not in candidate_rounds:
            candidate_rounds.append(rounds)

    attempted_rounds: list[int] = []
    attempt_results: list[dict[str, Any]] = []
    selected_synthetic_rounds: int | None = None
    selected_synthetic_summary: Any = None
    last_synthetic_summary: Any = None

    with tempfile.TemporaryDirectory(prefix=f"{run_id_norm}.pilot.synthetic.") as temp_dir:
        synthetic_path = Path(temp_dir) / "synthetic_raw_replay.jsonl"
        for rounds in candidate_rounds:
            attempted_rounds.append(rounds)
            try:
                write_synthetic_raw_replay_jsonl(
                    synthetic_path,
                    seed=int(synthetic_seed),
                    rounds=int(rounds),
                    ticks_per_round=int(synthetic_ticks_per_round),
                    policy_profile=synthetic_policy_profile,
                )
                current_summary = asyncio.run(run_assessment(str(synthetic_path)))
            except Exception as exc:  # pragma: no cover - runtime safeguard
                current_summary = {"_error": f"synthetic assessment failed: {exc}"}
            last_synthetic_summary = current_summary
            synthetic_total_points = _extract_total_points(current_summary)
            abs_delta = (
                abs(int(replay_total_points) - int(synthetic_total_points))
                if replay_total_points is not None and synthetic_total_points is not None
                else None
            )
            attempt_results.append(
                {
                    "attempted_rounds": int(rounds),
                    "synthetic_total_points": synthetic_total_points,
                    "abs_delta": abs_delta,
                }
            )
            if abs_delta is not None and abs_delta <= ALIGNMENT_ABS_DELTA_THRESHOLD and selected_synthetic_rounds is None:
                selected_synthetic_rounds = int(rounds)
                selected_synthetic_summary = current_summary
                break

    if selected_synthetic_summary is not None:
        synthetic_summary = selected_synthetic_summary
    else:
        synthetic_summary = (
            last_synthetic_summary
            if last_synthetic_summary is not None
            else {"_error": "synthetic assessment unavailable"}
        )

    alignment_payload: dict[str, Any] = {
        "target_replay_total_points": replay_total_points,
        "attempted_synthetic_rounds": attempted_rounds,
        "attempt_results": attempt_results,
        "selected_synthetic_rounds": selected_synthetic_rounds,
        "alignment_achieved": selected_synthetic_rounds is not None,
        "stop_reason": None if selected_synthetic_rounds is not None else ALIGNMENT_STOP_REASON,
    }

    artifact = build_pilot_decision_artifact(
        run_id=run_id_norm,
        replay_input_path=str(replay_input),
        synthetic_seed=int(synthetic_seed),
        synthetic_policy_profile=synthetic_policy_profile,
        synthetic_rounds=int(synthetic_rounds),
        synthetic_ticks_per_round=int(synthetic_ticks_per_round),
        generated_at=generated_at_value,
        replay_summary=replay_summary,
        synthetic_summary=synthetic_summary,
        alignment=alignment_payload,
        force_inconclusive_reason=None if selected_synthetic_rounds is not None else ALIGNMENT_STOP_REASON,
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
    parser.add_argument("--synthetic-policy-profile", default="balanced_v1", choices=list(POLICY_PROFILES))
    parser.add_argument("--synthetic-rounds", type=int, default=10)
    parser.add_argument("--synthetic-ticks-per-round", type=int, default=4)
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
