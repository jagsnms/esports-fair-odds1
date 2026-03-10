"""Policy-driven canonical simulation Phase 2 Stage 1 contract."""
from __future__ import annotations

import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.replay_verification_assess import run_assessment
from tools.synthetic_state_generator import (
    POLICY_FAMILIES,
    generate_synthetic_distribution_summary,
    write_synthetic_raw_replay_jsonl,
)

SIMULATION_PHASE2_SUMMARY_VERSION = "simulation_phase2_policy_summary.v1"
SIMULATION_PHASE2_TRACE_SCHEMA_VERSION = "simulation_phase2_trace.v1"
CANONICAL_ENGINE_PATH = "compute_bounds>compute_rails>resolve_p_hat"
PHASE2_STAGE1_POLICY_PROFILE = "balanced_v1"
PHASE2_STAGE1_ROUNDS = 32
PHASE2_STAGE1_TICKS_PER_ROUND = 4
PHASE2_STAGE1_FIXTURE_CLASS = "phase2_balanced_v1_policy_canonical"
PHASE2_STAGE2_PREMATCH_MAP = 0.55
PHASE2_TRACE_SOURCE_CONTRACT = "canonical_phase2_balanced_v1_trace"
PHASE2_TRACE_LABEL_SCOPE = "round_result"
PHASE2_TRACE_LABEL_EVENT_TYPE = "round_result"
PHASE2_TRACE_PAIRING_RULE_ID = "unique_round_result_by_game_map_round"
PHASE2_TRACE_JOIN_KEYS = ("game_number", "map_index", "round_number")
PHASE2_TRACE_EXPORT_CONDITION = "export_only_prediction_points_with_unique_round_result_event"



def _sanitize_replay_summary(replay_summary: dict[str, Any], *, seed: int) -> dict[str, Any]:
    sanitized = dict(replay_summary)
    sanitized["fixture_class"] = PHASE2_STAGE1_FIXTURE_CLASS
    sanitized["replay_path"] = f"synthetic://phase2/{PHASE2_STAGE1_POLICY_PROFILE}/seed/{int(seed)}"
    sanitized["replay_path_exists"] = False
    return sanitized


def _build_phase2_raw_replay(
    *,
    seed: int,
    rounds: int,
) -> tuple[dict[str, Any], Path, int]:
    round_count = int(rounds)
    policy_distribution = generate_synthetic_distribution_summary(
        seed=int(seed),
        rounds=round_count,
        ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
        policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
    )
    return policy_distribution, Path("phase2_balanced_v1_policy_raw.jsonl"), round_count


def _build_phase2_trace_export(*, captured_points: list[dict[str, Any]], seed: int, rounds: int) -> dict[str, Any]:
    round_result_labels: dict[tuple[int | None, int | None, int | None], dict[str, Any]] = {}
    round_result_event_count = 0
    for point in captured_points:
        event = point.get("event")
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != PHASE2_TRACE_LABEL_EVENT_TYPE:
            continue
        key = (
            point.get("game_number"),
            point.get("map_index"),
            point.get("round_number"),
        )
        if key in round_result_labels:
            if round_result_labels[key] != event:
                raise ValueError(f"conflicting round_result label for key={key!r}")
            continue
        round_result_labels[key] = {
            "label_scope": PHASE2_TRACE_LABEL_SCOPE,
            "round_winner_team_id": event.get("round_winner_team_id"),
            "round_winner_is_team_a": event.get("round_winner_is_team_a"),
        }
        round_result_event_count += 1

    per_round_index: dict[tuple[int | None, int | None, int | None], int] = defaultdict(int)
    trace_records: list[dict[str, Any]] = []
    total_prediction_points_seen = 0
    unlabeled_prediction_points_excluded = 0
    for point in captured_points:
        if isinstance(point.get("event"), dict):
            continue
        total_prediction_points_seen += 1
        key = (
            point.get("game_number"),
            point.get("map_index"),
            point.get("round_number"),
        )
        label = round_result_labels.get(key)
        if label is None:
            unlabeled_prediction_points_excluded += 1
            continue
        point_index_in_round = per_round_index[key]
        per_round_index[key] += 1
        trace_records.append(
            {
                "game_number": point.get("game_number"),
                "map_index": point.get("map_index"),
                "round_number": point.get("round_number"),
                "point_index_in_round": int(point_index_in_round),
                "p_hat": float(point["p_hat"]),
                "rail_low": float(point["rail_low"]),
                "rail_high": float(point["rail_high"]),
                "label_scope": label["label_scope"],
                "round_winner_team_id": label["round_winner_team_id"],
                "round_winner_is_team_a": bool(label["round_winner_is_team_a"]),
            }
        )

    return {
        "schema_version": SIMULATION_PHASE2_TRACE_SCHEMA_VERSION,
        "seed": int(seed),
        "policy_profile": PHASE2_STAGE1_POLICY_PROFILE,
        "round_count": int(rounds),
        "ticks_per_round": PHASE2_STAGE1_TICKS_PER_ROUND,
        "canonical_source_contract": PHASE2_TRACE_SOURCE_CONTRACT,
        "pairing_rule": {
            "id": PHASE2_TRACE_PAIRING_RULE_ID,
            "label_event_type": PHASE2_TRACE_LABEL_EVENT_TYPE,
            "join_keys": list(PHASE2_TRACE_JOIN_KEYS),
            "export_condition": PHASE2_TRACE_EXPORT_CONDITION,
        },
        "total_prediction_points_seen": int(total_prediction_points_seen),
        "round_result_event_count": int(round_result_event_count),
        "labeled_prediction_record_count": int(len(trace_records)),
        "unlabeled_prediction_points_excluded": int(unlabeled_prediction_points_excluded),
        "trace_records": trace_records,
    }


async def _generate_phase2_summary_async(seed: int, *, rounds: int = PHASE2_STAGE1_ROUNDS) -> dict[str, Any]:
    round_count = int(rounds)
    policy_distribution = generate_synthetic_distribution_summary(
        seed=int(seed),
        rounds=round_count,
        ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
        policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
    )

    with tempfile.TemporaryDirectory(prefix="phase2_balanced_v1.") as tmpdir:
        replay_path = Path(tmpdir) / "phase2_balanced_v1_policy_raw.jsonl"
        generated_payload_count = write_synthetic_raw_replay_jsonl(
            replay_path,
            seed=int(seed),
            rounds=round_count,
            ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
            policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
        )
        replay_summary = await run_assessment(
            str(replay_path),
            prematch_map=PHASE2_STAGE2_PREMATCH_MAP,
            include_captured_points=True,
        )

    captured_points = [
        dict(item.get("point") or {})
        for item in replay_summary.pop("captured_points", [])
    ]
    replay_comparable_summary = _sanitize_replay_summary(replay_summary, seed=int(seed))
    trace_export = _build_phase2_trace_export(
        captured_points=captured_points,
        seed=int(seed),
        rounds=round_count,
    )
    return {
        "schema_version": SIMULATION_PHASE2_SUMMARY_VERSION,
        "seed": int(seed),
        "policy_profile": PHASE2_STAGE1_POLICY_PROFILE,
        "canonical_engine_path": CANONICAL_ENGINE_PATH,
        "round_count": round_count,
        "ticks_per_round": PHASE2_STAGE1_TICKS_PER_ROUND,
        "generated_payload_count": int(generated_payload_count),
        "policy_families": list(POLICY_FAMILIES),
        "realized_family_counts": dict(policy_distribution["realized_family_counts"]),
        "structural_violations_total": int(replay_comparable_summary["structural_violations_total"]),
        "behavioral_violations_total": int(replay_comparable_summary["behavioral_violations_total"]),
        "invariant_violations_total": int(replay_comparable_summary["invariant_violations_total"]),
        "p_hat_min": replay_comparable_summary["p_hat_min"],
        "p_hat_max": replay_comparable_summary["p_hat_max"],
        "p_hat_count": int(replay_comparable_summary["p_hat_count"]),
        "p_hat_median": replay_comparable_summary["p_hat_median"],
        "policy_distribution": policy_distribution,
        "replay_comparable_summary": replay_comparable_summary,
        "trace_export": trace_export,
    }


def generate_phase2_summary(seed: int, *, rounds: int = PHASE2_STAGE1_ROUNDS) -> dict[str, Any]:
    return asyncio.run(_generate_phase2_summary_async(seed, rounds=rounds))


def emit_phase2_summary(
    seed: int,
    output_path: str | Path | None = None,
    *,
    rounds: int = PHASE2_STAGE1_ROUNDS,
) -> dict[str, Any]:
    summary = generate_phase2_summary(seed, rounds=rounds)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

