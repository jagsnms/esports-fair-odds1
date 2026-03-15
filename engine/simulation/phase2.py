"""Policy-driven canonical simulation Phase 2 Stage 1 contract."""
from __future__ import annotations

import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.replay_verification_assess import (
    common_point_source_basis_descriptor as replay_common_point_source_basis_descriptor,
    run_assessment,
)
from tools.synthetic_state_generator import (
    POLICY_FAMILIES,
    generate_synthetic_distribution_summary,
    write_synthetic_raw_replay_jsonl,
)

SIMULATION_PHASE2_SUMMARY_VERSION = "simulation_phase2_policy_summary.v1"
SIMULATION_PHASE2_TRACE_SCHEMA_VERSION = "simulation_phase2_trace.v1"
CANONICAL_ENGINE_PATH = "compute_bounds>compute_rails>resolve_p_hat"
PHASE2_DEFAULT_POLICY_PROFILE = "balanced_v1"
PHASE2_SECOND_SOURCE_POLICY_PROFILE = "eco_bias_v1"
PHASE2_SUPPORTED_POLICY_PROFILES = (
    PHASE2_DEFAULT_POLICY_PROFILE,
    PHASE2_SECOND_SOURCE_POLICY_PROFILE,
)
PHASE2_STAGE1_POLICY_PROFILE = PHASE2_DEFAULT_POLICY_PROFILE
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


def _normalize_phase2_policy_profile(policy_profile: str) -> str:
    profile = str(policy_profile or PHASE2_DEFAULT_POLICY_PROFILE)
    if profile not in PHASE2_SUPPORTED_POLICY_PROFILES:
        raise ValueError(f"unsupported canonical phase2 policy profile: {profile}")
    return profile


def phase2_source_contract(policy_profile: str) -> str:
    profile = _normalize_phase2_policy_profile(policy_profile)
    return f"canonical_phase2_{profile}"


def phase2_fixture_class(policy_profile: str) -> str:
    profile = _normalize_phase2_policy_profile(policy_profile)
    return f"phase2_{profile}_policy_canonical"


def phase2_trace_source_contract(policy_profile: str) -> str:
    profile = _normalize_phase2_policy_profile(policy_profile)
    return f"canonical_phase2_{profile}_trace"


def phase2_replay_path(*, policy_profile: str, seed: int) -> str:
    profile = _normalize_phase2_policy_profile(policy_profile)
    return f"synthetic://phase2/{profile}/seed/{int(seed)}"


def phase2_summary_metadata(policy_profile: str) -> dict[str, str]:
    profile = _normalize_phase2_policy_profile(policy_profile)
    return {
        "policy_profile": profile,
        "canonical_source_contract": phase2_source_contract(profile),
        "fixture_class": phase2_fixture_class(profile),
        "trace_source_contract": phase2_trace_source_contract(profile),
    }


def common_point_source_basis_descriptor() -> dict[str, Any]:
    return replay_common_point_source_basis_descriptor()


def _build_common_point_source_projection_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for record in records:
        projected.append(
            {
                "p_hat": float(record["p_hat"]),
                "rail_low": float(record["rail_low"]),
                "rail_high": float(record["rail_high"]),
                "game_number": int(record["game_number"]),
                "map_index": int(record["map_index"]),
                "round_number": int(record["round_number"]),
            }
        )
    return projected


def common_point_source_projection_descriptor(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract_id": "common_point_source_projection.v1",
        "source_surface": "canonical_phase2_trace",
        "shared_fields": list(common_point_source_basis_descriptor()["shared_fields"]),
        "projection_limits": {
            "side_local_projection_only": True,
            "record_matching_implied": False,
            "alignment_implied": False,
            "scoring_or_selection_implied": False,
        },
        "records": _build_common_point_source_projection_records(records),
    }



def _sanitize_replay_summary(
    replay_summary: dict[str, Any],
    *,
    seed: int,
    policy_profile: str,
) -> dict[str, Any]:
    metadata = phase2_summary_metadata(policy_profile)
    sanitized = dict(replay_summary)
    sanitized["fixture_class"] = metadata["fixture_class"]
    sanitized["replay_path"] = phase2_replay_path(policy_profile=policy_profile, seed=int(seed))
    sanitized["replay_path_exists"] = False
    sanitized["canonical_source_contract"] = metadata["canonical_source_contract"]
    return sanitized


def _build_phase2_trace_export(
    *,
    captured_points: list[dict[str, Any]],
    seed: int,
    rounds: int,
    policy_profile: str,
) -> dict[str, Any]:
    profile = _normalize_phase2_policy_profile(policy_profile)
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
        "policy_profile": profile,
        "round_count": int(rounds),
        "ticks_per_round": PHASE2_STAGE1_TICKS_PER_ROUND,
        "canonical_source_contract": phase2_trace_source_contract(profile),
        "pairing_rule": {
            "id": PHASE2_TRACE_PAIRING_RULE_ID,
            "label_event_type": PHASE2_TRACE_LABEL_EVENT_TYPE,
            "join_keys": list(PHASE2_TRACE_JOIN_KEYS),
            "export_condition": PHASE2_TRACE_EXPORT_CONDITION,
        },
        "common_point_source_basis": common_point_source_basis_descriptor(),
        "common_point_source_projection": common_point_source_projection_descriptor(trace_records),
        "total_prediction_points_seen": int(total_prediction_points_seen),
        "round_result_event_count": int(round_result_event_count),
        "labeled_prediction_record_count": int(len(trace_records)),
        "unlabeled_prediction_points_excluded": int(unlabeled_prediction_points_excluded),
        "trace_records": trace_records,
    }


async def _generate_phase2_summary_async(
    seed: int,
    *,
    rounds: int = PHASE2_STAGE1_ROUNDS,
    policy_profile: str = PHASE2_STAGE1_POLICY_PROFILE,
) -> dict[str, Any]:
    profile = _normalize_phase2_policy_profile(policy_profile)
    metadata = phase2_summary_metadata(profile)
    round_count = int(rounds)
    policy_distribution = generate_synthetic_distribution_summary(
        seed=int(seed),
        rounds=round_count,
        ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
        policy_profile=profile,
    )

    with tempfile.TemporaryDirectory(prefix=f"phase2_{profile}.") as tmpdir:
        replay_path = Path(tmpdir) / f"phase2_{profile}_policy_raw.jsonl"
        generated_payload_count = write_synthetic_raw_replay_jsonl(
            replay_path,
            seed=int(seed),
            rounds=round_count,
            ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
            policy_profile=profile,
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
    replay_comparable_summary = _sanitize_replay_summary(
        replay_summary,
        seed=int(seed),
        policy_profile=profile,
    )
    trace_export = _build_phase2_trace_export(
        captured_points=captured_points,
        seed=int(seed),
        rounds=round_count,
        policy_profile=profile,
    )
    return {
        "schema_version": SIMULATION_PHASE2_SUMMARY_VERSION,
        "seed": int(seed),
        "policy_profile": profile,
        "canonical_source_contract": metadata["canonical_source_contract"],
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


def generate_phase2_summary(
    seed: int,
    *,
    rounds: int = PHASE2_STAGE1_ROUNDS,
    policy_profile: str = PHASE2_STAGE1_POLICY_PROFILE,
) -> dict[str, Any]:
    return asyncio.run(_generate_phase2_summary_async(seed, rounds=rounds, policy_profile=policy_profile))


def emit_phase2_summary(
    seed: int,
    output_path: str | Path | None = None,
    *,
    rounds: int = PHASE2_STAGE1_ROUNDS,
    policy_profile: str = PHASE2_STAGE1_POLICY_PROFILE,
) -> dict[str, Any]:
    summary = generate_phase2_summary(seed, rounds=rounds, policy_profile=policy_profile)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
