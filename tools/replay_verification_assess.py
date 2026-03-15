#!/usr/bin/env python3
"""
Stage 4/4B/4C: Bounded replay/verification assessment for PHAT initiative.
Runs replay with invariant_diagnostics=True, captures derived.debug per point,
aggregates structural/behavioral violation counts and PHAT/rail metrics.
4B: Pre-loads payloads in script and injects into runner (unblocks lazy-load in script context).
4C: Supports raw-contract replay input (BO3-shaped snapshots) so full resolve path is exercised.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.models import Config, Derived, State
from backend.store.memory_store import MemoryStore
from backend.services.runner import Runner, _is_raw_bo3_snapshot
from engine.replay.bo3_jsonl import load_bo3_jsonl_entries, iter_payloads, load_generic_jsonl
from engine.compute.rails_cs2 import RAIL_INPUT_V2_REQUIRED_FIELDS

SCHEMA_VERSION = "replay_validation_summary.v1"
ENGINE_SPEC_PATH = ROOT / "docs" / "ENGINE_SPEC.json"
CONTRACT_DIAGNOSTIC_REQUIRED_KEYS = [
    "alive_counts",
    "hp_totals",
    "loadout_totals",
    "round_phase",
    "round_number",
    "q_intra_total",
    "rail_low",
    "rail_high",
    "p_hat_prev",
    "p_hat_final",
    "round_time_remaining_s",
    "is_bomb_planted",
    "movement_confidence",
    "target_p_hat",
    "expected_p_hat_after_movement",
    "movement_gap_abs",
    "structural_violations",
    "behavioral_violations",
]


def _load_engine_spec_required_keys() -> list[str]:
    """Load canonical diagnostics required keys from ENGINE_SPEC with trim+dedupe normalization."""
    try:
        spec_obj = json.loads(ENGINE_SPEC_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"engine spec unreadable: {ENGINE_SPEC_PATH} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"engine spec invalid JSON: {ENGINE_SPEC_PATH} ({exc})") from exc

    if not isinstance(spec_obj, dict):
        raise ValueError("engine spec must be a JSON object")
    invariants = spec_obj.get("invariants")
    if not isinstance(invariants, dict):
        raise ValueError("missing/wrong-type path: invariants")
    required = invariants.get("diagnostics_payload_required_fields")
    if not isinstance(required, list):
        raise ValueError("missing/wrong-type path: invariants.diagnostics_payload_required_fields")
    if len(required) == 0:
        raise ValueError("empty path: invariants.diagnostics_payload_required_fields")

    normalized: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(required):
        if not isinstance(item, str):
            raise ValueError(
                f"wrong-type value at invariants.diagnostics_payload_required_fields[{idx}]: expected string"
            )
        value = item.strip()
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if len(normalized) == 0:
        raise ValueError("empty path after normalization: invariants.diagnostics_payload_required_fields")
    return normalized


def _fixture_class_from_path(replay_path_str: str) -> str:
    """Derive deterministic fixture class from replay file stem."""
    return Path(replay_path_str).stem


def _load_replay_payloads(replay_path_str: str) -> list[dict[str, Any]]:
    """Load payloads: BO3 entries first, then generic JSONL. (4B: script-side load to unblock.)"""
    entries = load_bo3_jsonl_entries(replay_path_str)
    payloads = [p for _, p in iter_payloads(entries, None)]
    if not payloads:
        payloads = load_generic_jsonl(replay_path_str)
    return payloads


def _normalize_prematch_map(value: float | None) -> float | None:
    """Normalize optional prematch_map for replay assessment config."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v <= 0.0 or v >= 1.0:
        return None
    return v


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _build_replay_point_source(captured: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in captured:
        point = item.get("point") or {}
        records.append(
            {
                "p_hat": _number_or_none(point.get("p_hat")),
                "rail_low": _number_or_none(point.get("rail_low")),
                "rail_high": _number_or_none(point.get("rail_high")),
                "game_number": _int_or_none(point.get("game_number")),
                "map_index": _int_or_none(point.get("map_index")),
                "round_number": _int_or_none(point.get("round_number")),
                # Current promoted replay fixtures do not expose a stable source-like point time.
                "time": None,
            }
        )
    return {"records": records}

async def run_assessment(
    replay_path: str,
    *,
    prematch_map: float | None = None,
    include_captured_points: bool = False,
) -> dict[str, Any]:
    """Run replay with invariant_diagnostics=True; capture each appended point's derived; return metrics."""
    os.chdir(ROOT)
    path = Path(replay_path)
    if not path.is_absolute():
        path = ROOT / path
    replay_path_str = str(path)
    payloads = _load_replay_payloads(replay_path_str)
    direct_load_count = len(payloads)

    store = MemoryStore(max_history=5000)
    state = State(config=Config(source="REPLAY", replay_path=replay_path_str, replay_loop=False))
    derived = Derived(p_hat=0.5, bound_low=0.0, bound_high=1.0, rail_low=0.0, rail_high=1.0, kappa=0.0)
    await store.set_current(state, derived)

    captured: list[dict[str, Any]] = []
    original_append = store.append_point

    async def capture_append(point: Any, state: State, derived: Derived) -> None:
        pt = {
            "p_hat": getattr(point, "p_hat", None),
            "rail_low": getattr(point, "rail_low", None),
            "rail_high": getattr(point, "rail_high", None),
            "game_number": getattr(point, "game_number", None),
            "map_index": getattr(point, "map_index", None),
            "round_number": getattr(point, "round_number", None),
            "event": getattr(point, "event", None),
            "time": getattr(point, "time", None),
        }
        captured.append({"point": pt, "derived": asdict(derived)})
        await original_append(point, state, derived)

    store.append_point = capture_append

    class _NoopBroadcaster:
        async def broadcast(self, msg: Any) -> None:
            pass

    broadcaster = _NoopBroadcaster()
    runner = Runner(store=store, broadcaster=broadcaster)

    # 4B: Pre-load and inject payloads so the runner processes them.
    runner._replay_payloads = payloads
    runner._replay_index = 0
    runner._replay_path = replay_path_str
    runner._replay_match_id = None
    runner._replay_format = (
        "raw"
        if (payloads and _is_raw_bo3_snapshot(payloads[0]))
        else ("point" if payloads else None)
    )

    prematch_map_value = _normalize_prematch_map(prematch_map)
    config = Config(
        source="REPLAY",
        replay_path=replay_path_str,
        replay_loop=False,
        invariant_diagnostics=True,
        prematch_map=prematch_map_value,
    )
    max_ticks = 5000
    max_points_cap = 500
    for _ in range(max_ticks):
        if len(runner._replay_payloads) == 0:
            break
        if runner._replay_index >= len(runner._replay_payloads):
            break
        if len(captured) >= max_points_cap:
            break
        await runner._tick_replay(config)

    replay_contract_status = (
        runner.get_replay_contract_status()
        if hasattr(runner, "get_replay_contract_status")
        else {}
    )

    # Aggregate metrics
    total_points = len(captured)
    raw_contract = 0
    point_passthrough = 0
    non_canonical_point_points = 0
    replay_quarantine_status_counts: dict[str, int] = {}
    rail_input_contract_policy_counts: dict[str, int] = {}
    rail_input_active_endpoint_semantics_counts: dict[str, int] = {}
    rail_input_reason_code_counts: dict[str, int] = {}
    rail_input_v2_activated_points = 0
    rail_input_v1_fallback_points = 0
    unknown_replay_mode_points = 0
    carryover_evidence_by_source_class: dict[str, dict[str, Any]] = {}
    structural_violations_total = 0
    behavioral_violations_total = 0
    invariant_violations_total = 0
    points_with_contract_diagnostics = 0
    contract_diagnostics_key_presence_counts = {
        key: 0 for key in CONTRACT_DIAGNOSTIC_REQUIRED_KEYS
    }
    contract_diagnostics_missing_key_counts = {
        key: 0 for key in CONTRACT_DIAGNOSTIC_REQUIRED_KEYS
    }
    p_hats: list[float] = []
    rail_lows: list[float] = []
    rail_highs: list[float] = []

    for item in captured:
        der = item.get("derived") or {}
        debug = der.get("debug") or {}
        replay_mode = debug.get("replay_mode")
        if replay_mode == "raw_contract":
            raw_contract += 1
        elif replay_mode == "point_passthrough":
            point_passthrough += 1
        else:
            unknown_replay_mode_points += 1
        if debug.get("replay_contract_class") == "non_canonical_point":
            non_canonical_point_points += 1
        qs = debug.get("replay_quarantine_status")
        if isinstance(qs, str) and qs:
            replay_quarantine_status_counts[qs] = replay_quarantine_status_counts.get(qs, 0) + 1
        rail_policy = debug.get("rail_input_contract_policy")
        if isinstance(rail_policy, str) and rail_policy:
            rail_input_contract_policy_counts[rail_policy] = rail_input_contract_policy_counts.get(rail_policy, 0) + 1
        rail_sem = debug.get("rail_input_active_endpoint_semantics")
        if isinstance(rail_sem, str) and rail_sem:
            rail_input_active_endpoint_semantics_counts[rail_sem] = (
                rail_input_active_endpoint_semantics_counts.get(rail_sem, 0) + 1
            )
        rail_reason = debug.get("rail_input_v1_fallback_reason_code")
        if isinstance(rail_reason, str) and rail_reason:
            rail_input_reason_code_counts[rail_reason] = rail_input_reason_code_counts.get(rail_reason, 0) + 1
        if debug.get("rail_input_v2_activated") is True:
            rail_input_v2_activated_points += 1
        if debug.get("rail_input_v1_fallback_used") is True:
            rail_input_v1_fallback_points += 1

        src = debug.get("rail_input_source") or "unknown"
        kind = debug.get("rail_input_replay_kind") or "unknown"
        source_class_key = f"{src}_{kind}"
        if source_class_key not in carryover_evidence_by_source_class:
            carryover_evidence_by_source_class[source_class_key] = {
                "points": 0,
                "v2_activated_points": 0,
                "v1_fallback_points": 0,
                "reason_code_counts": {},
                "required_complete_points": 0,
                "required_incomplete_points": 0,
            }
        rec = carryover_evidence_by_source_class[source_class_key]
        rec["points"] += 1
        if debug.get("rail_input_v2_activated") is True:
            rec["v2_activated_points"] += 1
        if debug.get("rail_input_v1_fallback_used") is True:
            rec["v1_fallback_points"] += 1
        rc = debug.get("rail_input_v1_fallback_reason_code") or debug.get("rail_input_activation_reason_code")
        if isinstance(rc, str) and rc:
            rec["reason_code_counts"][rc] = rec["reason_code_counts"].get(rc, 0) + 1
        if debug.get("rail_input_v2_required_complete") is True:
            rec["required_complete_points"] += 1
        else:
            rec["required_incomplete_points"] += 1

        cd = debug.get("contract_diagnostics")
        if isinstance(cd, dict):
            points_with_contract_diagnostics += 1
            for key in CONTRACT_DIAGNOSTIC_REQUIRED_KEYS:
                if key in cd:
                    contract_diagnostics_key_presence_counts[key] += 1
                else:
                    contract_diagnostics_missing_key_counts[key] += 1
            sv = cd.get("structural_violations") or []
            bv = cd.get("behavioral_violations") or []
            if isinstance(sv, list):
                structural_violations_total += len(sv)
            if isinstance(bv, list):
                behavioral_violations_total += len(bv)

        inv = debug.get("invariant_violations")
        if isinstance(inv, list):
            invariant_violations_total += len(inv)

        pt = item.get("point") or {}
        p = pt.get("p_hat")
        if isinstance(p, (int, float)):
            p_hats.append(float(p))
        rl = pt.get("rail_low")
        rh = pt.get("rail_high")
        if isinstance(rl, (int, float)):
            rail_lows.append(float(rl))
        if isinstance(rh, (int, float)):
            rail_highs.append(float(rh))

    contract_diagnostics_key_presence_rates = {
        key: (
            contract_diagnostics_key_presence_counts[key] / points_with_contract_diagnostics
            if points_with_contract_diagnostics > 0
            else 0.0
        )
        for key in CONTRACT_DIAGNOSTIC_REQUIRED_KEYS
    }
    contract_diagnostics_spec_required_keys = _load_engine_spec_required_keys()

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_class": _fixture_class_from_path(replay_path_str),
        "replay_path": replay_path_str,
        "replay_path_exists": path.exists(),
        "assessment_prematch_map": prematch_map_value,
        "replay_contract_policy": getattr(config, "replay_contract_policy", "reject_point_like"),
        "replay_point_transition_enabled": bool(getattr(config, "replay_point_transition_enabled", False)),
        "replay_point_transition_sunset_epoch": getattr(config, "replay_point_transition_sunset_epoch", None),
        "direct_load_payload_count": direct_load_count,
        "replay_payload_count_loaded": len(runner._replay_payloads),
        "total_points_captured": total_points,
        "raw_contract_points": raw_contract,
        "point_passthrough_points": point_passthrough,
        "unknown_replay_mode_points": unknown_replay_mode_points,
        "rail_input_contract_policy_counts": rail_input_contract_policy_counts,
        "rail_input_active_endpoint_semantics_counts": rail_input_active_endpoint_semantics_counts,
        "rail_input_reason_code_counts": rail_input_reason_code_counts,
        "rail_input_v2_activated_points": rail_input_v2_activated_points,
        "rail_input_v1_fallback_points": rail_input_v1_fallback_points,
        "carryover_evidence_by_source_class": carryover_evidence_by_source_class,
        "carryover_completeness_required_fields": list(RAIL_INPUT_V2_REQUIRED_FIELDS),
        "non_canonical_point_points": non_canonical_point_points,
        "replay_quarantine_status_counts": replay_quarantine_status_counts,
        "replay_mode_usage_matrix": {
            "raw_contract": raw_contract,
            "point_passthrough": point_passthrough,
            "non_canonical_point": non_canonical_point_points,
            "unknown": unknown_replay_mode_points,
            "total_points": total_points,
        },
        "point_like_inputs_seen": int(replay_contract_status.get("point_like_inputs_seen", 0)),
        "point_like_inputs_rejected": int(replay_contract_status.get("point_like_inputs_rejected", 0)),
        "point_like_inputs_transition_passthrough": int(
            replay_contract_status.get("point_like_inputs_transition_passthrough", 0)
        ),
        "point_like_reject_reason_counts": replay_contract_status.get("point_like_reject_reason_counts", {}),
        "raw_mode_point_like_skipped": int(replay_contract_status.get("raw_mode_point_like_skipped", 0)),
        "points_with_contract_diagnostics": points_with_contract_diagnostics,
        "contract_diagnostics_spec_required_keys": contract_diagnostics_spec_required_keys,
        "contract_diagnostics_required_keys": CONTRACT_DIAGNOSTIC_REQUIRED_KEYS,
        "contract_diagnostics_key_presence_counts": contract_diagnostics_key_presence_counts,
        "contract_diagnostics_missing_key_counts": contract_diagnostics_missing_key_counts,
        "contract_diagnostics_key_presence_rates": contract_diagnostics_key_presence_rates,
        "structural_violations_total": structural_violations_total,
        "behavioral_violations_total": behavioral_violations_total,
        "invariant_violations_total": invariant_violations_total,
        "p_hat_min": min(p_hats) if p_hats else None,
        "p_hat_max": max(p_hats) if p_hats else None,
        "p_hat_count": len(p_hats),
        "p_hat_median": _median(p_hats),
        "rail_low_min": min(rail_lows) if rail_lows else None,
        "rail_high_max": max(rail_highs) if rail_highs else None,
        **({"replay_point_source": _build_replay_point_source(captured)} if include_captured_points else {}),
        **({"captured_points": captured} if include_captured_points else {}),
    }


def main() -> None:
    # 4C: default to raw replay fixture when no arg (for raw-contract verification)
    default_path = str(ROOT / "tools" / "fixtures" / "raw_replay_sample.jsonl")
    replay_path = default_path
    prematch_map: float | None = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--prematch-map" and (i + 1) < len(args):
            try:
                prematch_map = float(args[i + 1])
            except (TypeError, ValueError):
                prematch_map = None
            i += 2
            continue
        if arg.startswith("--prematch-map="):
            raw = arg.split("=", 1)[1]
            try:
                prematch_map = float(raw)
            except (TypeError, ValueError):
                prematch_map = None
            i += 1
            continue
        replay_path = arg
        i += 1
    summary = asyncio.run(run_assessment(replay_path, prematch_map=prematch_map))
    # Deterministic key ordering supports stable artifact diffs.
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


