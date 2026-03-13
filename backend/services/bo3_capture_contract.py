from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from engine.models import Derived, Frame, HistoryPoint

_BO3_BACKEND_CAPTURE_ENABLED = os.environ.get("BO3_BACKEND_CAPTURE_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
_BO3_BACKEND_CAPTURE_PATH = os.environ.get("BO3_BACKEND_CAPTURE_PATH", "logs/bo3_backend_live_capture_contract.jsonl")
_BO3_CAPTURE_IDENTITY_CONFLICT_SCHEMA_VERSION = "backend_bo3_live_capture_identity_conflict.v1"
_BO3_MATCH_IDENTITY_LOCKS: dict[int, tuple[int, int, bool]] = {}
logger = logging.getLogger(__name__)


def default_bo3_backend_capture_path() -> str:
    configured_path = os.environ.get("BO3_BACKEND_CAPTURE_PATH")
    if configured_path and configured_path.strip():
        return os.path.normpath(configured_path.strip())
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata and local_appdata.strip():
        return os.path.join(
            local_appdata.strip(),
            "EsportsFairOdds",
            "corpus",
            "bo3_backend_live_capture_contract.jsonl",
        )
    return os.path.join(
        os.path.expanduser("~"),
        ".esports-fair-odds",
        "corpus",
        "bo3_backend_live_capture_contract.jsonl",
    )


_BO3_BACKEND_CAPTURE_PATH = default_bo3_backend_capture_path()


def _isoformat_utc(epoch_s: float | None) -> str:
    ts = float(epoch_s) if isinstance(epoch_s, (int, float)) else datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _identity_lock_tuple(record: dict[str, Any]) -> tuple[int, int, bool] | None:
    try:
        team_one_id = int(record["team_one_id"])
        team_two_id = int(record["team_two_id"])
        team_a_is_team_one = bool(record["team_a_is_team_one"])
    except (KeyError, TypeError, ValueError):
        return None
    return (team_one_id, team_two_id, team_a_is_team_one)


def _seed_match_identity_lock_from_file(path: str, match_id: int) -> tuple[int, int, bool] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                try:
                    row_match_id = int(row.get("match_id"))
                except (TypeError, ValueError):
                    continue
                if row_match_id != int(match_id):
                    continue
                identity = _identity_lock_tuple(row)
                if identity is not None:
                    return identity
    except OSError:
        return None
    return None


def _identity_conflict_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    ext = ext or ".jsonl"
    return stem + "_identity_conflicts" + ext


def _append_identity_conflict(
    *,
    path: str,
    match_id: int,
    canonical_identity: tuple[int, int, bool],
    conflicting_record: dict[str, Any],
) -> str:
    conflict_record = {
        "schema_version": _BO3_CAPTURE_IDENTITY_CONFLICT_SCHEMA_VERSION,
        "capture_ts_iso": conflicting_record.get("capture_ts_iso"),
        "match_id": int(match_id),
        "canonical_team_one_id": int(canonical_identity[0]),
        "canonical_team_two_id": int(canonical_identity[1]),
        "canonical_team_a_is_team_one": bool(canonical_identity[2]),
        "conflicting_team_one_id": conflicting_record.get("team_one_id"),
        "conflicting_team_two_id": conflicting_record.get("team_two_id"),
        "conflicting_team_a_is_team_one": conflicting_record.get("team_a_is_team_one"),
        "game_number": conflicting_record.get("game_number"),
        "map_index": conflicting_record.get("map_index"),
        "round_number": conflicting_record.get("round_number"),
        "raw_provider_event_id": conflicting_record.get("raw_provider_event_id"),
        "reason": "team_identity_conflict_same_match_id",
    }
    conflict_path = _identity_conflict_path(path)
    conflict_dir = os.path.dirname(conflict_path)
    if conflict_dir:
        os.makedirs(conflict_dir, exist_ok=True)
    with open(conflict_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(conflict_record, ensure_ascii=False, default=str) + "\n")
    return conflict_path


def _team_identity(point: HistoryPoint, frame: Frame) -> tuple[int | None, int | None, str | None, str | None]:
    team_one_id = point.team_one_id if point.team_one_id is not None else frame.team_one_id
    team_two_id = point.team_two_id if point.team_two_id is not None else frame.team_two_id
    team_one_provider_id = (
        point.team_one_provider_id if point.team_one_provider_id is not None else frame.team_one_provider_id
    )
    team_two_provider_id = (
        point.team_two_provider_id if point.team_two_provider_id is not None else frame.team_two_provider_id
    )
    return team_one_id, team_two_id, team_one_provider_id, team_two_provider_id


def build_bo3_live_capture_record(
    *,
    match_id: int,
    team_a_is_team_one: bool,
    raw_snapshot: dict[str, Any],
    raw_record_path: str | None,
    frame: Frame,
    point: HistoryPoint,
    derived: Derived,
) -> dict[str, Any]:
    debug = derived.debug if isinstance(derived.debug, dict) else {}
    explain = point.explain if isinstance(point.explain, dict) else {}
    final = explain.get("final") if isinstance(explain.get("final"), dict) else {}
    clamp_reason = final.get("clamp_reason")
    if clamp_reason is None:
        clamp_reason = "ok"
    bomb_phase = frame.bomb_phase_time_remaining if isinstance(frame.bomb_phase_time_remaining, dict) else {}
    round_phase = raw_snapshot.get("round_phase") or raw_snapshot.get("phase") or bomb_phase.get("round_phase")
    raw_snapshot_ts = (
        raw_snapshot.get("updated_at")
        or raw_snapshot.get("created_at")
        or raw_snapshot.get("sent_time")
        or raw_snapshot.get("ts")
    )
    team_one_id, team_two_id, team_one_provider_id, team_two_provider_id = _team_identity(point, frame)

    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "capture_ts_iso": _isoformat_utc(point.time),
        "match_id": int(match_id),
        "team_a_is_team_one": bool(team_a_is_team_one),
        "raw_provider_event_id": raw_snapshot.get("provider_event_id"),
        "raw_seq_index": raw_snapshot.get("seq_index"),
        "raw_sent_time": raw_snapshot.get("sent_time"),
        "raw_updated_at": raw_snapshot.get("updated_at"),
        "raw_snapshot_ts": raw_snapshot_ts,
        "raw_record_path": raw_record_path,
        "game_number": point.game_number,
        "map_index": point.map_index,
        "round_number": point.round_number,
        "round_phase": round_phase,
        "team_one_id": team_one_id,
        "team_two_id": team_two_id,
        "team_one_provider_id": team_one_provider_id,
        "team_two_provider_id": team_two_provider_id,
        "a_side": point.a_side if point.a_side is not None else frame.a_side,
        "round_score_a": int(frame.scores[0]),
        "round_score_b": int(frame.scores[1]),
        "series_score_a": int(frame.series_score[0]),
        "series_score_b": int(frame.series_score[1]),
        "map_name": frame.map_name,
        "series_fmt": frame.series_fmt,
        "round_time_remaining_s": frame.round_time_remaining_s,
        "bomb_planted": bomb_phase.get("is_bomb_planted"),
        "alive_count_a": int(frame.alive_counts[0]),
        "alive_count_b": int(frame.alive_counts[1]),
        "hp_alive_total_a": float(frame.hp_totals[0]),
        "hp_alive_total_b": float(frame.hp_totals[1]),
        "cash_total_a": float(frame.cash_totals[0]) if frame.cash_totals is not None else None,
        "cash_total_b": float(frame.cash_totals[1]) if frame.cash_totals is not None else None,
        "loadout_est_total_a": float(frame.loadout_totals[0]) if frame.loadout_totals is not None else None,
        "loadout_est_total_b": float(frame.loadout_totals[1]) if frame.loadout_totals is not None else None,
        "armor_alive_total_a": float(frame.armor_totals[0]) if frame.armor_totals is not None else None,
        "armor_alive_total_b": float(frame.armor_totals[1]) if frame.armor_totals is not None else None,
        "loadout_source": frame.loadout_source,
        "round_time_remaining_was_ms": frame.round_time_remaining_was_ms,
        "round_time_remaining_was_out_of_range": frame.round_time_remaining_was_out_of_range,
        "round_time_remaining_was_missing": frame.round_time_remaining_was_missing,
        "p_hat": float(derived.p_hat),
        "rail_low": float(derived.rail_low),
        "rail_high": float(derived.rail_high),
        "series_low": float(derived.bound_low),
        "series_high": float(derived.bound_high),
        "bo3_snapshot_status": debug.get("bo3_snapshot_status"),
        "bo3_health": debug.get("bo3_health"),
        "bo3_health_reason": debug.get("bo3_health_reason"),
        "bo3_feed_error": debug.get("bo3_feed_error"),
        "q_intra_total": explain.get("q_intra_total"),
        "midround_weight": explain.get("midround_weight"),
        "clamp_reason": clamp_reason,
        "dominance_score": debug.get("dominance_score"),
        "fragility_missing_microstate_flag": ((debug.get("fragility") or {}).get("missing_microstate_flag")),
        "fragility_clock_invalid_flag": ((debug.get("fragility") or {}).get("clock_invalid_flag")),
    }


def append_bo3_live_capture_record(record: dict[str, Any]) -> str | None:
    if not _BO3_BACKEND_CAPTURE_ENABLED:
        return None
    path = _BO3_BACKEND_CAPTURE_PATH
    match_id = record.get("match_id")
    identity = _identity_lock_tuple(record)
    if identity is not None and match_id is not None:
        try:
            match_id_int = int(match_id)
        except (TypeError, ValueError):
            match_id_int = None
        if match_id_int is not None:
            canonical_identity = _BO3_MATCH_IDENTITY_LOCKS.get(match_id_int)
            if canonical_identity is None:
                canonical_identity = _seed_match_identity_lock_from_file(path, match_id_int)
                if canonical_identity is not None:
                    _BO3_MATCH_IDENTITY_LOCKS[match_id_int] = canonical_identity
            if canonical_identity is None:
                _BO3_MATCH_IDENTITY_LOCKS[match_id_int] = identity
            elif canonical_identity != identity:
                conflict_path = _append_identity_conflict(
                    path=path,
                    match_id=match_id_int,
                    canonical_identity=canonical_identity,
                    conflicting_record=record,
                )
                logger.warning(
                    "BO3 capture identity conflict match_id=%s canonical=%s conflicting=%s conflict_path=%s",
                    match_id_int,
                    canonical_identity,
                    identity,
                    conflict_path,
                )
                return None
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path
