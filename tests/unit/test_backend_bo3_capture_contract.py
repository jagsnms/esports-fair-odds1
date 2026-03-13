from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from engine.models import Config, Derived, State

from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


def _live_snapshot(*, provider_event_id: str, seq_index: int, updated_at: str, round_number: int, score_a: int, score_b: int) -> dict:
    return {
        "provider_event_id": provider_event_id,
        "seq_index": seq_index,
        "sent_time": updated_at,
        "updated_at": updated_at,
        "game_number": 1,
        "round_number": round_number,
        "round_phase": "IN_PROGRESS",
        "round_time_remaining": 92.0,
        "is_bomb_planted": False,
        "map_name": "Ancient",
        "bo_type": 3,
        "team_one": {
            "id": 101,
            "provider_id": "bo3_team_101",
            "name": "HAVU",
            "side": "CT",
            "score": score_a,
            "match_score": 1,
            "player_states": [
                {"is_alive": True, "health": 100, "balance": 4200, "equipment_value": 3200, "armor": 100},
                {"is_alive": True, "health": 83, "balance": 3500, "equipment_value": 2800, "armor": 75},
            ],
        },
        "team_two": {
            "id": 202,
            "provider_id": "bo3_team_202",
            "name": "ex-Zero Tenacity",
            "side": "T",
            "score": score_b,
            "match_score": 1,
            "player_states": [
                {"is_alive": True, "health": 91, "balance": 3900, "equipment_value": 3000, "armor": 80},
                {"is_alive": True, "health": 76, "balance": 3100, "equipment_value": 2500, "armor": 60},
            ],
        },
    }


def _capture_record(
    *,
    match_id: int,
    round_number: int,
    team_one_id: int = 101,
    team_two_id: int = 202,
    team_a_is_team_one: bool = True,
    provider_event_id: str | None = None,
) -> dict:
    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "capture_ts_iso": f"2026-03-13T05:{round_number:02d}:00.000Z",
        "match_id": match_id,
        "team_a_is_team_one": team_a_is_team_one,
        "raw_provider_event_id": provider_event_id or f"evt-{match_id}-{round_number}",
        "raw_seq_index": round_number,
        "raw_sent_time": f"2026-03-13T05:{round_number:02d}:00.000Z",
        "raw_updated_at": f"2026-03-13T05:{round_number:02d}:00.000Z",
        "raw_snapshot_ts": f"2026-03-13T05:{round_number:02d}:00.000Z",
        "raw_record_path": None,
        "game_number": 1,
        "map_index": 0,
        "round_number": round_number,
        "round_phase": "IN_PROGRESS",
        "team_one_id": team_one_id,
        "team_two_id": team_two_id,
        "team_one_provider_id": f"team_{team_one_id}",
        "team_two_provider_id": f"team_{team_two_id}",
        "a_side": "CT" if team_a_is_team_one else "T",
        "round_score_a": round_number,
        "round_score_b": max(round_number - 1, 0),
        "series_score_a": 1,
        "series_score_b": 0,
        "map_name": "Ancient",
        "series_fmt": "bo3",
        "round_time_remaining_s": 92.0,
        "bomb_planted": False,
        "alive_count_a": 5,
        "alive_count_b": 5,
        "hp_alive_total_a": 500.0,
        "hp_alive_total_b": 500.0,
        "cash_total_a": 20000.0,
        "cash_total_b": 20000.0,
        "loadout_est_total_a": 15000.0,
        "loadout_est_total_b": 15000.0,
        "armor_alive_total_a": 500.0,
        "armor_alive_total_b": 500.0,
        "loadout_source": "ev",
        "round_time_remaining_was_ms": False,
        "round_time_remaining_was_out_of_range": False,
        "round_time_remaining_was_missing": False,
        "p_hat": 0.5,
        "rail_low": 0.4,
        "rail_high": 0.6,
        "series_low": 0.2,
        "series_high": 0.8,
        "bo3_snapshot_status": "live",
        "bo3_health": "GOOD",
        "bo3_health_reason": None,
        "bo3_feed_error": None,
        "q_intra_total": 0.5,
        "midround_weight": 0.75,
        "clamp_reason": "ok",
        "dominance_score": 0.1,
        "fragility_missing_microstate_flag": False,
        "fragility_clock_invalid_flag": False,
    }


async def test_backend_bo3_capture_contract_appends_jsonl_rows(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    capture_path = logs_dir / "bo3_backend_live_capture_contract.jsonl"
    raw_path = logs_dir / "bo3_raw_match_321.jsonl"

    store = MemoryStore(max_history=50)
    config = Config(source="BO3", match_id=321, poll_interval_s=5.0, team_a_is_team_one=True)
    await store.set_current(
        State(config=config, segment_id=1),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.4, rail_high=0.6, kappa=0.0, debug={}),
    )

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)

    snap1 = _live_snapshot(
        provider_event_id="evt-1",
        seq_index=10,
        updated_at="2026-03-10T15:08:12.054Z",
        round_number=5,
        score_a=7,
        score_b=6,
    )
    snap2 = _live_snapshot(
        provider_event_id="evt-2",
        seq_index=11,
        updated_at="2026-03-10T15:08:17.319Z",
        round_number=6,
        score_a=8,
        score_b=6,
    )

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(side_effect=[snap1, snap2])),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            side_effect=[("live", None, snap1["updated_at"], True), ("live", None, snap2["updated_at"], True)],
        ),
        patch("backend.services.runner._BO3_RAW_RECORD_DIR", str(logs_dir)),
        patch("backend.services.runner._BO3_RAW_RECORD_PER_MATCH", True),
        patch("backend.services.runner._BO3_RAW_RECORD_ENABLED", True),
        patch("backend.services.bo3_capture_contract._BO3_BACKEND_CAPTURE_ENABLED", True),
        patch("backend.services.bo3_capture_contract._BO3_BACKEND_CAPTURE_PATH", str(capture_path)),
    ):
        did_first = await runner._tick_bo3(config)
        did_second = await runner._tick_bo3(config)

    assert did_first is True
    assert did_second is True
    assert raw_path.exists(), "raw BO3 path should still be written by the real backend runner"
    assert capture_path.exists(), "backend-native capture artifact should be written"

    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(raw_rows) == 2

    rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2, "accepted live BO3 frames should append one contract row each"

    first = rows[0]
    second = rows[1]

    assert first["schema_version"] == "backend_bo3_live_capture_contract.v1"
    assert first["live_source"] == "BO3"
    assert first["match_id"] == 321
    assert first["team_a_is_team_one"] is True
    assert first["raw_provider_event_id"] == "evt-1"
    assert first["raw_seq_index"] == 10
    assert first["raw_sent_time"] == "2026-03-10T15:08:12.054Z"
    assert first["raw_updated_at"] == "2026-03-10T15:08:12.054Z"
    assert first["raw_snapshot_ts"] == "2026-03-10T15:08:12.054Z"
    assert first["raw_record_path"] == str(raw_path)
    assert first["game_number"] == 1
    assert first["map_index"] == 0
    assert first["round_number"] == 5
    assert first["round_phase"] == "IN_PROGRESS"
    assert first["team_one_id"] == 101
    assert first["team_two_id"] == 202
    assert first["team_one_provider_id"] == "bo3_team_101"
    assert first["team_two_provider_id"] == "bo3_team_202"
    assert first["a_side"] == "CT"
    assert first["round_score_a"] == 7
    assert first["round_score_b"] == 6
    assert first["series_score_a"] == 1
    assert first["series_score_b"] == 1
    assert first["round_time_remaining_s"] == 92.0
    assert first["bomb_planted"] is False
    assert first["alive_count_a"] == 2
    assert first["alive_count_b"] == 2
    assert first["hp_alive_total_a"] == 183.0
    assert first["hp_alive_total_b"] == 167.0
    assert first["cash_total_a"] == 7700.0
    assert first["cash_total_b"] == 7000.0
    assert first["loadout_est_total_a"] == 6000.0
    assert first["loadout_est_total_b"] == 5500.0
    assert first["armor_alive_total_a"] == 175.0
    assert first["armor_alive_total_b"] == 140.0
    assert first["p_hat"] >= first["rail_low"]
    assert first["p_hat"] <= first["rail_high"]
    assert first["bo3_snapshot_status"] == "live"
    assert first["bo3_health"] == "GOOD"
    assert first["bo3_health_reason"] is None
    assert first["clamp_reason"] == "ok"
    assert "clamp_reason" in first
    assert "q_intra_total" in first
    assert "midround_weight" in first
    assert "dominance_score" in first

    assert second["raw_provider_event_id"] == "evt-2"
    assert second["raw_seq_index"] == 11
    assert second["round_number"] == 6
    assert second["round_score_a"] == 8
    assert second["round_score_b"] == 6
    assert second["clamp_reason"] == "ok"


def test_backend_bo3_capture_contract_appends_jsonl_rows_sync(tmp_path: Path) -> None:
    asyncio.run(test_backend_bo3_capture_contract_appends_jsonl_rows(tmp_path))


def test_backend_bo3_capture_contract_default_path_uses_continuity_protected_local_store() -> None:
    from backend.services import bo3_capture_contract

    expected_path = os.path.normpath(
        os.path.join(
            os.environ["LOCALAPPDATA"],
            "EsportsFairOdds",
            "corpus",
            "bo3_backend_live_capture_contract.jsonl",
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    assert bo3_capture_contract._BO3_BACKEND_CAPTURE_PATH == expected_path
    assert bo3_capture_contract.default_bo3_backend_capture_path() == expected_path
    assert os.path.commonpath([expected_path, str(repo_root)]) != str(repo_root)


def test_backend_bo3_capture_contract_keeps_same_match_identity_stable(tmp_path: Path) -> None:
    from backend.services import bo3_capture_contract

    capture_path = tmp_path / "bo3_backend_live_capture_contract.jsonl"
    conflict_path = tmp_path / "bo3_backend_live_capture_contract_identity_conflicts.jsonl"
    bo3_capture_contract._BO3_MATCH_IDENTITY_LOCKS.clear()

    stable_first = _capture_record(match_id=113092, round_number=12)
    stable_second = _capture_record(match_id=113092, round_number=13)

    with patch("backend.services.bo3_capture_contract._BO3_BACKEND_CAPTURE_PATH", str(capture_path)):
        first_path = bo3_capture_contract.append_bo3_live_capture_record(stable_first)
        second_path = bo3_capture_contract.append_bo3_live_capture_record(stable_second)

    assert first_path == str(capture_path)
    assert second_path == str(capture_path)
    rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert [row["round_number"] for row in rows] == [12, 13]
    assert all(row["team_one_id"] == 101 for row in rows)
    assert all(row["team_two_id"] == 202 for row in rows)
    assert all(row["team_a_is_team_one"] is True for row in rows)
    assert not conflict_path.exists()


def test_backend_bo3_capture_contract_refuses_mid_session_team_flip_and_quarantines_conflict(tmp_path: Path) -> None:
    from backend.services import bo3_capture_contract

    capture_path = tmp_path / "bo3_backend_live_capture_contract.jsonl"
    conflict_path = tmp_path / "bo3_backend_live_capture_contract_identity_conflicts.jsonl"
    bo3_capture_contract._BO3_MATCH_IDENTITY_LOCKS.clear()

    stable_first = _capture_record(match_id=113092, round_number=12, team_one_id=22733, team_two_id=4175)
    conflicting_second = _capture_record(match_id=113092, round_number=13, team_one_id=11042, team_two_id=22733)
    stable_third = _capture_record(match_id=113092, round_number=14, team_one_id=22733, team_two_id=4175)

    with patch("backend.services.bo3_capture_contract._BO3_BACKEND_CAPTURE_PATH", str(capture_path)):
        first_path = bo3_capture_contract.append_bo3_live_capture_record(stable_first)
        conflict_result = bo3_capture_contract.append_bo3_live_capture_record(conflicting_second)
        third_path = bo3_capture_contract.append_bo3_live_capture_record(stable_third)

    assert first_path == str(capture_path)
    assert conflict_result is None
    assert third_path == str(capture_path)

    kept_rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept_rows) == 2
    assert [row["round_number"] for row in kept_rows] == [12, 14]
    assert all(row["team_one_id"] == 22733 for row in kept_rows)
    assert all(row["team_two_id"] == 4175 for row in kept_rows)
    assert all(row["team_a_is_team_one"] is True for row in kept_rows)

    conflict_rows = [json.loads(line) for line in conflict_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(conflict_rows) == 1
    conflict = conflict_rows[0]
    assert conflict["schema_version"] == "backend_bo3_live_capture_identity_conflict.v1"
    assert conflict["match_id"] == 113092
    assert conflict["canonical_team_one_id"] == 22733
    assert conflict["canonical_team_two_id"] == 4175
    assert conflict["canonical_team_a_is_team_one"] is True
    assert conflict["conflicting_team_one_id"] == 11042
    assert conflict["conflicting_team_two_id"] == 22733
    assert conflict["conflicting_team_a_is_team_one"] is True
    assert conflict["reason"] == "team_identity_conflict_same_match_id"


def test_backend_bo3_capture_contract_restart_seeding_refuses_conflict_from_existing_file(tmp_path: Path) -> None:
    from backend.services import bo3_capture_contract

    capture_path = tmp_path / "bo3_backend_live_capture_contract.jsonl"
    conflict_path = tmp_path / "bo3_backend_live_capture_contract_identity_conflicts.jsonl"

    existing_canonical = _capture_record(match_id=113092, round_number=12, team_one_id=22733, team_two_id=4175)
    capture_path.write_text(json.dumps(existing_canonical) + "\n", encoding="utf-8")
    bo3_capture_contract._BO3_MATCH_IDENTITY_LOCKS.clear()

    conflicting_later = _capture_record(match_id=113092, round_number=13, team_one_id=11042, team_two_id=22733)

    with patch("backend.services.bo3_capture_contract._BO3_BACKEND_CAPTURE_PATH", str(capture_path)):
        result = bo3_capture_contract.append_bo3_live_capture_record(conflicting_later)

    assert result is None
    kept_rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept_rows) == 1
    assert kept_rows[0]["round_number"] == 12
    assert kept_rows[0]["team_one_id"] == 22733
    assert kept_rows[0]["team_two_id"] == 4175

    conflict_rows = [json.loads(line) for line in conflict_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(conflict_rows) == 1
    conflict = conflict_rows[0]
    assert conflict["match_id"] == 113092
    assert conflict["canonical_team_one_id"] == 22733
    assert conflict["canonical_team_two_id"] == 4175
    assert conflict["conflicting_team_one_id"] == 11042
    assert conflict["conflicting_team_two_id"] == 22733
