import ast
from pathlib import Path
import pandas as pd

from legacy.fair_odds import logs as logs_mod


RAW_RECORD_PATH = "logs/bo3_pulls.jsonl"


def _feed(*, team_a_is_team_one: bool = True, seq_index: int = 42) -> dict:
    return {
        "team_a_is_team_one": team_a_is_team_one,
        "raw_ts_utc": "2026-03-10T12:00:00Z",
        "raw_record_path": RAW_RECORD_PATH,
        "payload": {
            "provider_event_id": "evt-123",
            "seq_index": seq_index,
            "sent_time": "2026-03-10T12:00:00Z",
            "updated_at": "2026-03-10T12:00:01Z",
            "game_number": 2,
            "round_number": 7,
            "round_phase": "IN_PROGRESS",
            "team_one": {
                "id": 101,
                "provider_id": "55754",
                "side": "CT",
            },
            "team_two": {
                "id": 202,
                "provider_id": "56498",
                "side": "TERRORIST",
            },
        },
    }


def _snapshot_row() -> dict:
    return {
        "snapshot_ts_iso": "2026-03-10T12:00:05.000Z",
        "snapshot_ts": "2026-03-10T12:00:05",
        "snapshot_ts_epoch_ms": 1773144005000,
        "match_id": "110456",
        "p_hat": 0.6123,
        "p_hat_map": 0.5876,
        "q_intra_round_win_a": 0.63,
        "q_intra_round_win_a_source": "midround_v2_mixture",
        "rail_p_if_next_round_loss": 0.44,
        "rail_p_if_next_round_win": 0.68,
        "intra_score_alive": 0.07,
        "intra_score_hp": 0.03,
        "intra_score_bomb": 0.0,
        "intra_score_loadout": 1.44,
    }


def _live_frame() -> dict:
    return {
        "game_number": 2,
        "round_number": 7,
        "round_phase": "IN_PROGRESS",
        "a_side": "CT",
        "team_a_side_used": "CT",
        "team_b_side_used": "T",
        "bomb_planted": False,
        "round_time_remaining_s": 43.2,
        "alive_count_a": 4,
        "alive_count_b": 3,
        "hp_alive_total_a": 362.0,
        "hp_alive_total_b": 255.0,
        "cash_total_a": 18250.0,
        "cash_total_b": 17100.0,
        "loadout_est_total_a": 25300.0,
        "loadout_est_total_b": 23800.0,
        "alive_loadout_total_a": 21400.0,
        "alive_loadout_total_b": 17600.0,
        "armor_alive_total_a": 280.0,
        "armor_alive_total_b": 180.0,
        "intraround_state_source": "bo3_player_states",
    }


def test_build_bo3_live_capture_context_preserves_raw_linkage_and_team_mapping():
    context = logs_mod.build_bo3_live_capture_context(_feed(team_a_is_team_one=False, seq_index=99))

    assert context["raw_ts_utc"] == "2026-03-10T12:00:00Z"
    assert context["raw_provider_event_id"] == "evt-123"
    assert context["raw_seq_index"] == 99
    assert context["raw_record_path"] == RAW_RECORD_PATH
    assert context["game_number"] == 2
    assert context["round_number"] == 7
    assert context["team_a_id"] == 202
    assert context["team_b_id"] == 101
    assert context["team_a_provider_id"] == "56498"
    assert context["team_b_provider_id"] == "55754"
    assert context["team_a_side_used"] == "T"
    assert context["team_b_side_used"] == "CT"


def test_bo3_live_capture_contract_persists_append_only_artifact(tmp_path, monkeypatch):
    parquet_path = tmp_path / "cs2_replay_snapshots.parquet"
    monkeypatch.setattr(logs_mod, "CS2_REPLAY_SNAPSHOT_PARQUET_PATH", parquet_path)

    row_one = logs_mod.augment_bo3_live_capture_contract(_snapshot_row(), _feed(seq_index=42), _live_frame())
    row_two = logs_mod.augment_bo3_live_capture_contract(
        {**_snapshot_row(), "snapshot_ts_iso": "2026-03-10T12:00:06.000Z"},
        _feed(seq_index=43),
        _live_frame(),
    )

    assert logs_mod.should_persist_bo3_live_capture_contract(
        row_one,
        bo3_source_mode="LIVE (poller feed file)",
        snapshot_status="live",
    )

    logs_mod.persist_cs2_replay_snapshot(row_one)
    logs_mod.persist_cs2_replay_snapshot(row_two)

    df = pd.read_parquet(parquet_path)

    assert len(df) == 2
    assert list(df["raw_seq_index"]) == [42, 43]
    assert list(df["live_source"]) == ["BO3", "BO3"]
    assert df.loc[0, "schema_version"] == logs_mod.BO3_LIVE_CAPTURE_SCHEMA_VERSION
    assert df.loc[0, "raw_record_path"] == RAW_RECORD_PATH
    assert df.loc[0, "team_a_is_team_one"] == True
    assert df.loc[0, "alive_count_a"] == 4
    assert df.loc[0, "hp_alive_total_b"] == 255.0
    assert df.loc[0, "loadout_est_total_a"] == 25300.0
    assert df.loc[0, "armor_alive_total_b"] == 180.0
    assert df.loc[0, "p_hat"] == 0.6123
    assert df.loc[0, "p_hat_map"] == 0.5876
    assert df.loc[0, "rail_low"] == 0.44
    assert df.loc[0, "rail_high"] == 0.68
    assert df.loc[0, "q_intra_round_win_a_source"] == "midround_v2_mixture"
    assert df.loc[0, "intraround_state_source"] == "bo3_player_states"


def test_should_persist_bo3_live_capture_contract_requires_live_mode_and_raw_identity():
    row = logs_mod.augment_bo3_live_capture_contract(_snapshot_row(), _feed(), _live_frame())

    assert logs_mod.should_persist_bo3_live_capture_contract(
        row,
        bo3_source_mode="LIVE (poller feed file)",
        snapshot_status="live",
    )

    assert not logs_mod.should_persist_bo3_live_capture_contract(
        row,
        bo3_source_mode="REPLAY (from bo3_pulls.jsonl)",
        snapshot_status="replay",
    )

    row_without_raw = dict(row)
    row_without_raw["raw_record_path"] = None
    assert not logs_mod.should_persist_bo3_live_capture_contract(
        row_without_raw,
        bo3_source_mode="LIVE (poller feed file)",
        snapshot_status="live",
    )



def test_bo3_auto_activation_enables_capture_lock_by_default():
    source = Path("legacy/app/app35_ml.py").read_text(encoding="utf-8")

    assert 'st.session_state["cs2_live_config_locked"] = True' in source
    assert 'st.session_state["cs2_live_config_locked_prev"] = False' in source
    assert 'st.session_state["cs2_auto_add_snapshot_this_run"] = True' in source

def test_app35_ml_parses_after_bo3_live_capture_wiring():
    source = Path("legacy/app/app35_ml.py").read_text(encoding="utf-8")
    ast.parse(source)
