from __future__ import annotations

import json
from pathlib import Path

import tools.run_backend_bo3_live_parity_diagnostic as diagnostic


def _base_row(
    *,
    match_id: int = 113437,
    capture_ts_iso: str = "2026-03-10T22:58:18.578Z",
    round_number: int = 12,
    provider_event_id: str = "evt-1",
    seq_index: int = 101,
    q_intra_total: float | None = 0.5,
    clamp_reason: str = "ok",
    round_phase: str = "IN_PROGRESS",
    a_side: str | None = "CT",
    bomb_planted: bool = False,
    alive_count_a: int = 5,
    alive_count_b: int = 5,
    hp_alive_total_a: float = 500.0,
    hp_alive_total_b: float = 500.0,
    loadout_est_total_a: float = 32000.0,
    loadout_est_total_b: float = 24000.0,
) -> dict:
    return {
        "schema_version": diagnostic.CAPTURE_SCHEMA_VERSION,
        "live_source": "BO3",
        "capture_ts_iso": capture_ts_iso,
        "match_id": match_id,
        "team_a_is_team_one": True,
        "raw_provider_event_id": provider_event_id,
        "raw_seq_index": seq_index,
        "raw_sent_time": capture_ts_iso,
        "raw_updated_at": capture_ts_iso,
        "raw_snapshot_ts": capture_ts_iso,
        "raw_record_path": f"logs\\bo3_raw_match_{match_id}.jsonl",
        "game_number": 1,
        "map_index": 0,
        "round_number": round_number,
        "round_phase": round_phase,
        "team_one_id": 1,
        "team_two_id": 2,
        "team_one_provider_id": "team-1",
        "team_two_provider_id": "team-2",
        "a_side": a_side,
        "round_score_a": 8,
        "round_score_b": 7,
        "series_score_a": 1,
        "series_score_b": 0,
        "map_name": "de_ancient",
        "series_fmt": "bo3",
        "round_time_remaining_s": 65.0,
        "bomb_planted": bomb_planted,
        "alive_count_a": alive_count_a,
        "alive_count_b": alive_count_b,
        "hp_alive_total_a": hp_alive_total_a,
        "hp_alive_total_b": hp_alive_total_b,
        "cash_total_a": 18000.0,
        "cash_total_b": 16000.0,
        "loadout_est_total_a": loadout_est_total_a,
        "loadout_est_total_b": loadout_est_total_b,
        "armor_alive_total_a": 500.0,
        "armor_alive_total_b": 500.0,
        "loadout_source": "ev",
        "round_time_remaining_was_ms": False,
        "round_time_remaining_was_out_of_range": False,
        "round_time_remaining_was_missing": False,
        "p_hat": 0.5,
        "rail_low": 0.35,
        "rail_high": 0.65,
        "series_low": 0.000001,
        "series_high": 0.999999,
        "bo3_snapshot_status": "live",
        "bo3_health": "GOOD",
        "bo3_health_reason": None,
        "bo3_feed_error": None,
        "q_intra_total": q_intra_total,
        "midround_weight": 1.0,
        "clamp_reason": clamp_reason,
        "dominance_score": 0.5,
        "fragility_missing_microstate_flag": False,
        "fragility_clock_invalid_flag": False,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_diagnostic_excludes_non_primary_and_unfit_rows_explicitly(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    rows = [
        _base_row(match_id=113437, provider_event_id="evt-a", seq_index=1, round_number=10),
        _base_row(match_id=113437, provider_event_id="evt-b", seq_index=2, round_number=11),
        _base_row(match_id=111953, provider_event_id="evt-c", seq_index=3, round_number=12),
        _base_row(match_id=113437, provider_event_id="evt-d", seq_index=4, round_number=13, q_intra_total=None),
    ]
    _write_rows(capture_path, rows)

    report = diagnostic.build_backend_bo3_live_parity_diagnostic_report(capture_path)

    assert report["input_artifact"]["primary_match_id"] == 113437
    assert report["input_artifact"]["match_id_counts"] == {"113437": 3, "111953": 1}
    assert report["compared_rows"]["eligible_compared_row_count"] == 2
    assert report["compared_rows"]["distinct_raw_event_count"] == 2
    assert report["compared_rows"]["exclusion_reasons"]["non_primary_match"] == 1
    assert report["compared_rows"]["exclusion_reasons"]["missing_current_q_intra"] == 1
    assert report["decision"] in diagnostic.ALLOWED_DECISIONS


def test_diagnostic_marks_loadout_sensitive_pattern_materially_wrong(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    rows = [
        _base_row(
            match_id=113437,
            provider_event_id=f"evt-{idx}",
            seq_index=idx,
            round_number=idx,
            capture_ts_iso=f"2026-03-10T22:58:{idx:02d}.000Z",
            loadout_est_total_a=32000.0,
            loadout_est_total_b=18000.0,
            q_intra_total=0.5,
        )
        for idx in range(1, 41)
    ]
    _write_rows(capture_path, rows)

    report = diagnostic.build_backend_bo3_live_parity_diagnostic_report(capture_path)

    assert report["decision"] == diagnostic.DECISION_MATERIALLY_WRONG
    assert report["compared_rows"]["eligible_compared_row_count"] == 40
    assert report["compared_rows"]["distinct_raw_event_count"] == 40
    assert report["compared_rows"]["distinct_signal_active_raw_event_count"] == 40
    assert report["mismatch_class_counts"]["loadout_sensitive_large_gap"] >= 1


def test_diagnostic_does_not_treat_duplicate_ticks_as_independent_evidence(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    rows: list[dict] = []
    for idx in range(1, 11):
        for dup in range(4):
            rows.append(
                _base_row(
                    match_id=113437,
                    provider_event_id=f"evt-{idx}",
                    seq_index=idx,
                    round_number=idx,
                    capture_ts_iso=f"2026-03-10T22:{idx:02d}:{dup:02d}.000Z",
                    loadout_est_total_a=32000.0,
                    loadout_est_total_b=18000.0,
                    q_intra_total=0.5,
                )
            )
    _write_rows(capture_path, rows)

    report = diagnostic.build_backend_bo3_live_parity_diagnostic_report(capture_path)

    assert report["compared_rows"]["eligible_compared_row_count"] == 40
    assert report["compared_rows"]["distinct_raw_event_count"] == 10
    assert report["compared_rows"]["distinct_signal_active_raw_event_count"] == 10
    assert report["decision"] == diagnostic.DECISION_INCONCLUSIVE
    assert "duplicate ticks are not independent evidence" in report["reasons"][1]
