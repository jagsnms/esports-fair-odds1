from __future__ import annotations

import json
import os
from pathlib import Path

from backend.services.bo3_capture_contract import default_bo3_backend_capture_path
import tools.run_backend_bo3_corpus_readiness_analyzer as readiness
import tools.run_backend_bo3_live_parity_diagnostic as bounded_diagnostic


def _base_row(
    *,
    match_id: int,
    capture_ts_iso: str,
    provider_event_id: str,
    seq_index: int,
    round_number: int,
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
    loadout_est_total_b: float = 18000.0,
    bo3_health: str = "GOOD",
    bo3_snapshot_status: str = "live",
) -> dict:
    return {
        "schema_version": readiness.CAPTURE_SCHEMA_VERSION,
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
        "bo3_snapshot_status": bo3_snapshot_status,
        "bo3_health": bo3_health,
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


def test_corpus_readiness_rolls_up_multi_match_counts_and_surfaces_cross_match_blockers(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    rows: list[dict] = []
    rows.extend(
        _base_row(
            match_id=113086,
            provider_event_id=f"m1-{idx}",
            seq_index=idx,
            round_number=idx,
            capture_ts_iso=f"2026-03-12T05:00:{idx:02d}.000Z",
        )
        for idx in range(1, 33)
    )
    rows.extend(
        _base_row(
            match_id=113086,
            provider_event_id="m1-missing-side",
            seq_index=101,
            round_number=33,
            capture_ts_iso="2026-03-12T05:01:01.000Z",
            a_side=None,
        )
        for _ in range(2)
    )
    rows.extend(
        _base_row(
            match_id=113084,
            provider_event_id=f"m2-{idx}",
            seq_index=200 + idx,
            round_number=idx,
            capture_ts_iso=f"2026-03-12T06:00:{idx:02d}.000Z",
        )
        for idx in range(1, 13)
    )
    rows.extend(
        _base_row(
            match_id=113084,
            provider_event_id=f"m2-excluded-{idx}",
            seq_index=300 + idx,
            round_number=20 + idx,
            capture_ts_iso=f"2026-03-12T06:01:{idx:02d}.000Z",
            round_phase="FINISHED",
        )
        for idx in range(1, 4)
    )
    _write_rows(capture_path, rows)

    report = readiness.build_backend_bo3_corpus_readiness_report(capture_path)

    assert report["input_artifact"]["path"] == str(capture_path)
    assert report["corpus_summary"]["total_rows"] == len(rows)
    assert report["corpus_summary"]["distinct_match_count"] == 2
    assert report["corpus_summary"]["rows_per_match"] == {"113086": 34, "113084": 15}
    assert report["corpus_summary"]["eligible_compared_row_count"] == 44
    assert report["corpus_summary"]["distinct_raw_event_count"] == 44
    assert report["corpus_summary"]["distinct_signal_active_raw_event_count"] == 44
    assert report["corpus_summary"]["exclusion_reasons"] == {"missing_a_side": 2, "not_in_progress_phase": 3}
    assert "non_primary_match" not in report["corpus_summary"]["exclusion_reasons"]
    assert report["readiness_summary"]["blockage_assessment"] == readiness.BLOCKAGE_EASING_CONCENTRATED

    match_summaries = {summary["match_id"]: summary for summary in report["match_summaries"]}
    assert match_summaries[113086]["readiness_contribution_class"] == readiness.READINESS_MATERIAL
    assert match_summaries[113084]["readiness_contribution_class"] == readiness.READINESS_WEAK
    assert match_summaries[113084]["exclusion_reasons"] == {"not_in_progress_phase": 3}
    assert report["top_blockers"][0]["reason"] == "not_in_progress_phase"
    assert report["top_blockers"][0]["match_counts"] == {"113084": 3}


def test_corpus_readiness_can_show_multi_match_easing_when_two_matches_contribute_material_evidence(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    rows: list[dict] = []
    rows.extend(
        _base_row(
            match_id=113086,
            provider_event_id=f"m1-{idx}",
            seq_index=idx,
            round_number=idx,
            capture_ts_iso=f"2026-03-12T05:10:{idx:02d}.000Z",
        )
        for idx in range(1, 31)
    )
    rows.extend(
        _base_row(
            match_id=113084,
            provider_event_id=f"m2-{idx}",
            seq_index=100 + idx,
            round_number=idx,
            capture_ts_iso=f"2026-03-12T05:11:{idx:02d}.000Z",
        )
        for idx in range(1, 31)
    )
    _write_rows(capture_path, rows)

    report = readiness.build_backend_bo3_corpus_readiness_report(capture_path)

    assert report["readiness_summary"]["blockage_assessment"] == readiness.BLOCKAGE_EASING_MULTI_MATCH
    assert report["readiness_summary"]["materially_usable_match_count"] == 2
    assert report["corpus_summary"]["distinct_match_count"] == 2
    assert report["corpus_summary"]["distinct_raw_event_count"] == 60
    assert report["corpus_summary"]["distinct_signal_active_raw_event_count"] == 60
    assert report["corpus_summary"]["excluded_row_count"] == 0


def test_corpus_readiness_default_path_uses_continuity_protected_corpus_and_preserves_bounded_tool_separation() -> None:
    expected_path = Path(default_bo3_backend_capture_path())
    repo_root = Path(__file__).resolve().parents[2]
    assert readiness.DEFAULT_CAPTURE_PATH == expected_path
    assert readiness.DEFAULT_REPORT_PATH == Path("automation/reports/backend_bo3_corpus_readiness_report.json")
    assert bounded_diagnostic.DEFAULT_CAPTURE_PATH == Path(
        "automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl"
    )
    assert os.path.commonpath([str(expected_path), str(repo_root)]) != str(repo_root)
