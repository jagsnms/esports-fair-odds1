from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.recover_backend_bo3_divergent_corpus as recovery


def _row(match_id: int, provider_event_id: str, seq_index: int, capture_ts_iso: str, *, p_hat: float = 0.5) -> dict:
    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "match_id": match_id,
        "raw_provider_event_id": provider_event_id,
        "raw_seq_index": seq_index,
        "capture_ts_iso": capture_ts_iso,
        "p_hat": p_hat,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_recovery_unions_unique_rows_and_writes_only_when_safe(tmp_path: Path) -> None:
    source_a_path = tmp_path / "active.jsonl"
    source_b_path = tmp_path / "divergent.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    source_a_rows = [_row(113437, "a-1", 1, "2026-03-12T05:00:01.000Z")]
    source_b_rows = [_row(113084, "b-1", 1, "2026-03-11T05:00:01.000Z")]
    _write_rows(source_a_path, source_a_rows)
    _write_rows(source_b_path, source_b_rows)

    report = recovery.recover_backend_bo3_divergent_corpus(
        source_a_path=source_a_path,
        source_b_path=source_b_path,
        output_path=output_path,
        write_output=True,
    )

    assert report["recovery_decision"] == "safe_union_written"
    assert report["unique_row_count_from_a"] == 1
    assert report["unique_row_count_from_b"] == 1
    assert report["identical_duplicate_row_count"] == 0
    assert report["conflict_row_count"] == 0
    merged_rows = recovery._load_rows_from_path(output_path)
    assert merged_rows == source_a_rows + source_b_rows


def test_recovery_dedupes_identical_duplicate_rows(tmp_path: Path) -> None:
    source_a_path = tmp_path / "active.jsonl"
    source_b_path = tmp_path / "divergent.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    shared = _row(113437, "shared-1", 1, "2026-03-12T05:00:01.000Z")
    _write_rows(source_a_path, [shared])
    _write_rows(source_b_path, [shared])

    report = recovery.recover_backend_bo3_divergent_corpus(
        source_a_path=source_a_path,
        source_b_path=source_b_path,
        output_path=output_path,
        write_output=True,
    )

    assert report["recovery_decision"] == "safe_union_written"
    assert report["unique_row_count_from_a"] == 0
    assert report["unique_row_count_from_b"] == 0
    assert report["identical_duplicate_row_count"] == 1
    assert recovery._load_rows_from_path(output_path) == [shared]


def test_recovery_refuses_conflicting_same_identity_rows_and_leaves_output_unchanged(tmp_path: Path) -> None:
    source_a_path = tmp_path / "active.jsonl"
    source_b_path = tmp_path / "divergent.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    source_a_rows = [_row(113437, "shared-1", 1, "2026-03-12T05:00:01.000Z", p_hat=0.51)]
    source_b_rows = [_row(113437, "shared-1", 1, "2026-03-12T05:00:01.000Z", p_hat=0.73)]
    sentinel_rows = [_row(999999, "sentinel", 1, "2026-03-12T06:00:00.000Z")]
    _write_rows(source_a_path, source_a_rows)
    _write_rows(source_b_path, source_b_rows)
    _write_rows(output_path, sentinel_rows)

    report = recovery.recover_backend_bo3_divergent_corpus(
        source_a_path=source_a_path,
        source_b_path=source_b_path,
        output_path=output_path,
        write_output=True,
    )

    assert report["recovery_decision"] == "refused_due_to_conflicts"
    assert report["conflict_row_count"] == 1
    assert recovery._load_rows_from_path(output_path) == sentinel_rows
    assert len(report["conflict_sample"]) == 1


def test_recovery_dry_run_reports_safe_union_without_writing_output(tmp_path: Path) -> None:
    source_a_path = tmp_path / "active.jsonl"
    source_b_path = tmp_path / "divergent.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    _write_rows(source_a_path, [_row(113437, "a-1", 1, "2026-03-12T05:00:01.000Z")])
    _write_rows(source_b_path, [_row(113084, "b-1", 1, "2026-03-11T05:00:01.000Z")])

    report = recovery.recover_backend_bo3_divergent_corpus(
        source_a_path=source_a_path,
        source_b_path=source_b_path,
        output_path=output_path,
        write_output=False,
    )

    assert report["recovery_decision"] == "safe_union_dry_run_only"
    assert report["output_path"] is None
    assert not output_path.exists()
