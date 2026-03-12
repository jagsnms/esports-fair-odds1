from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.align_backend_bo3_active_corpus as alignment


def _row(match_id: int, seq_index: int) -> dict:
    return {
        "schema_version": "backend_bo3_live_capture_contract.v1",
        "live_source": "BO3",
        "capture_ts_iso": f"2026-03-12T05:00:{seq_index:02d}.000Z",
        "match_id": match_id,
        "raw_provider_event_id": f"{match_id}-{seq_index}",
        "raw_seq_index": seq_index,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_alignment_copies_source_into_missing_target(tmp_path: Path) -> None:
    source_path = tmp_path / "logs" / "bo3_backend_live_capture_contract.jsonl"
    target_path = tmp_path / "external" / "bo3_backend_live_capture_contract.jsonl"
    rows = [_row(113437, 1), _row(113437, 2)]
    _write_rows(source_path, rows)

    result = alignment.align_backend_bo3_active_corpus(source_path=source_path, target_path=target_path)

    assert result["action"] == "copy_source_into_missing_target"
    assert result["target_row_count_after"] == 2
    assert alignment._load_rows(target_path) == rows


def test_alignment_replaces_shorter_target_subset_with_source_superset(tmp_path: Path) -> None:
    source_path = tmp_path / "logs" / "bo3_backend_live_capture_contract.jsonl"
    target_path = tmp_path / "external" / "bo3_backend_live_capture_contract.jsonl"
    source_rows = [_row(113437, 1), _row(113437, 2), _row(113437, 3)]
    target_rows = source_rows[:2]
    _write_rows(source_path, source_rows)
    _write_rows(target_path, target_rows)

    result = alignment.align_backend_bo3_active_corpus(source_path=source_path, target_path=target_path)

    assert result["action"] == "replace_target_with_source_superset"
    assert result["target_row_count_before"] == 2
    assert result["target_row_count_after"] == 3
    assert alignment._load_rows(target_path) == source_rows


def test_alignment_leaves_target_alone_when_already_superset(tmp_path: Path) -> None:
    source_path = tmp_path / "logs" / "bo3_backend_live_capture_contract.jsonl"
    target_path = tmp_path / "external" / "bo3_backend_live_capture_contract.jsonl"
    source_rows = [_row(113437, 1), _row(113437, 2)]
    target_rows = source_rows + [_row(113437, 3)]
    _write_rows(source_path, source_rows)
    _write_rows(target_path, target_rows)

    result = alignment.align_backend_bo3_active_corpus(source_path=source_path, target_path=target_path)

    assert result["action"] == "noop_target_already_superset"
    assert result["target_row_count_after"] == 3
    assert alignment._load_rows(target_path) == target_rows


def test_alignment_refuses_divergent_corpora(tmp_path: Path) -> None:
    source_path = tmp_path / "logs" / "bo3_backend_live_capture_contract.jsonl"
    target_path = tmp_path / "external" / "bo3_backend_live_capture_contract.jsonl"
    _write_rows(source_path, [_row(113437, 1), _row(113437, 2)])
    _write_rows(target_path, [_row(113437, 1), _row(999999, 2)])

    with pytest.raises(alignment.CorpusAlignmentError, match="diverge"):
        alignment.align_backend_bo3_active_corpus(source_path=source_path, target_path=target_path)
