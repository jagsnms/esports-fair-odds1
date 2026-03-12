from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.bo3_capture_contract import default_bo3_backend_capture_path

DEFAULT_SOURCE_PATH = Path("logs/bo3_backend_live_capture_contract.jsonl")
DEFAULT_TARGET_PATH = Path(default_bo3_backend_capture_path())


class CorpusAlignmentError(RuntimeError):
    pass


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _is_prefix(prefix_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> bool:
    return len(prefix_rows) <= len(full_rows) and full_rows[: len(prefix_rows)] == prefix_rows


def build_alignment_plan(source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not source_rows and not target_rows:
        return {
            "action": "noop_both_empty",
            "source_row_count": 0,
            "target_row_count_before": 0,
            "target_row_count_after": 0,
            "will_write_target": False,
        }
    if not source_rows:
        return {
            "action": "noop_source_empty",
            "source_row_count": 0,
            "target_row_count_before": len(target_rows),
            "target_row_count_after": len(target_rows),
            "will_write_target": False,
        }
    if not target_rows:
        return {
            "action": "copy_source_into_missing_target",
            "source_row_count": len(source_rows),
            "target_row_count_before": 0,
            "target_row_count_after": len(source_rows),
            "will_write_target": True,
            "merged_rows": source_rows,
        }
    if _is_prefix(source_rows, target_rows):
        return {
            "action": "noop_target_already_superset",
            "source_row_count": len(source_rows),
            "target_row_count_before": len(target_rows),
            "target_row_count_after": len(target_rows),
            "will_write_target": False,
        }
    if _is_prefix(target_rows, source_rows):
        return {
            "action": "replace_target_with_source_superset",
            "source_row_count": len(source_rows),
            "target_row_count_before": len(target_rows),
            "target_row_count_after": len(source_rows),
            "will_write_target": True,
            "merged_rows": source_rows,
        }
    raise CorpusAlignmentError(
        "source and target corpora diverge; refusing blind overwrite or blind duplicate append"
    )


def align_backend_bo3_active_corpus(
    *,
    source_path: Path = DEFAULT_SOURCE_PATH,
    target_path: Path = DEFAULT_TARGET_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not source_path.exists() and not target_path.exists():
        raise CorpusAlignmentError("neither source nor target corpus exists")
    source_rows = _load_rows(source_path) if source_path.exists() else []
    target_rows = _load_rows(target_path) if target_path.exists() else []
    plan = build_alignment_plan(source_rows, target_rows)
    if plan.get("will_write_target") and not dry_run:
        _write_rows(target_path, plan["merged_rows"])
    return {
        "source_path": str(source_path),
        "target_path": str(target_path),
        **{k: v for k, v in plan.items() if k != "merged_rows"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align the old in-repo BO3 corpus into the external continuity-protected active corpus path."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_PATH),
        help="Path to the legacy in-repo BO3 corpus JSONL.",
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET_PATH),
        help="Path to the external active BO3 corpus JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned continuity-safe alignment action without writing the target file.",
    )
    args = parser.parse_args()

    try:
        result = align_backend_bo3_active_corpus(
            source_path=Path(args.source),
            target_path=Path(args.target),
            dry_run=args.dry_run,
        )
    except CorpusAlignmentError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

