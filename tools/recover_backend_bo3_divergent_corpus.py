from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.bo3_capture_contract import default_bo3_backend_capture_path

SCHEMA_VERSION = "backend_bo3_divergent_corpus_recovery_report.v1"
DEFAULT_SOURCE_A_PATH = Path(default_bo3_backend_capture_path())
DEFAULT_STASH_CORPUS_PATH = Path("logs/bo3_backend_live_capture_contract.jsonl")
DEFAULT_OUTPUT_PATH = Path(default_bo3_backend_capture_path())
CONFLICT_SAMPLE_LIMIT = 5


class CorpusRecoveryError(RuntimeError):
    pass


def _load_rows_from_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def _load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    return _load_rows_from_text(path.read_text(encoding="utf-8"))


def _load_rows_from_stash(stash_ref: str, stash_path: Path) -> list[dict[str, Any]]:
    spec = f"{stash_ref}:{stash_path.as_posix()}"
    result = subprocess.run(["git", "show", spec], capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise CorpusRecoveryError(
            f"failed to read stash corpus '{spec}': {result.stderr.strip() or result.stdout.strip() or 'unknown git error'}"
        )
    return _load_rows_from_text(result.stdout)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _row_identity(row: dict[str, Any]) -> tuple[int, str, int, str]:
    try:
        match_id = int(row["match_id"])
        provider_event_id = str(row["raw_provider_event_id"])
        seq_index = int(row["raw_seq_index"])
        capture_ts_iso = str(row["capture_ts_iso"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusRecoveryError(f"row missing required recovery identity fields: {exc}") from exc
    return (match_id, provider_event_id, seq_index, capture_ts_iso)


def _identity_key(identity: tuple[int, str, int, str]) -> str:
    return json.dumps(
        {
            "match_id": identity[0],
            "raw_provider_event_id": identity[1],
            "raw_seq_index": identity[2],
            "capture_ts_iso": identity[3],
        },
        sort_keys=True,
    )


def _index_rows(rows: list[dict[str, Any]], *, source_label: str) -> tuple[dict[tuple[int, str, int, str], dict[str, Any]], int]:
    indexed: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    identical_duplicate_count = 0
    for row in rows:
        identity = _row_identity(row)
        existing = indexed.get(identity)
        if existing is None:
            indexed[identity] = row
            continue
        if existing == row:
            identical_duplicate_count += 1
            continue
        raise CorpusRecoveryError(
            f"{source_label} contains conflicting rows for identity {_identity_key(identity)}"
        )
    return indexed, identical_duplicate_count


def build_recovery_report(
    *,
    source_a_rows: list[dict[str, Any]],
    source_b_rows: list[dict[str, Any]],
    source_a_path: str,
    source_b_path: str,
    output_path: str,
    write_output: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    source_a_index, source_a_internal_dupes = _index_rows(source_a_rows, source_label="source_a")
    source_b_index, source_b_internal_dupes = _index_rows(source_b_rows, source_label="source_b")

    unique_a = 0
    unique_b = 0
    cross_source_identical_duplicates = 0
    conflicts: list[dict[str, Any]] = []

    merged_rows = [source_a_index[_row_identity(row)] for row in source_a_rows if _row_identity(row) in source_a_index]
    seen = set()
    deduped_source_a_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        identity = _row_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        deduped_source_a_rows.append(row)
    merged_rows = deduped_source_a_rows

    for identity, row_a in source_a_index.items():
        row_b = source_b_index.get(identity)
        if row_b is None:
            unique_a += 1
            continue
        if row_a == row_b:
            cross_source_identical_duplicates += 1
            continue
        conflicts.append(
            {
                "identity": _identity_key(identity),
                "source_a": row_a,
                "source_b": row_b,
            }
        )

    if not conflicts:
        source_a_identities = set(source_a_index)
        for row in source_b_rows:
            identity = _row_identity(row)
            if identity in source_a_identities:
                continue
            unique_b += 1
            merged_rows.append(row)

    identical_duplicate_row_count = source_a_internal_dupes + source_b_internal_dupes + cross_source_identical_duplicates
    conflict_row_count = len(conflicts)

    if conflicts:
        decision = "refused_due_to_conflicts"
        reasons = [f"refused recovery because {conflict_row_count} row identity conflicts were found"]
        safe_rows = None
        final_output_path: str | None = None
    elif write_output:
        decision = "safe_union_written"
        reasons = [
            f"kept {unique_a} unique source A rows",
            f"added {unique_b} unique source B rows",
            f"deduped {identical_duplicate_row_count} identical duplicate rows",
        ]
        safe_rows = merged_rows
        final_output_path = output_path
    else:
        decision = "safe_union_dry_run_only"
        reasons = [
            f"safe union available with {unique_a} unique source A rows and {unique_b} unique source B rows",
            f"deduped {identical_duplicate_row_count} identical duplicate rows",
        ]
        safe_rows = None
        final_output_path = None

    report = {
        "schema_version": SCHEMA_VERSION,
        "source_a_path": source_a_path,
        "source_b_path": source_b_path,
        "source_a_row_count": len(source_a_rows),
        "source_b_row_count": len(source_b_rows),
        "unique_row_count_from_a": unique_a,
        "unique_row_count_from_b": unique_b,
        "identical_duplicate_row_count": identical_duplicate_row_count,
        "conflict_row_count": conflict_row_count,
        "recovery_decision": decision,
        "output_path": final_output_path,
        "reasons": reasons,
        "conflict_sample": {
            item["identity"]: {"source_a": item["source_a"], "source_b": item["source_b"]}
            for item in conflicts[:CONFLICT_SAMPLE_LIMIT]
        },
    }
    return report, safe_rows


def recover_backend_bo3_divergent_corpus(
    *,
    source_a_path: Path = DEFAULT_SOURCE_A_PATH,
    source_b_path: Path | None = None,
    source_b_stash_ref: str | None = None,
    stash_corpus_path: Path = DEFAULT_STASH_CORPUS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    write_output: bool = False,
) -> dict[str, Any]:
    if source_b_path is None and source_b_stash_ref is None:
        raise CorpusRecoveryError("one of source_b_path or source_b_stash_ref is required")
    if source_b_path is not None and source_b_stash_ref is not None:
        raise CorpusRecoveryError("source_b_path and source_b_stash_ref are mutually exclusive")
    if not source_a_path.exists():
        raise CorpusRecoveryError(f"source_a path does not exist: {source_a_path}")

    source_a_rows = _load_rows_from_path(source_a_path)
    if source_b_path is not None:
        if not source_b_path.exists():
            raise CorpusRecoveryError(f"source_b path does not exist: {source_b_path}")
        source_b_rows = _load_rows_from_path(source_b_path)
        source_b_descriptor = str(source_b_path)
    else:
        source_b_rows = _load_rows_from_stash(source_b_stash_ref, stash_corpus_path)
        source_b_descriptor = f"{source_b_stash_ref}:{stash_corpus_path.as_posix()}"

    report, safe_rows = build_recovery_report(
        source_a_rows=source_a_rows,
        source_b_rows=source_b_rows,
        source_a_path=str(source_a_path),
        source_b_path=source_b_descriptor,
        output_path=str(output_path),
        write_output=write_output,
    )
    if safe_rows is not None:
        _write_rows(output_path, safe_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a special one-off divergent BO3 corpus recovery union with refusal on unresolved row conflicts."
    )
    parser.add_argument(
        "--source-a-path",
        default=str(DEFAULT_SOURCE_A_PATH),
        help="Path to the current active BO3 corpus baseline.",
    )
    parser.add_argument(
        "--source-b-path",
        default=None,
        help="Path to a divergent BO3 corpus branch to union against the active corpus.",
    )
    parser.add_argument(
        "--source-b-stash-ref",
        default=None,
        help="Git stash ref containing a divergent BO3 corpus branch, for example stash@{1}.",
    )
    parser.add_argument(
        "--stash-corpus-path",
        default=str(DEFAULT_STASH_CORPUS_PATH),
        help="Path inside the stash ref to the BO3 corpus JSONL.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the recovered corpus only when --write-output is set and no conflicts exist.",
    )
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="Write the recovered corpus only if the union is safe and conflict-free. Default is dry-run only.",
    )
    args = parser.parse_args()

    try:
        report = recover_backend_bo3_divergent_corpus(
            source_a_path=Path(args.source_a_path),
            source_b_path=None if args.source_b_path is None else Path(args.source_b_path),
            source_b_stash_ref=args.source_b_stash_ref,
            stash_corpus_path=Path(args.stash_corpus_path),
            output_path=Path(args.output_path),
            write_output=bool(args.write_output),
        )
    except CorpusRecoveryError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
