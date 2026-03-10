#!/usr/bin/env python3
"""
Export gate-ready calibration reliability evidence with provenance manifest.

This exporter is intentionally narrow:
- Reads only approved upstream calibration outputs.
- Emits one replay evidence JSON and one explicit empty simulation evidence JSON when
  no true simulation calibration source exists.
- Emits one provenance manifest with source/output hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")

SCHEMA_VERSION = "calibration_reliability_evidence_export.v1"
EXPORTER_IDENTITY = "tools/export_calibration_reliability_evidence.py"
EXPORTER_VERSION = "v1"

DEFAULT_SOURCE_REPORT = ROOT / "config" / "p_calibration_report.json"
DEFAULT_SOURCE_CALIBRATION = ROOT / "config" / "p_calibration.json"
DEFAULT_REPLAY_OUTPUT = ROOT / "tools" / "fixtures" / "calibration_reliability_replay_exported_v1.json"
DEFAULT_SIMULATION_OUTPUT = ROOT / "tools" / "fixtures" / "calibration_reliability_simulation_exported_v1.json"
DEFAULT_MANIFEST_OUTPUT = ROOT / "automation" / "reports" / "calibration_reliability_evidence_export_manifest_v1.json"

GAME_TO_SURFACE = {
    "cs2": "replay",
}
BIN_PRESERVATION_POLICY = "preserve_full_source_bin_coverage"
NULL_EMPTY_BIN_POLICY = "retain_all_bins; for n==0 with null p_mean/y_rate, export 0.0 and preserve source_* fields"
SIMULATION_EXPORT_STATUS = "disabled_no_true_simulation_source"
SIMULATION_EXPORT_REASON = "true simulation calibration evidence is not available from the approved source report"


@dataclass(frozen=True)
class ExportResult:
    exit_code: int
    message: str
    replay_output_path: Path | None = None
    simulation_output_path: Path | None = None
    manifest_output_path: Path | None = None


def _now_iso_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} missing: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} invalid JSON: {path} ({exc})") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be top-level JSON object: {path}")
    return loaded


def _normalize_bin_value(*, value: Any, count: int, field_name: str, game: str, scope: str) -> tuple[float, bool]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value), False
    if value is None and count == 0:
        return 0.0, True
    raise ValueError(
        f"non-finite {field_name} for game={game} scope={scope} count={count}: {value!r}"
    )


def _normalize_bins(*, bins: Any, game: str, scope: str) -> list[dict[str, Any]]:
    if not isinstance(bins, list):
        raise ValueError(f"reliability bins must be list for game={game} scope={scope}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(bins):
        if not isinstance(row, dict):
            raise ValueError(f"reliability bin row must be object for game={game} scope={scope} index={idx}")
        if "n" not in row:
            raise ValueError(f"missing n in reliability row for game={game} scope={scope} index={idx}")
        if "bin" not in row:
            raise ValueError(f"missing bin in reliability row for game={game} scope={scope} index={idx}")
        count_raw = row.get("n")
        bin_raw = row.get("bin")
        if isinstance(count_raw, bool) or not isinstance(count_raw, (int, float)):
            raise ValueError(f"invalid n in reliability row for game={game} scope={scope} index={idx}")
        if isinstance(bin_raw, bool) or not isinstance(bin_raw, (int, float)):
            raise ValueError(f"invalid bin in reliability row for game={game} scope={scope} index={idx}")
        count = int(count_raw)
        bin_index = int(bin_raw)
        if count < 0:
            raise ValueError(f"negative n in reliability row for game={game} scope={scope} index={idx}")

        p_mean_raw = row.get("p_mean")
        y_rate_raw = row.get("y_rate")
        mean_prediction, p_was_null = _normalize_bin_value(
            value=p_mean_raw, count=count, field_name="p_mean", game=game, scope=scope
        )
        empirical_rate, y_was_null = _normalize_bin_value(
            value=y_rate_raw, count=count, field_name="y_rate", game=game, scope=scope
        )
        out.append(
            {
                "bin_index": bin_index,
                "count": count,
                "mean_prediction": mean_prediction,
                "empirical_rate": empirical_rate,
                "source_p_mean": p_mean_raw,
                "source_y_rate": y_rate_raw,
                "source_null_fill_applied": bool(p_was_null or y_was_null),
            }
        )
    return out


def _build_record(
    *,
    report_games: dict[str, Any],
    game: str,
    evidence_source: str,
    evaluation_scope: str,
    seed: int | None,
) -> dict[str, Any]:
    if game not in report_games or not isinstance(report_games[game], dict):
        raise ValueError(f"missing game block in source report: {game}")
    game_block = report_games[game]

    if evaluation_scope == "baseline":
        metrics_key = "metrics_before"
        reliability_key = "reliability_before"
    elif evaluation_scope == "current":
        metrics_key = "metrics_after"
        reliability_key = "reliability_after"
    else:
        raise ValueError(f"unsupported evaluation_scope: {evaluation_scope}")

    metrics = game_block.get(metrics_key)
    if not isinstance(metrics, dict):
        raise ValueError(f"missing metrics block {metrics_key} for game={game}")
    brier_raw = metrics.get("brier")
    logloss_raw = metrics.get("logloss")
    if not isinstance(brier_raw, (int, float)) or not math.isfinite(float(brier_raw)):
        raise ValueError(f"missing/invalid metrics {metrics_key}.brier for game={game}")
    if not isinstance(logloss_raw, (int, float)) or not math.isfinite(float(logloss_raw)):
        raise ValueError(f"missing/invalid metrics {metrics_key}.logloss for game={game}")

    reliability_bins = _normalize_bins(
        bins=game_block.get(reliability_key),
        game=game,
        scope=evaluation_scope,
    )
    return {
        "evidence_source": evidence_source,
        "dataset_id": f"{game}_calibration_report_v1",
        "evaluation_scope": evaluation_scope,
        "seed": seed,
        "segment": "global",
        "brier_score": float(brier_raw),
        "log_loss": float(logloss_raw),
        "reliability_curve_bins": reliability_bins,
    }


def run_export(
    *,
    source_report_path: Path,
    source_calibration_path: Path | None,
    replay_output_path: Path,
    simulation_output_path: Path,
    manifest_output_path: Path,
    baseline_ref: str,
    current_ref: str,
    run_id: str,
    generated_at: str | None,
    simulation_seed: int,
) -> ExportResult:
    run_id_norm = str(run_id or "")
    if RUN_ID_PATTERN.fullmatch(run_id_norm) is None:
        return ExportResult(exit_code=1, message="invalid run_id: must match [a-z0-9_-]+")
    if not isinstance(baseline_ref, str) or not baseline_ref.strip():
        return ExportResult(exit_code=1, message="invalid baseline_ref: must be non-empty string")
    if not isinstance(current_ref, str) or not current_ref.strip():
        return ExportResult(exit_code=1, message="invalid current_ref: must be non-empty string")

    try:
        source_report = _read_json_object(source_report_path, label="source report")
    except ValueError as exc:
        return ExportResult(exit_code=1, message=str(exc))

    source_calibration_sha = None
    source_calibration_path_str = None
    if source_calibration_path is not None and source_calibration_path.exists():
        source_calibration_path_str = str(source_calibration_path)
        source_calibration_sha = _sha256(source_calibration_path)

    games = source_report.get("games")
    if not isinstance(games, dict):
        return ExportResult(exit_code=1, message="source report missing top-level games object")

    try:
        replay_records = [
            _build_record(
                report_games=games,
                game="cs2",
                evidence_source="replay",
                evaluation_scope="baseline",
                seed=None,
            ),
            _build_record(
                report_games=games,
                game="cs2",
                evidence_source="replay",
                evaluation_scope="current",
                seed=None,
            ),
        ]
        simulation_records: list[dict[str, Any]] = []
    except ValueError as exc:
        return ExportResult(exit_code=1, message=str(exc))

    replay_output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)

    replay_output_path.write_text(
        json.dumps(replay_records, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    simulation_output_path.write_text(
        json.dumps(simulation_records, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    generated_at_value = generated_at or _now_iso_z()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at_value,
        "export_run_id": run_id_norm,
        "exporter_tool_identity": EXPORTER_IDENTITY,
        "exporter_tool_version": EXPORTER_VERSION,
        "source_report_path": str(source_report_path),
        "source_report_sha256": _sha256(source_report_path),
        "source_report_ran_at": source_report.get("ran_at"),
        "source_calibration_path": source_calibration_path_str,
        "source_calibration_sha256": source_calibration_sha,
        "baseline_ref": baseline_ref,
        "current_ref": current_ref,
        "extraction_parameters": {
            "game_to_surface_mapping": GAME_TO_SURFACE,
            "dataset_id_pattern": "{game}_calibration_report_v1",
            "segment_value": "global",
            "simulation_seed": int(simulation_seed),
            "simulation_export_status": SIMULATION_EXPORT_STATUS,
            "simulation_export_reason": SIMULATION_EXPORT_REASON,
            "bin_preservation_policy": BIN_PRESERVATION_POLICY,
            "null_empty_bin_policy": NULL_EMPTY_BIN_POLICY,
        },
        "outputs": {
            "replay_evidence_path": str(replay_output_path),
            "replay_evidence_sha256": _sha256(replay_output_path),
            "replay_record_count": len(replay_records),
            "simulation_evidence_path": str(simulation_output_path),
            "simulation_evidence_sha256": _sha256(simulation_output_path),
            "simulation_record_count": len(simulation_records),
        },
    }
    manifest_output_path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return ExportResult(
        exit_code=0,
        message="export completed; true simulation calibration evidence unavailable",
        replay_output_path=replay_output_path,
        simulation_output_path=simulation_output_path,
        manifest_output_path=manifest_output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export provenance-strong calibration reliability evidence from source report."
    )
    parser.add_argument(
        "--source-report",
        default=str(DEFAULT_SOURCE_REPORT),
        help="Approved source report path (default: config/p_calibration_report.json)",
    )
    parser.add_argument(
        "--source-calibration",
        default=str(DEFAULT_SOURCE_CALIBRATION),
        help="Optional read-only calibration metadata source path",
    )
    parser.add_argument(
        "--replay-output",
        default=str(DEFAULT_REPLAY_OUTPUT),
        help="Output path for replay evidence JSON array",
    )
    parser.add_argument(
        "--simulation-output",
        default=str(DEFAULT_SIMULATION_OUTPUT),
        help="Output path for simulation evidence JSON array",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEFAULT_MANIFEST_OUTPUT),
        help="Output path for provenance manifest JSON",
    )
    parser.add_argument("--baseline-ref", required=True, help="Baseline reference")
    parser.add_argument("--current-ref", required=True, help="Current reference")
    parser.add_argument("--run-id", required=True, help="Export run id ([a-z0-9_-]+)")
    parser.add_argument("--generated-at", default=None, help="Optional fixed generated_at timestamp")
    parser.add_argument("--simulation-seed", type=int, default=1337, help="Simulation seed metadata only when no true simulation evidence exists")
    args = parser.parse_args()

    source_report_path = Path(args.source_report)
    source_calibration_path = Path(args.source_calibration) if args.source_calibration else None
    replay_output_path = Path(args.replay_output)
    simulation_output_path = Path(args.simulation_output)
    manifest_output_path = Path(args.manifest_output)

    result = run_export(
        source_report_path=source_report_path,
        source_calibration_path=source_calibration_path,
        replay_output_path=replay_output_path,
        simulation_output_path=simulation_output_path,
        manifest_output_path=manifest_output_path,
        baseline_ref=args.baseline_ref,
        current_ref=args.current_ref,
        run_id=args.run_id,
        generated_at=args.generated_at,
        simulation_seed=int(args.simulation_seed),
    )
    payload = {
        "exit_code": result.exit_code,
        "message": result.message,
        "replay_output_path": str(result.replay_output_path) if result.replay_output_path else None,
        "simulation_output_path": str(result.simulation_output_path) if result.simulation_output_path else None,
        "manifest_output_path": str(result.manifest_output_path) if result.manifest_output_path else None,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())