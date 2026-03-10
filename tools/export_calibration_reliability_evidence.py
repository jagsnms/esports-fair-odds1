#!/usr/bin/env python3
"""
Export gate-ready calibration reliability evidence with provenance manifest.

This exporter is intentionally narrow:
- Reads only approved upstream replay calibration outputs.
- Emits bounded simulation evidence only from explicit canonical Phase 2 trace inputs.
- Emits one provenance manifest with source/output hashes and labeled-point exclusion counts.
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
DEFAULT_SIMULATION_BASELINE_TRACE_INPUT = (
    ROOT / "tools" / "fixtures" / "canonical_phase2_balanced_v1_trace_baseline_v1.json"
)
DEFAULT_SIMULATION_CURRENT_TRACE_INPUT = (
    ROOT / "tools" / "fixtures" / "canonical_phase2_balanced_v1_trace_current_v1.json"
)

GAME_TO_SURFACE = {
    "cs2": "replay",
}
BIN_PRESERVATION_POLICY = "preserve_full_source_bin_coverage"
NULL_EMPTY_BIN_POLICY = "retain_all_bins; for n==0 with null p_mean/y_rate, export 0.0 and preserve source_* fields"
SIMULATION_EXPORT_STATUS = "enabled_canonical_phase2_trace"
SIMULATION_EXPORT_REASON = "bounded simulation calibration evidence is derived from explicit canonical phase2 trace inputs"
SIMULATION_DATASET_ID = "canonical_phase2_balanced_v1_trace_v1"
SIMULATION_POLICY_PROFILE = "balanced_v1"
SIMULATION_TRACE_SOURCE_CONTRACT = "canonical_phase2_balanced_v1_trace"
SIMULATION_TRACE_SCHEMA_VERSION = "simulation_phase2_trace.v1"
SIMULATION_SEGMENT = "global"
SIMULATION_LABEL_SCOPE = "round_result"
SIMULATION_REQUIRED_SEED = 20260310
SIMULATION_REQUIRED_TICKS_PER_ROUND = 4
SIMULATION_BIN_COUNT = 10
LOG_LOSS_EPSILON = 1e-15


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


def _read_trace_export(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_json_object(path, label=label)
    trace = payload.get("trace_export") if isinstance(payload.get("trace_export"), dict) else payload
    if not isinstance(trace, dict):
        raise ValueError(f"{label} missing trace_export object: {path}")
    if trace.get("schema_version") != SIMULATION_TRACE_SCHEMA_VERSION:
        raise ValueError(f"{label} unexpected trace schema_version: {trace.get('schema_version')!r}")
    if trace.get("canonical_source_contract") != SIMULATION_TRACE_SOURCE_CONTRACT:
        raise ValueError(f"{label} unexpected canonical_source_contract: {trace.get('canonical_source_contract')!r}")
    if trace.get("policy_profile") != SIMULATION_POLICY_PROFILE:
        raise ValueError(f"{label} unexpected policy_profile: {trace.get('policy_profile')!r}")
    if trace.get("seed") != SIMULATION_REQUIRED_SEED:
        raise ValueError(f"{label} unexpected seed: {trace.get('seed')!r}")
    if trace.get("ticks_per_round") != SIMULATION_REQUIRED_TICKS_PER_ROUND:
        raise ValueError(f"{label} unexpected ticks_per_round: {trace.get('ticks_per_round')!r}")
    pairing_rule = trace.get("pairing_rule")
    if not isinstance(pairing_rule, dict) or pairing_rule.get("label_event_type") != SIMULATION_LABEL_SCOPE:
        raise ValueError(f"{label} unexpected pairing rule")
    if not isinstance(trace.get("trace_records"), list):
        raise ValueError(f"{label} missing trace_records array")
    return trace


def _build_trace_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(SIMULATION_BIN_COUNT)]
    for rec in records:
        p_hat = float(rec["p_hat"])
        label = 1 if bool(rec["round_winner_is_team_a"]) else 0
        bin_index = min(int(p_hat * SIMULATION_BIN_COUNT), SIMULATION_BIN_COUNT - 1)
        bins[bin_index].append((p_hat, label))
    out: list[dict[str, Any]] = []
    for idx, rows in enumerate(bins):
        if rows:
            count = len(rows)
            mean_prediction = sum(p for p, _ in rows) / count
            empirical_rate = sum(y for _, y in rows) / count
        else:
            count = 0
            mean_prediction = 0.0
            empirical_rate = 0.0
        out.append(
            {
                "bin_index": idx,
                "count": int(count),
                "mean_prediction": float(mean_prediction),
                "empirical_rate": float(empirical_rate),
            }
        )
    return out


def _build_simulation_record_from_trace(
    *,
    trace: dict[str, Any],
    evaluation_scope: str,
    source_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if evaluation_scope not in {"baseline", "current"}:
        raise ValueError(f"unsupported simulation evaluation_scope: {evaluation_scope}")
    records = trace.get("trace_records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"simulation trace {evaluation_scope} missing labeled trace_records")
    scored_rows: list[tuple[float, int]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"simulation trace {evaluation_scope} record {idx} must be object")
        if rec.get("label_scope") != SIMULATION_LABEL_SCOPE:
            raise ValueError(
                f"simulation trace {evaluation_scope} record {idx} unexpected label_scope: {rec.get('label_scope')!r}"
            )
        p_hat = rec.get("p_hat")
        label = rec.get("round_winner_is_team_a")
        if not isinstance(p_hat, (int, float)) or not math.isfinite(float(p_hat)):
            raise ValueError(f"simulation trace {evaluation_scope} record {idx} invalid p_hat")
        if not isinstance(label, bool):
            raise ValueError(f"simulation trace {evaluation_scope} record {idx} invalid round_winner_is_team_a")
        probability = min(max(float(p_hat), LOG_LOSS_EPSILON), 1.0 - LOG_LOSS_EPSILON)
        scored_rows.append((probability, 1 if label else 0))

    count = len(scored_rows)
    brier_score = sum((p - y) ** 2 for p, y in scored_rows) / count
    log_loss = -sum((y * math.log(p)) + ((1 - y) * math.log(1.0 - p)) for p, y in scored_rows) / count
    record = {
        "evidence_source": "simulation",
        "dataset_id": SIMULATION_DATASET_ID,
        "evaluation_scope": evaluation_scope,
        "seed": SIMULATION_REQUIRED_SEED,
        "segment": SIMULATION_SEGMENT,
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
        "reliability_curve_bins": _build_trace_bins(records),
    }
    provenance = {
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "schema_version": trace["schema_version"],
        "canonical_source_contract": trace["canonical_source_contract"],
        "policy_profile": trace["policy_profile"],
        "seed": int(trace["seed"]),
        "round_count": int(trace["round_count"]),
        "ticks_per_round": int(trace["ticks_per_round"]),
        "label_scope": SIMULATION_LABEL_SCOPE,
        "labeled_prediction_record_count": int(trace["labeled_prediction_record_count"]),
        "unlabeled_prediction_points_excluded": int(trace["unlabeled_prediction_points_excluded"]),
        "round_result_event_count": int(trace["round_result_event_count"]),
    }
    return record, provenance


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
    simulation_baseline_trace_path: Path | None = DEFAULT_SIMULATION_BASELINE_TRACE_INPUT,
    simulation_current_trace_path: Path | None = DEFAULT_SIMULATION_CURRENT_TRACE_INPUT,
) -> ExportResult:
    run_id_norm = str(run_id or "")
    if RUN_ID_PATTERN.fullmatch(run_id_norm) is None:
        return ExportResult(exit_code=1, message="invalid run_id: must match [a-z0-9_-]+")
    if not isinstance(baseline_ref, str) or not baseline_ref.strip():
        return ExportResult(exit_code=1, message="invalid baseline_ref: must be non-empty string")
    if not isinstance(current_ref, str) or not current_ref.strip():
        return ExportResult(exit_code=1, message="invalid current_ref: must be non-empty string")
    if int(simulation_seed) != SIMULATION_REQUIRED_SEED:
        return ExportResult(exit_code=1, message=f"invalid simulation_seed: must equal {SIMULATION_REQUIRED_SEED}")
    if (simulation_baseline_trace_path is None) != (simulation_current_trace_path is None):
        return ExportResult(
            exit_code=1,
            message="simulation trace inputs must provide both baseline and current paths or neither",
        )

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
        simulation_manifest_inputs: dict[str, Any] | None = None
        if simulation_baseline_trace_path is not None and simulation_current_trace_path is not None:
            baseline_trace = _read_trace_export(simulation_baseline_trace_path, label="simulation baseline trace")
            current_trace = _read_trace_export(simulation_current_trace_path, label="simulation current trace")
            baseline_record, baseline_meta = _build_simulation_record_from_trace(
                trace=baseline_trace,
                evaluation_scope="baseline",
                source_path=simulation_baseline_trace_path,
            )
            current_record, current_meta = _build_simulation_record_from_trace(
                trace=current_trace,
                evaluation_scope="current",
                source_path=simulation_current_trace_path,
            )
            simulation_records = [baseline_record, current_record]
            simulation_manifest_inputs = {
                "baseline_trace": baseline_meta,
                "current_trace": current_meta,
            }
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
            "simulation_source_contract": SIMULATION_TRACE_SOURCE_CONTRACT,
            "simulation_dataset_id": SIMULATION_DATASET_ID,
            "simulation_policy_profile": SIMULATION_POLICY_PROFILE,
            "simulation_label_scope": SIMULATION_LABEL_SCOPE,
            "simulation_bin_policy": "fixed_decile_bins_from_labeled_trace_records",
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
    if simulation_manifest_inputs is not None:
        manifest["simulation_trace_inputs"] = simulation_manifest_inputs
    manifest_output_path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return ExportResult(
        exit_code=0,
        message="export completed with bounded canonical simulation calibration evidence",
        replay_output_path=replay_output_path,
        simulation_output_path=simulation_output_path,
        manifest_output_path=manifest_output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export provenance-strong calibration reliability evidence from approved replay report plus canonical simulation traces."
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
    parser.add_argument(
        "--simulation-baseline-trace-input",
        default=str(DEFAULT_SIMULATION_BASELINE_TRACE_INPUT),
        help="Explicit baseline canonical Phase 2 trace input",
    )
    parser.add_argument(
        "--simulation-current-trace-input",
        default=str(DEFAULT_SIMULATION_CURRENT_TRACE_INPUT),
        help="Explicit current canonical Phase 2 trace input",
    )
    parser.add_argument("--baseline-ref", required=True, help="Baseline reference")
    parser.add_argument("--current-ref", required=True, help="Current reference")
    parser.add_argument("--run-id", required=True, help="Export run id ([a-z0-9_-]+)")
    parser.add_argument("--generated-at", default=None, help="Optional fixed generated_at timestamp")
    parser.add_argument(
        "--simulation-seed",
        type=int,
        default=SIMULATION_REQUIRED_SEED,
        help=f"Simulation seed metadata; must equal {SIMULATION_REQUIRED_SEED} for the bounded source",
    )
    args = parser.parse_args()

    source_report_path = Path(args.source_report)
    source_calibration_path = Path(args.source_calibration) if args.source_calibration else None
    replay_output_path = Path(args.replay_output)
    simulation_output_path = Path(args.simulation_output)
    manifest_output_path = Path(args.manifest_output)
    simulation_baseline_trace_path = (
        Path(args.simulation_baseline_trace_input) if args.simulation_baseline_trace_input else None
    )
    simulation_current_trace_path = (
        Path(args.simulation_current_trace_input) if args.simulation_current_trace_input else None
    )

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
        simulation_baseline_trace_path=simulation_baseline_trace_path,
        simulation_current_trace_path=simulation_current_trace_path,
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
