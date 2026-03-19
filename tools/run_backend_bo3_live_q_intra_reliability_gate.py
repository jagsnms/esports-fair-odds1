#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "backend_bo3_live_q_intra_reliability_gate_v1"
SOURCE_EVIDENCE_SCHEMA_VERSION = "backend_bo3_live_round_calibration_evidence_v1"
PREDICTION_TARGET = "q_intra_total"
LABEL_EVENT_TYPE = "round_result"
GATE_KIND = "evidence_sufficiency_only"
OUTPUT_PATH = Path("automation/reports/backend_bo3_live_q_intra_reliability_gate_v1.json")
LOG_LOSS_EPSILON = 1e-15
BIN_COUNT = 10
MIN_LABELED_RECORD_COUNT = 100
MIN_NON_EMPTY_BIN_COUNT = 3


@dataclass(frozen=True)
class RunnerResult:
    exit_code: int
    output_path: Path | None
    message: str
    artifact: dict[str, Any] | None = None


def _now_iso_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _zero_bins() -> list[dict[str, Any]]:
    return [
        {
            "bin_index": idx,
            "count": 0,
            "mean_prediction": 0.0,
            "empirical_rate": 0.0,
        }
        for idx in range(BIN_COUNT)
    ]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} invalid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be top-level JSON object: {path}")
    return data


def _validate_source_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SOURCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"source evidence schema_version must be {SOURCE_EVIDENCE_SCHEMA_VERSION}")
    if payload.get("prediction_target") != PREDICTION_TARGET:
        raise ValueError(f"source evidence prediction_target must be {PREDICTION_TARGET}")
    if payload.get("label_event_type") != LABEL_EVENT_TYPE:
        raise ValueError(f"source evidence label_event_type must be {LABEL_EVENT_TYPE}")

    records = payload.get("labeled_prediction_records")
    if not isinstance(records, list):
        raise ValueError("source evidence labeled_prediction_records must be an array")

    normalized: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"labeled_prediction_records[{idx}] must be an object")
        q_raw = rec.get("q_intra_total")
        if not isinstance(q_raw, (int, float)) or isinstance(q_raw, bool) or not math.isfinite(float(q_raw)):
            raise ValueError(f"labeled_prediction_records[{idx}].q_intra_total must be finite number")
        q_value = float(q_raw)
        if not 0.0 <= q_value <= 1.0:
            raise ValueError(f"labeled_prediction_records[{idx}].q_intra_total must be within [0,1]")
        label_raw = rec.get("round_winner_is_team_a")
        if not isinstance(label_raw, bool):
            raise ValueError(f"labeled_prediction_records[{idx}].round_winner_is_team_a must be boolean")
        normalized.append(
            {
                "q_intra_total": q_value,
                "round_winner_is_team_a": bool(label_raw),
            }
        )
    return normalized


def _build_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(BIN_COUNT)]
    for rec in records:
        q = float(rec["q_intra_total"])
        label = 1 if rec["round_winner_is_team_a"] else 0
        bin_index = min(int(q * BIN_COUNT), BIN_COUNT - 1)
        buckets[bin_index].append((q, label))

    out: list[dict[str, Any]] = []
    for idx, rows in enumerate(buckets):
        if rows:
            count = len(rows)
            mean_prediction = sum(q for q, _ in rows) / count
            empirical_rate = sum(label for _, label in rows) / count
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


def build_backend_bo3_live_q_intra_reliability_gate(
    *,
    source_evidence_payload: dict[str, Any],
    source_evidence_path: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    records = _validate_source_evidence(source_evidence_payload)
    labeled_record_count = len(records)
    reliability_curve_bins = _build_bins(records) if records else _zero_bins()
    non_empty_bin_count = sum(1 for row in reliability_curve_bins if int(row["count"]) > 0)
    empty_bin_count = BIN_COUNT - non_empty_bin_count

    insufficiency_reasons: list[str] = []
    if labeled_record_count == 0:
        insufficiency_reasons.append("no_labeled_records")
    if labeled_record_count < MIN_LABELED_RECORD_COUNT:
        insufficiency_reasons.append("labeled_record_count_below_threshold")
    if non_empty_bin_count < MIN_NON_EMPTY_BIN_COUNT:
        insufficiency_reasons.append("non_empty_bin_count_below_threshold")

    if records:
        probabilities = [float(rec["q_intra_total"]) for rec in records]
        labels = [1 if rec["round_winner_is_team_a"] else 0 for rec in records]
        brier_score = sum((q - y) ** 2 for q, y in zip(probabilities, labels)) / labeled_record_count
        log_loss = -sum(
            (y * math.log(min(max(q, LOG_LOSS_EPSILON), 1.0 - LOG_LOSS_EPSILON)))
            + ((1 - y) * math.log(1.0 - min(max(q, LOG_LOSS_EPSILON), 1.0 - LOG_LOSS_EPSILON)))
            for q, y in zip(probabilities, labels)
        ) / labeled_record_count
    else:
        brier_score = None
        log_loss = None

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso_z(),
        "source_evidence_path": str(source_evidence_path),
        "source_evidence_schema_version": str(source_evidence_payload["schema_version"]),
        "prediction_target": PREDICTION_TARGET,
        "label_event_type": LABEL_EVENT_TYPE,
        "gate_kind": GATE_KIND,
        "gate_status": "sufficient_evidence" if not insufficiency_reasons else "insufficient_evidence",
        "insufficiency_reasons": insufficiency_reasons,
        "sample_counts": {
            "labeled_record_count": labeled_record_count,
            "non_empty_bin_count": non_empty_bin_count,
            "empty_bin_count": empty_bin_count,
        },
        "metric_policy": {
            "includes_brier_score": True,
            "includes_log_loss": True,
            "includes_binned_reliability": True,
            "log_loss_clipping": {
                "enabled": True,
                "epsilon": LOG_LOSS_EPSILON,
            },
            "binning_policy": {
                "bin_count": BIN_COUNT,
                "binning_rule": "equal_width_[0,1]_with_bin_index_min(int(q_intra_total*10),9)",
            },
            "sufficiency_thresholds": {
                "min_labeled_record_count": MIN_LABELED_RECORD_COUNT,
                "min_non_empty_bin_count": MIN_NON_EMPTY_BIN_COUNT,
            },
        },
        "metrics": {
            "brier_score": None if brier_score is None else float(brier_score),
            "log_loss": None if log_loss is None else float(log_loss),
            "reliability_curve_bins": reliability_curve_bins,
        },
        "truth_boundary": {
            "what_this_is": "live_bo3_round_level_q_intra_calibration_measurement",
            "what_this_is_not": [
                "parameter_tuning",
                "calibration_quality_certification",
                "p_hat_calibration",
                "rails_calibration",
                "engine_math_change",
                "replay_canonical_matching",
                "bo3_upstream_coarse_progression_work",
            ],
        },
    }
    return artifact


def run_backend_bo3_live_q_intra_reliability_gate(
    *,
    source_evidence_input: str,
    output_path: str | None = None,
    generated_at: str | None = None,
) -> RunnerResult:
    source_path = Path(source_evidence_input)
    try:
        source_payload = _load_json_object(source_path, label="source evidence")
        artifact = build_backend_bo3_live_q_intra_reliability_gate(
            source_evidence_payload=source_payload,
            source_evidence_path=source_evidence_input,
            generated_at=generated_at,
        )
    except ValueError as exc:
        return RunnerResult(exit_code=1, output_path=None, message=str(exc), artifact=None)

    artifact_path = Path(output_path) if output_path is not None else OUTPUT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return RunnerResult(
        exit_code=0,
        output_path=artifact_path,
        message=f"wrote {artifact_path}",
        artifact=artifact,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BO3 live q_intra reliability sufficiency artifact from labeled live evidence."
    )
    parser.add_argument(
        "--source-evidence-input",
        default="automation/reports/backend_bo3_live_round_calibration_evidence_v1.json",
        help="Path to detailed BO3 live labeled q_intra evidence JSON artifact.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to write the reliability gate artifact.",
    )
    parser.add_argument("--generated-at", default=None, help="Optional fixed generated_at timestamp")
    args = parser.parse_args()

    result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=args.source_evidence_input,
        output_path=args.output,
        generated_at=args.generated_at,
    )
    payload = {
        "exit_code": result.exit_code,
        "message": result.message,
        "output_path": str(result.output_path) if result.output_path is not None else None,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
