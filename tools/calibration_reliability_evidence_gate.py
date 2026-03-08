#!/usr/bin/env python3
"""
Calibration & reliability evidence gate summary builder (Stage 1).

This module is evidence/reporting only. It does not change engine math,
replay architecture, or simulation architecture.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "calibration_reliability_evidence.v1"
MANDATORY_METRIC_KEYS = ("brier_score", "log_loss", "reliability_curve_bins")
MANDATORY_TOP_LEVEL_KEYS = (
    "schema_version",
    "generated_at",
    "baseline_ref",
    "current_ref",
    "evidence_records",
    "comparison_pairs",
    "gate_status",
    "incomplete_reasons",
)
ALLOWED_SOURCES = {"replay", "simulation"}
ALLOWED_SCOPES = {"baseline", "current"}


@dataclass(frozen=True)
class _AlignedKey:
    evidence_source: str
    dataset_id: str
    seed: int | None
    segment: str | None

    def sort_tuple(self) -> tuple[str, str, int, str]:
        seed_sort = -1 if self.seed is None else int(self.seed)
        seg_sort = "" if self.segment is None else str(self.segment)
        return (self.evidence_source, self.dataset_id, seed_sort, seg_sort)


def _now_iso_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _normalize_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer() and math.isfinite(float(value)):
        return int(value)
    return None


def _validate_bins_shape(bins: Any) -> bool:
    if not isinstance(bins, list):
        return False
    for row in bins:
        if not isinstance(row, dict):
            return False
        if "bin_index" not in row or "count" not in row or "mean_prediction" not in row or "empirical_rate" not in row:
            return False
        if not isinstance(row["bin_index"], int):
            return False
        if not isinstance(row["count"], int) or row["count"] < 0:
            return False
        if not _is_finite_number(row["mean_prediction"]):
            return False
        if not _is_finite_number(row["empirical_rate"]):
            return False
    return True


def _delta_bins(
    baseline_bins: list[dict[str, Any]],
    current_bins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    b_map = {int(row["bin_index"]): row for row in baseline_bins}
    c_map = {int(row["bin_index"]): row for row in current_bins}
    out: list[dict[str, Any]] = []
    for idx in sorted(set(b_map) | set(c_map)):
        b = b_map.get(idx, {"count": 0, "mean_prediction": 0.0, "empirical_rate": 0.0})
        c = c_map.get(idx, {"count": 0, "mean_prediction": 0.0, "empirical_rate": 0.0})
        out.append(
            {
                "bin_index": int(idx),
                "count_delta": int(c["count"]) - int(b["count"]),
                "mean_prediction_delta": float(c["mean_prediction"]) - float(b["mean_prediction"]),
                "empirical_rate_delta": float(c["empirical_rate"]) - float(b["empirical_rate"]),
            }
        )
    return out


def build_calibration_reliability_summary(
    *,
    baseline_ref: str,
    current_ref: str,
    evidence_records: list[dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    incomplete_reasons: list[str] = []
    fail_reasons: list[str] = []
    normalized_records: list[dict[str, Any]] = []

    if not isinstance(baseline_ref, str) or not baseline_ref.strip():
        fail_reasons.append("baseline_ref must be explicit non-empty string")
    if not isinstance(current_ref, str) or not current_ref.strip():
        fail_reasons.append("current_ref must be explicit non-empty string")
    if not isinstance(evidence_records, list):
        fail_reasons.append("evidence_records must be an array")
        evidence_records = []

    for idx, rec in enumerate(evidence_records):
        if not isinstance(rec, dict):
            fail_reasons.append(f"evidence_records[{idx}] must be object")
            continue
        src = rec.get("evidence_source")
        if src not in ALLOWED_SOURCES:
            fail_reasons.append(f"evidence_records[{idx}].evidence_source invalid enum")
        scope = rec.get("evaluation_scope")
        if scope not in ALLOWED_SCOPES:
            fail_reasons.append(f"evidence_records[{idx}].evaluation_scope invalid enum")
        dataset_id = rec.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            fail_reasons.append(f"evidence_records[{idx}].dataset_id must be non-empty string")

        seed = _normalize_seed(rec.get("seed"))
        if src == "simulation" and seed is None:
            incomplete_reasons.append(f"simulation record missing non-null seed at evidence_records[{idx}]")
        segment_val = rec.get("segment")
        if segment_val is not None and not isinstance(segment_val, str):
            fail_reasons.append(f"evidence_records[{idx}].segment must be string|null")
        segment = segment_val if isinstance(segment_val, str) else None

        metrics_complete = True
        for key in MANDATORY_METRIC_KEYS:
            if key not in rec:
                metrics_complete = False
                incomplete_reasons.append(f"missing metric key {key} at evidence_records[{idx}]")
        if "brier_score" in rec and not _is_finite_number(rec["brier_score"]):
            fail_reasons.append(f"evidence_records[{idx}].brier_score invalid type/value")
            metrics_complete = False
        if "log_loss" in rec and not _is_finite_number(rec["log_loss"]):
            fail_reasons.append(f"evidence_records[{idx}].log_loss invalid type/value")
            metrics_complete = False
        if "reliability_curve_bins" in rec and not _validate_bins_shape(rec["reliability_curve_bins"]):
            fail_reasons.append(f"evidence_records[{idx}].reliability_curve_bins invalid type/value")
            metrics_complete = False

        normalized_records.append(
            {
                "evidence_source": src,
                "dataset_id": dataset_id,
                "evaluation_scope": scope,
                "seed": seed,
                "segment": segment,
                "brier_score": rec.get("brier_score"),
                "log_loss": rec.get("log_loss"),
                "reliability_curve_bins": rec.get("reliability_curve_bins"),
                "_metrics_complete": bool(metrics_complete),
            }
        )

    by_source_scope: dict[tuple[str, str], int] = {}
    for rec in normalized_records:
        if not rec["_metrics_complete"]:
            continue
        src = rec.get("evidence_source")
        scope = rec.get("evaluation_scope")
        if src in ALLOWED_SOURCES and scope in ALLOWED_SCOPES:
            key = (str(src), str(scope))
            by_source_scope[key] = by_source_scope.get(key, 0) + 1
    for src in ("replay", "simulation"):
        for scope in ("baseline", "current"):
            if by_source_scope.get((src, scope), 0) <= 0:
                incomplete_reasons.append(f"required slice missing: source={src}, scope={scope}")

    pair_table: dict[_AlignedKey, dict[str, dict[str, Any]]] = {}
    for rec in sorted(
        normalized_records,
        key=lambda r: (
            str(r.get("evidence_source")),
            str(r.get("dataset_id")),
            -1 if r.get("seed") is None else int(r["seed"]),
            "" if r.get("segment") is None else str(r.get("segment")),
            str(r.get("evaluation_scope")),
        ),
    ):
        if not rec["_metrics_complete"]:
            continue
        key = _AlignedKey(
            evidence_source=str(rec["evidence_source"]),
            dataset_id=str(rec["dataset_id"]),
            seed=rec["seed"],
            segment=rec["segment"],
        )
        scope = str(rec["evaluation_scope"])
        pair_table.setdefault(key, {})
        if scope in pair_table[key]:
            incomplete_reasons.append(f"duplicate complete record for pair key={key.sort_tuple()} scope={scope}")
            continue
        pair_table[key][scope] = rec

    comparison_pairs: list[dict[str, Any]] = []
    for key in sorted(pair_table, key=lambda k: k.sort_tuple()):
        scopes = pair_table[key]
        if "baseline" not in scopes or "current" not in scopes:
            incomplete_reasons.append(f"unpairable record key={key.sort_tuple()}")
            continue
        baseline_metrics = {
            "brier_score": float(scopes["baseline"]["brier_score"]),
            "log_loss": float(scopes["baseline"]["log_loss"]),
            "reliability_curve_bins": scopes["baseline"]["reliability_curve_bins"],
        }
        current_metrics = {
            "brier_score": float(scopes["current"]["brier_score"]),
            "log_loss": float(scopes["current"]["log_loss"]),
            "reliability_curve_bins": scopes["current"]["reliability_curve_bins"],
        }
        comparison_pairs.append(
            {
                "evidence_source": key.evidence_source,
                "dataset_id": key.dataset_id,
                "seed": key.seed,
                "segment": key.segment,
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "delta": {
                    "brier_score_delta": float(current_metrics["brier_score"]) - float(baseline_metrics["brier_score"]),
                    "log_loss_delta": float(current_metrics["log_loss"]) - float(baseline_metrics["log_loss"]),
                    "reliability_curve_bins_delta": _delta_bins(
                        baseline_metrics["reliability_curve_bins"],
                        current_metrics["reliability_curve_bins"],
                    ),
                },
            }
        )

    if fail_reasons:
        gate_status = "fail"
        incomplete_all = fail_reasons + incomplete_reasons
    elif incomplete_reasons:
        gate_status = "incomplete_evidence"
        incomplete_all = incomplete_reasons
    else:
        gate_status = "pass"
        incomplete_all = []

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso_z(),
        "baseline_ref": str(baseline_ref),
        "current_ref": str(current_ref),
        "evidence_records": [
            {k: v for k, v in rec.items() if k != "_metrics_complete"}
            for rec in sorted(
                normalized_records,
                key=lambda r: (
                    str(r.get("evidence_source")),
                    str(r.get("dataset_id")),
                    -1 if r.get("seed") is None else int(r["seed"]),
                    "" if r.get("segment") is None else str(r.get("segment")),
                    str(r.get("evaluation_scope")),
                ),
            )
        ],
        "comparison_pairs": comparison_pairs,
        "gate_status": gate_status,
        "incomplete_reasons": incomplete_all,
    }

    for key in MANDATORY_TOP_LEVEL_KEYS:
        if key not in summary:
            # Defensive guarantee of contract shape.
            summary[key] = None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration/reliability evidence gate summary JSON.")
    parser.add_argument("--baseline-ref", required=True, help="Explicit baseline reference identifier")
    parser.add_argument("--current-ref", required=True, help="Explicit current reference identifier")
    parser.add_argument(
        "--records-json",
        required=True,
        help="Path to JSON file containing evidence_records array",
    )
    args = parser.parse_args()

    records_path = Path(args.records_json)
    data = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--records-json must contain a top-level array")

    summary = build_calibration_reliability_summary(
        baseline_ref=args.baseline_ref,
        current_ref=args.current_ref,
        evidence_records=data,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
