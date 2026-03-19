#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.export_backend_bo3_live_round_calibration_evidence import (
        DEFAULT_CAPTURE_PATH,
        DEFAULT_EVIDENCE_OUTPUT,
        DEFAULT_HISTORY_PATH,
        DEFAULT_REPORT_OUTPUT,
        build_backend_bo3_live_round_calibration_evidence,
        write_json,
    )
    from tools.run_backend_bo3_live_q_intra_reliability_gate import (
        OUTPUT_PATH as DEFAULT_GATE_OUTPUT,
        run_backend_bo3_live_q_intra_reliability_gate,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from export_backend_bo3_live_round_calibration_evidence import (  # type: ignore[no-redef]
        DEFAULT_CAPTURE_PATH,
        DEFAULT_EVIDENCE_OUTPUT,
        DEFAULT_HISTORY_PATH,
        DEFAULT_REPORT_OUTPUT,
        build_backend_bo3_live_round_calibration_evidence,
        write_json,
    )
    from run_backend_bo3_live_q_intra_reliability_gate import (  # type: ignore[no-redef]
        OUTPUT_PATH as DEFAULT_GATE_OUTPUT,
        run_backend_bo3_live_q_intra_reliability_gate,
    )


@dataclass(frozen=True)
class MeasurementRunnerResult:
    exit_code: int
    message: str
    payload: dict[str, Any] | None = None


def run_backend_bo3_live_q_intra_measurement(
    *,
    capture_path: str,
    history_path: str,
    evidence_output: str | None = None,
    report_output: str | None = None,
    gate_output: str | None = None,
    generated_at: str | None = None,
) -> MeasurementRunnerResult:
    capture_path_obj = Path(capture_path)
    history_path_obj = Path(history_path)
    evidence_output_str = str(evidence_output) if evidence_output is not None else str(DEFAULT_EVIDENCE_OUTPUT)
    report_output_str = str(report_output) if report_output is not None else str(DEFAULT_REPORT_OUTPUT)
    gate_output_str = str(gate_output) if gate_output is not None else str(DEFAULT_GATE_OUTPUT)

    try:
        evidence_payload, report_payload = build_backend_bo3_live_round_calibration_evidence(
            capture_path_obj,
            history_path_obj,
        )
        write_json(Path(evidence_output_str), evidence_payload)
        write_json(Path(report_output_str), report_payload)
    except Exception as exc:
        return MeasurementRunnerResult(
            exit_code=1,
            message=f"exporter_failed: {exc}",
            payload=None,
        )

    gate_result = run_backend_bo3_live_q_intra_reliability_gate(
        source_evidence_input=evidence_output_str,
        output_path=gate_output_str,
        generated_at=generated_at,
    )
    if gate_result.exit_code != 0 or gate_result.artifact is None:
        return MeasurementRunnerResult(
            exit_code=gate_result.exit_code or 1,
            message=f"gate_failed: {gate_result.message}",
            payload=None,
        )

    payload = {
        "exporter_evidence_output_path": evidence_output_str,
        "exporter_report_output_path": report_output_str,
        "gate_output_path": gate_output_str,
        "gate_status": gate_result.artifact.get("gate_status"),
        "insufficiency_reasons": gate_result.artifact.get("insufficiency_reasons"),
        "labeled_record_count": ((gate_result.artifact.get("sample_counts") or {}).get("labeled_record_count")),
    }
    return MeasurementRunnerResult(
        exit_code=0,
        message="measurement completed",
        payload=payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the BO3 live q_intra measurement loop: exporter then reliability gate."
    )
    parser.add_argument("--capture-path", default=str(DEFAULT_CAPTURE_PATH), help="Path to BO3 live capture JSONL.")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Path to persisted history_points.jsonl.")
    parser.add_argument(
        "--evidence-output",
        default=str(DEFAULT_EVIDENCE_OUTPUT),
        help="Path to write detailed BO3 live q_intra evidence JSON.",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
        help="Path to write BO3 live q_intra evidence report JSON.",
    )
    parser.add_argument(
        "--gate-output",
        default=str(DEFAULT_GATE_OUTPUT),
        help="Path to write BO3 live q_intra reliability gate JSON.",
    )
    parser.add_argument("--generated-at", default=None, help="Optional fixed timestamp passed through to the gate.")
    args = parser.parse_args()

    result = run_backend_bo3_live_q_intra_measurement(
        capture_path=args.capture_path,
        history_path=args.history_path,
        evidence_output=args.evidence_output,
        report_output=args.report_output,
        gate_output=args.gate_output,
        generated_at=args.generated_at,
    )
    if result.exit_code != 0:
        print(result.message, file=sys.stderr)
        return result.exit_code
    assert result.payload is not None
    print(json.dumps(result.payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
