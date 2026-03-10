#!/usr/bin/env python3
"""
Stage 1 orchestration runner for calibration/reliability evidence gate summaries.

Consumes precomputed replay + simulation evidence-record JSON arrays and emits
one machine-readable artifact under automation/reports.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.calibration_reliability_evidence_gate import (
        build_calibration_reliability_summary,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from calibration_reliability_evidence_gate import (  # type: ignore[no-redef]
        build_calibration_reliability_summary,
    )

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = ROOT / "automation" / "reports"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
OUTPUT_PREFIX = "calibration_reliability_evidence_gate_"


@dataclass(frozen=True)
class RunnerResult:
    exit_code: int
    output_path: Path | None
    gate_status: str | None
    message: str
    summary: dict[str, Any] | None = None


def _load_records_array(path: Path, label: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.exists():
        return None, f"{label} input file missing: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{label} input JSON invalid: {path} ({exc})"
    if not isinstance(data, list):
        return None, f"{label} input must be top-level JSON array: {path}"
    out: list[dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            out.append(row)
        else:
            # Keep process-level type contract strict at top-level only.
            out.append({"_invalid_row": row})
    return out, None


def run_evidence_gate_runner(
    *,
    replay_input_path: str,
    simulation_input_path: str,
    baseline_ref: str,
    current_ref: str,
    run_id: str,
    generated_at: str | None = None,
    reports_dir: Path | None = None,
) -> RunnerResult:
    run_id_norm = str(run_id or "")
    if not run_id_norm or RUN_ID_PATTERN.fullmatch(run_id_norm) is None:
        return RunnerResult(
            exit_code=1,
            output_path=None,
            gate_status=None,
            message="invalid run_id: must match [a-z0-9_-]+ and be non-empty",
        )
    if not isinstance(baseline_ref, str) or not baseline_ref.strip():
        return RunnerResult(
            exit_code=1,
            output_path=None,
            gate_status=None,
            message="invalid baseline_ref: must be explicit non-empty string",
        )
    if not isinstance(current_ref, str) or not current_ref.strip():
        return RunnerResult(
            exit_code=1,
            output_path=None,
            gate_status=None,
            message="invalid current_ref: must be explicit non-empty string",
        )

    replay_records, replay_err = _load_records_array(Path(replay_input_path), "replay")
    if replay_err is not None:
        return RunnerResult(exit_code=1, output_path=None, gate_status=None, message=replay_err)
    sim_records, sim_err = _load_records_array(Path(simulation_input_path), "simulation")
    if sim_err is not None:
        return RunnerResult(exit_code=1, output_path=None, gate_status=None, message=sim_err)
    assert replay_records is not None
    assert sim_records is not None

    combined_records = [*replay_records, *sim_records]
    summary = build_calibration_reliability_summary(
        baseline_ref=baseline_ref,
        current_ref=current_ref,
        evidence_records=combined_records,
        generated_at=generated_at,
    )

    out_dir = reports_dir or DEFAULT_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{OUTPUT_PREFIX}{run_id_norm}.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    gate_status = str(summary.get("gate_status"))
    if gate_status == "pass":
        return RunnerResult(
            exit_code=0,
            output_path=output_path,
            gate_status=gate_status,
            message=f"evidence gate pass: wrote {output_path}",
            summary=summary,
        )
    return RunnerResult(
        exit_code=2,
        output_path=output_path,
        gate_status=gate_status,
        message=f"evidence gate non-pass ({gate_status}): wrote {output_path}",
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run calibration/reliability evidence-gate orchestration from replay+simulation records."
    )
    parser.add_argument("--replay-input", required=True, help="Path to replay evidence-record JSON array")
    parser.add_argument("--simulation-input", required=True, help="Path to simulation evidence-record JSON array")
    parser.add_argument("--baseline-ref", required=True, help="Explicit baseline reference")
    parser.add_argument("--current-ref", required=True, help="Explicit current reference")
    parser.add_argument("--run-id", required=True, help="Run id ([a-z0-9_-]+)")
    parser.add_argument("--generated-at", default=None, help="Optional fixed generated_at timestamp")
    args = parser.parse_args()

    result = run_evidence_gate_runner(
        replay_input_path=args.replay_input,
        simulation_input_path=args.simulation_input,
        baseline_ref=args.baseline_ref,
        current_ref=args.current_ref,
        run_id=args.run_id,
        generated_at=args.generated_at,
    )
    payload = {
        "exit_code": result.exit_code,
        "gate_status": result.gate_status,
        "message": result.message,
        "output_path": str(result.output_path) if result.output_path else None,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
