#!/usr/bin/env python3
"""
Deterministic promotion-packet assembler for initiative-lane review artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
REQUIRED_BRANCH_PROOF_FIELDS = (
    "source_branch",
    "local_source_head",
    "origin_source_head",
    "pre_initiative_base",
    "initiative_commit_range",
    "initiative_commits",
    "files_in_scope",
)
PASS_GATE_STATUS = "pass"
NON_PASS_GATE_STATUSES = {"incomplete_evidence", "fail"}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKET_ROOT = ROOT / "automation" / "reports" / "promotion_packets"


@dataclass(frozen=True)
class PacketAssemblyResult:
    exit_code: int
    status: str
    message: str
    packet_dir: Path | None
    gate_status: str | None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_required_file(path: Path, label: str) -> tuple[bytes | None, str | None]:
    if not path.exists():
        return None, f"{label} missing: {path}"
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, f"{label} unreadable: {path} ({exc})"


def _load_required_json(path: Path, label: str) -> tuple[dict[str, Any] | list[Any] | None, bytes | None, str | None]:
    payload, err = _read_required_file(path, label)
    if err is not None:
        return None, None, err
    assert payload is not None
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, f"{label} invalid JSON: {path} ({exc})"
    return parsed, payload, None


def _validate_run_id(run_id: str) -> str | None:
    normalized = str(run_id or "")
    if not normalized:
        return "invalid run_id: must match [a-z0-9_-]+ and be non-empty"
    if RUN_ID_PATTERN.fullmatch(normalized) is None:
        return "invalid run_id: must match [a-z0-9_-]+ and be non-empty"
    return None


def _validate_branch_proof(proof: dict[str, Any]) -> str | None:
    for key in REQUIRED_BRANCH_PROOF_FIELDS:
        if key not in proof:
            return f"branch_commit_proof missing required field: {key}"
    return None


def assemble_promotion_packet(
    *,
    run_id: str,
    assembled_at: str,
    evidence_summary_path: str,
    checks_output_path: str,
    branch_commit_proof_path: str,
    packet_root: Path | None = None,
) -> PacketAssemblyResult:
    run_err = _validate_run_id(run_id)
    if run_err is not None:
        return PacketAssemblyResult(exit_code=1, status="blocked", message=run_err, packet_dir=None, gate_status=None)
    if not isinstance(assembled_at, str) or not assembled_at.strip():
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message="invalid assembled_at: required explicit non-empty string",
            packet_dir=None,
            gate_status=None,
        )

    evidence_json, evidence_bytes, evidence_err = _load_required_json(
        Path(evidence_summary_path), "evidence_summary_path"
    )
    if evidence_err is not None:
        return PacketAssemblyResult(exit_code=1, status="blocked", message=evidence_err, packet_dir=None, gate_status=None)
    assert evidence_json is not None
    assert evidence_bytes is not None
    if not isinstance(evidence_json, dict):
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message="evidence_summary_path must contain a top-level JSON object",
            packet_dir=None,
            gate_status=None,
        )
    gate_status = evidence_json.get("gate_status")
    if gate_status not in ({PASS_GATE_STATUS} | NON_PASS_GATE_STATUSES):
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message="evidence_summary.gate_status must be one of: pass, incomplete_evidence, fail",
            packet_dir=None,
            gate_status=None,
        )

    checks_bytes, checks_err = _read_required_file(Path(checks_output_path), "checks_output_path")
    if checks_err is not None:
        return PacketAssemblyResult(exit_code=1, status="blocked", message=checks_err, packet_dir=None, gate_status=None)
    assert checks_bytes is not None

    branch_proof_json, branch_proof_bytes, branch_err = _load_required_json(
        Path(branch_commit_proof_path), "branch_commit_proof_path"
    )
    if branch_err is not None:
        return PacketAssemblyResult(exit_code=1, status="blocked", message=branch_err, packet_dir=None, gate_status=None)
    if not isinstance(branch_proof_json, dict):
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message="branch_commit_proof_path must contain a top-level JSON object",
            packet_dir=None,
            gate_status=None,
        )
    missing_field_err = _validate_branch_proof(branch_proof_json)
    if missing_field_err is not None:
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message=missing_field_err,
            packet_dir=None,
            gate_status=None,
        )
    assert branch_proof_bytes is not None

    packet_root_dir = packet_root or DEFAULT_PACKET_ROOT
    packet_dir = packet_root_dir / run_id
    temp_dir: Path | None = None

    try:
        packet_root_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp.", dir=str(packet_root_dir)))
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        evidence_dst = artifacts_dir / "evidence_summary.json"
        checks_dst = artifacts_dir / "checks_output.txt"
        branch_dst = artifacts_dir / "branch_commit_proof.json"
        evidence_dst.write_bytes(evidence_bytes)
        checks_dst.write_bytes(checks_bytes)
        branch_dst.write_bytes(branch_proof_bytes)

        manifest = {
            "schema_version": "promotion_packet.v1",
            "run_id": run_id,
            "assembled_at": assembled_at,
            "gate_status": gate_status,
            "inputs": {
                "evidence_summary_path": str(Path(evidence_summary_path)),
                "checks_output_path": str(Path(checks_output_path)),
                "branch_commit_proof_path": str(Path(branch_commit_proof_path)),
            },
            "artifact_hashes": {
                "artifacts/evidence_summary.json": _sha256_bytes(evidence_bytes),
                "artifacts/checks_output.txt": _sha256_bytes(checks_bytes),
                "artifacts/branch_commit_proof.json": _sha256_bytes(branch_proof_bytes),
            },
        }
        (temp_dir / "packet_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if packet_dir.exists():
            shutil.rmtree(packet_dir)
        temp_dir.rename(packet_dir)
        temp_dir = None

        if gate_status == PASS_GATE_STATUS:
            return PacketAssemblyResult(
                exit_code=0,
                status="assembled_pass_ready",
                message=f"packet assembled: {packet_dir}",
                packet_dir=packet_dir,
                gate_status=gate_status,
            )
        return PacketAssemblyResult(
            exit_code=2,
            status="assembled_with_nonpass_evidence",
            message=f"packet assembled with non-pass evidence ({gate_status}): {packet_dir}",
            packet_dir=packet_dir,
            gate_status=gate_status,
        )
    except OSError as exc:
        return PacketAssemblyResult(
            exit_code=1,
            status="blocked",
            message=f"packet assembly failed: {exc}",
            packet_dir=None,
            gate_status=None,
        )
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble deterministic initiative-lane promotion packet bundle.")
    parser.add_argument("--run-id", required=True, help="Packet run id ([a-z0-9_-]+)")
    parser.add_argument("--assembled-at", required=True, help="Explicit assembly timestamp (UTC string)")
    parser.add_argument("--evidence-summary-path", required=True, help="Path to existing evidence summary JSON")
    parser.add_argument("--checks-output-path", required=True, help="Path to existing checks output text file")
    parser.add_argument("--branch-commit-proof-path", required=True, help="Path to existing branch/commit proof JSON")
    args = parser.parse_args()

    result = assemble_promotion_packet(
        run_id=args.run_id,
        assembled_at=args.assembled_at,
        evidence_summary_path=args.evidence_summary_path,
        checks_output_path=args.checks_output_path,
        branch_commit_proof_path=args.branch_commit_proof_path,
    )
    print(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "status": result.status,
                "message": result.message,
                "packet_dir": str(result.packet_dir) if result.packet_dir else None,
                "gate_status": result.gate_status,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
