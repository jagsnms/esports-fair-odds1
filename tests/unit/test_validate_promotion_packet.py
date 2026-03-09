from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import tools.validate_promotion_packet as promotion_packet_validator
from tools.validate_promotion_packet import validate_promotion_packet


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "tools" / "validate_promotion_packet.py"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_branch_proof() -> dict:
    return {
        "source_branch": "agent-initiative-base",
        "local_source_head": "c28486cc0b4c015b5fd763079f23ca977f3aa1f8",
        "origin_source_head": "c28486cc0b4c015b5fd763079f23ca977f3aa1f8",
        "pre_initiative_base": "097132aa1e5bd621556fc78099bdcf45234d176c",
        "initiative_commit_range": "097132a..c28486c",
        "initiative_commits": ["c28486cc0b4c015b5fd763079f23ca977f3aa1f8"],
        "files_in_scope": [
            "tools/assemble_promotion_packet.py",
            "tests/unit/test_assemble_promotion_packet.py",
        ],
    }


def _valid_evidence_summary(gate_status: str = "pass") -> dict:
    return {
        "schema_version": "calibration_reliability_evidence.v1",
        "generated_at": "2026-03-12T00:00:00Z",
        "baseline_ref": "baseline:abc",
        "current_ref": "current:def",
        "gate_status": gate_status,
        "incomplete_reasons": [],
        "evidence_records": [],
        "comparison_pairs": [],
    }


def _build_packet(root: Path, run_id: str, gate_status: str = "pass") -> Path:
    packet_dir = root / run_id
    artifacts = packet_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    evidence = artifacts / "evidence_summary.json"
    checks = artifacts / "checks_output.txt"
    proof = artifacts / "branch_commit_proof.json"
    _write_json(evidence, _valid_evidence_summary(gate_status))
    checks.write_text("check output\n", encoding="utf-8")
    _write_json(proof, _valid_branch_proof())

    manifest = {
        "schema_version": "promotion_packet.v1",
        "run_id": run_id,
        "assembled_at": "2026-03-12T00:00:00Z",
        "gate_status": gate_status,
        "inputs": {
            "evidence_summary_path": "/tmp/evidence.json",
            "checks_output_path": "/tmp/checks.txt",
            "branch_commit_proof_path": "/tmp/proof.json",
        },
        "artifact_hashes": {
            "artifacts/evidence_summary.json": _sha(evidence),
            "artifacts/checks_output.txt": _sha(checks),
            "artifacts/branch_commit_proof.json": _sha(proof),
        },
    }
    _write_json(packet_dir / "packet_manifest.json", manifest)
    return packet_dir


def test_pass_case_validates_complete_packet(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_pass"
    _build_packet(packet_root, run_id)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 0
    assert result["status"] == "pass"
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["checks"]["artifact_hashes_match"] is True


def test_invalid_packet_when_evidence_summary_schema_nonconformant(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_bad_evidence_shape"
    packet_dir = _build_packet(packet_root, run_id)
    evidence_path = packet_dir / "artifacts" / "evidence_summary.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    del payload["baseline_ref"]
    _write_json(evidence_path, payload)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["artifacts/evidence_summary.json"] = _sha(evidence_path)
    _write_json(manifest_path, manifest)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 2
    assert result["status"] == "invalid_packet"
    assert any("evidence_summary schema nonconformant" in err for err in result["errors"])


def test_blocked_when_schema_validator_engine_unavailable(tmp_path: Path, monkeypatch) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_schema_engine_missing"
    _build_packet(packet_root, run_id)
    monkeypatch.setattr(promotion_packet_validator, "Draft202012Validator", None)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 1
    assert result["status"] == "blocked"
    assert any("schema validator unavailable" in err for err in result["errors"])


def test_blocked_when_evidence_summary_wrong_top_level_type(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_bad_evidence_top_level"
    packet_dir = _build_packet(packet_root, run_id)
    evidence_path = packet_dir / "artifacts" / "evidence_summary.json"
    evidence_path.write_text("[]", encoding="utf-8")
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["artifacts/evidence_summary.json"] = _sha(evidence_path)
    _write_json(manifest_path, manifest)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 1
    assert result["status"] == "blocked"
    assert any("evidence_summary must be a top-level JSON object" in err for err in result["errors"])


def test_blocked_case_when_packet_directory_missing(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"

    exit_code, result = validate_promotion_packet(
        run_id="packet_missing",
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 1
    assert result["status"] == "blocked"
    assert any("packet_dir missing or unreadable" in err for err in result["errors"])


def test_invalid_packet_due_to_missing_required_manifest_field(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_missing_field"
    packet_dir = _build_packet(packet_root, run_id)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["inputs"]["checks_output_path"]
    _write_json(manifest_path, manifest)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 2
    assert result["status"] == "invalid_packet"
    assert "missing required manifest field: inputs.checks_output_path" in result["errors"]


def test_invalid_packet_due_to_artifact_hash_mismatch(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_hash_mismatch"
    packet_dir = _build_packet(packet_root, run_id)
    (packet_dir / "artifacts" / "checks_output.txt").write_text("tampered", encoding="utf-8")

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 2
    assert result["status"] == "invalid_packet"
    assert "artifact hash mismatch: artifacts/checks_output.txt" in result["errors"]


def test_invalid_packet_due_to_disallowed_gate_status(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_bad_status"
    _build_packet(packet_root, run_id, gate_status="unknown_status")

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 2
    assert result["status"] == "invalid_packet"
    assert "manifest gate_status invalid: must be one of pass, incomplete_evidence, fail" in result["errors"]


def test_invalid_packet_when_manifest_and_evidence_gate_status_differ(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_mismatched_gate_status"
    packet_dir = _build_packet(packet_root, run_id, gate_status="pass")
    evidence_path = packet_dir / "artifacts" / "evidence_summary.json"
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_payload["gate_status"] = "fail"
    _write_json(evidence_path, evidence_payload)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["artifact_hashes"]["artifacts/evidence_summary.json"] = _sha(evidence_path)
    _write_json(manifest_path, manifest_payload)

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 2
    assert result["status"] == "invalid_packet"
    assert any("gate_status mismatch" in err for err in result["errors"])


def test_extra_files_are_warning_only_not_invalidating(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_extra_files"
    packet_dir = _build_packet(packet_root, run_id)
    (packet_dir / "extra.txt").write_text("extra", encoding="utf-8")
    extra_dir = packet_dir / "extra_dir"
    extra_dir.mkdir(parents=True, exist_ok=True)
    (extra_dir / "nested.txt").write_text("nested", encoding="utf-8")

    exit_code, result = validate_promotion_packet(
        run_id=run_id,
        validated_at="2026-03-15T00:00:00Z",
        packet_root=packet_root,
    )
    assert exit_code == 0
    assert result["status"] == "pass"
    assert result["warnings"] == [
        "extra path tolerated: extra.txt",
        "extra path tolerated: extra_dir",
        "extra path tolerated: extra_dir/nested.txt",
    ]


def test_deterministic_stdout_json_structure_and_order(tmp_path: Path) -> None:
    tmp = tmp_path
    packet_root = tmp / "packets"
    run_id = "packet_stdout"
    _build_packet(packet_root, run_id)

    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--run-id",
        run_id,
        "--validated-at",
        "2026-03-15T00:00:00Z",
        "--packet-root",
        str(packet_root),
    ]
    first = subprocess.run(cmd, check=False, capture_output=True, text=True)
    second = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    assert sorted(payload.keys()) == sorted(
        ["status", "packet_dir", "run_id", "checks", "errors", "warnings", "gate_status", "validated_at"]
    )
    assert payload["status"] == "pass"
