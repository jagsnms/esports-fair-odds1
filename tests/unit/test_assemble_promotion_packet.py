from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.assemble_promotion_packet import assemble_promotion_packet


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _branch_proof() -> dict:
    return {
        "source_branch": "agent-initiative-base",
        "local_source_head": "097132aa1e5bd621556fc78099bdcf45234d176c",
        "origin_source_head": "097132aa1e5bd621556fc78099bdcf45234d176c",
        "pre_initiative_base": "4066fe7df144e5a21b9bad91ad581ee97c2bc4e8",
        "initiative_commit_range": "4066fe7..097132a",
        "initiative_commits": ["097132aa1e5bd621556fc78099bdcf45234d176c"],
        "files_in_scope": [
            "tools/assemble_promotion_packet.py",
            "tests/unit/test_assemble_promotion_packet.py",
        ],
    }


def _evidence_summary(gate_status: str) -> dict:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pass_packet_assembly_creates_required_layout_and_byte_copies(tmp_path: Path) -> None:
    evidence_src = tmp_path / "evidence.json"
    checks_src = tmp_path / "checks.txt"
    proof_src = tmp_path / "proof.json"
    packet_root = tmp_path / "packets"
    _write_json(evidence_src, _evidence_summary("pass"))
    checks_src.write_bytes(b"check one: pass\ncheck two: pass\n")
    _write_json(proof_src, _branch_proof())

    result = assemble_promotion_packet(
        run_id="packet_001",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )

    assert result.exit_code == 0
    assert result.status == "assembled_pass_ready"
    assert result.packet_dir == packet_root / "packet_001"
    assert result.packet_dir is not None and result.packet_dir.exists()

    manifest_path = result.packet_dir / "packet_manifest.json"
    evidence_dst = result.packet_dir / "artifacts" / "evidence_summary.json"
    checks_dst = result.packet_dir / "artifacts" / "checks_output.txt"
    proof_dst = result.packet_dir / "artifacts" / "branch_commit_proof.json"
    assert manifest_path.exists()
    assert evidence_dst.exists()
    assert checks_dst.exists()
    assert proof_dst.exists()
    assert evidence_dst.read_bytes() == evidence_src.read_bytes()
    assert checks_dst.read_bytes() == checks_src.read_bytes()
    assert proof_dst.read_bytes() == proof_src.read_bytes()


def test_nonpass_evidence_still_assembles_packet_and_returns_nonzero(tmp_path: Path) -> None:
    evidence_src = tmp_path / "evidence.json"
    checks_src = tmp_path / "checks.txt"
    proof_src = tmp_path / "proof.json"
    packet_root = tmp_path / "packets"
    _write_json(evidence_src, _evidence_summary("incomplete_evidence"))
    checks_src.write_text("non-pass check output", encoding="utf-8")
    _write_json(proof_src, _branch_proof())

    result = assemble_promotion_packet(
        run_id="packet_002",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )

    assert result.exit_code != 0
    assert result.status == "assembled_with_nonpass_evidence"
    assert result.gate_status == "incomplete_evidence"
    assert result.packet_dir is not None and result.packet_dir.exists()


def test_blocked_cases_missing_invalid_or_incomplete_inputs_leave_no_packet(tmp_path: Path) -> None:
    evidence_src = tmp_path / "evidence.json"
    checks_src = tmp_path / "checks.txt"
    proof_src = tmp_path / "proof.json"
    packet_root = tmp_path / "packets"
    _write_json(evidence_src, _evidence_summary("pass"))
    checks_src.write_text("check output", encoding="utf-8")
    _write_json(proof_src, _branch_proof())

    missing_file_result = assemble_promotion_packet(
        run_id="packet_003",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(tmp_path / "missing.json"),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )
    assert missing_file_result.exit_code == 1
    assert missing_file_result.packet_dir is None
    assert not (packet_root / "packet_003").exists()

    evidence_src.write_text("{not-json", encoding="utf-8")
    invalid_json_result = assemble_promotion_packet(
        run_id="packet_004",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )
    assert invalid_json_result.exit_code == 1
    assert invalid_json_result.packet_dir is None
    assert not (packet_root / "packet_004").exists()

    _write_json(evidence_src, _evidence_summary("pass"))
    bad_proof = _branch_proof()
    del bad_proof["files_in_scope"]
    _write_json(proof_src, bad_proof)
    missing_field_result = assemble_promotion_packet(
        run_id="packet_005",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )
    assert missing_field_result.exit_code == 1
    assert missing_field_result.packet_dir is None
    assert not (packet_root / "packet_005").exists()


def test_invalid_run_id_is_rejected_without_packet_creation(tmp_path: Path) -> None:
    evidence_src = tmp_path / "evidence.json"
    checks_src = tmp_path / "checks.txt"
    proof_src = tmp_path / "proof.json"
    packet_root = tmp_path / "packets"
    _write_json(evidence_src, _evidence_summary("pass"))
    checks_src.write_text("check output", encoding="utf-8")
    _write_json(proof_src, _branch_proof())

    result = assemble_promotion_packet(
        run_id="Packet 006",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )

    assert result.exit_code == 1
    assert result.packet_dir is None
    assert not any(packet_root.glob("*"))


def test_deterministic_manifest_and_hashes_are_stable(tmp_path: Path) -> None:
    evidence_src = tmp_path / "evidence.json"
    checks_src = tmp_path / "checks.txt"
    proof_src = tmp_path / "proof.json"
    packet_root = tmp_path / "packets"
    _write_json(evidence_src, _evidence_summary("pass"))
    checks_src.write_bytes(b"check output fixed\n")
    _write_json(proof_src, _branch_proof())

    first = assemble_promotion_packet(
        run_id="packet_007",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )
    second = assemble_promotion_packet(
        run_id="packet_007",
        assembled_at="2026-03-12T00:00:00Z",
        evidence_summary_path=str(evidence_src),
        checks_output_path=str(checks_src),
        branch_commit_proof_path=str(proof_src),
        packet_root=packet_root,
    )
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.packet_dir == second.packet_dir
    assert first.packet_dir is not None

    manifest_path = first.packet_dir / "packet_manifest.json"
    manifest_first = manifest_path.read_text(encoding="utf-8")
    manifest_second = manifest_path.read_text(encoding="utf-8")
    assert manifest_first == manifest_second

    manifest_obj = json.loads(manifest_first)
    assert manifest_obj["assembled_at"] == "2026-03-12T00:00:00Z"
    assert manifest_obj["artifact_hashes"]["artifacts/evidence_summary.json"] == _sha256(
        first.packet_dir / "artifacts" / "evidence_summary.json"
    )
    assert manifest_obj["artifact_hashes"]["artifacts/checks_output.txt"] == _sha256(
        first.packet_dir / "artifacts" / "checks_output.txt"
    )
    assert manifest_obj["artifact_hashes"]["artifacts/branch_commit_proof.json"] == _sha256(
        first.packet_dir / "artifacts" / "branch_commit_proof.json"
    )
