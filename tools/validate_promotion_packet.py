#!/usr/bin/env python3
"""
Validate initiative-lane promotion packet completeness and integrity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - exercised in runtime environments without jsonschema
    Draft202012Validator = None

RUN_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
ALLOWED_GATE_STATUSES = {"pass", "incomplete_evidence", "fail"}
REQUIRED_PACKET_FILES = (
    "packet_manifest.json",
    "artifacts/evidence_summary.json",
    "artifacts/checks_output.txt",
    "artifacts/branch_commit_proof.json",
)
REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "run_id",
    "assembled_at",
    "gate_status",
)
REQUIRED_MANIFEST_INPUT_FIELDS = (
    "evidence_summary_path",
    "checks_output_path",
    "branch_commit_proof_path",
)
REQUIRED_MANIFEST_HASH_FIELDS = (
    "artifacts/evidence_summary.json",
    "artifacts/checks_output.txt",
    "artifacts/branch_commit_proof.json",
)
REQUIRED_BRANCH_PROOF_FIELDS = (
    "source_branch",
    "local_source_head",
    "origin_source_head",
    "pre_initiative_base",
    "initiative_commit_range",
    "initiative_commits",
    "files_in_scope",
)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKET_ROOT = ROOT / "automation" / "reports" / "promotion_packets"
CALIBRATION_EVIDENCE_SCHEMA_PATH = ROOT / "tools" / "schemas" / "calibration_reliability_evidence.schema.json"

CHECK_KEYS = (
    "packet_dir_exists",
    "required_files_present",
    "manifest_json_readable",
    "evidence_json_readable",
    "branch_proof_json_readable",
    "manifest_fields_complete",
    "branch_proof_fields_complete",
    "branch_proof_sanity",
    "gate_status_allowed",
    "artifact_hashes_match",
)


def _empty_checks() -> dict[str, bool]:
    return {k: False for k in CHECK_KEYS}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_json_file(path: Path, label: str) -> tuple[Any | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"{label} unreadable: {path} ({exc})"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{label} invalid JSON: {path} ({exc})"


def _path_to_str(path_parts: list[Any]) -> str:
    path = "$"
    for part in path_parts:
        if isinstance(part, int):
            path += f"[{part}]"
            continue
        if isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
            continue
        path += f"[{json.dumps(part, ensure_ascii=True)}]"
    return path


def validate_promotion_packet(
    *,
    run_id: str | None,
    validated_at: str | None,
    packet_root: Path | None = None,
    cli_unknown_args: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    checks = _empty_checks()
    errors: list[str] = []
    warnings: list[str] = []
    unknown_args = cli_unknown_args or []
    packet_base = packet_root or DEFAULT_PACKET_ROOT
    packet_dir = packet_base / (run_id or "")
    gate_status: str | None = None

    if unknown_args:
        errors.append(f"invalid CLI args: {' '.join(unknown_args)}")
    if not run_id or RUN_ID_PATTERN.fullmatch(run_id) is None:
        errors.append("invalid run_id: must match [a-z0-9_-]+ and be non-empty")
    if not isinstance(validated_at, str) or not validated_at.strip():
        errors.append("invalid validated_at: required explicit non-empty string")
    if errors:
        result = {
            "status": "blocked",
            "packet_dir": str(packet_dir),
            "run_id": run_id,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "gate_status": gate_status,
            "validated_at": validated_at,
        }
        return 1, result

    if not packet_dir.exists() or not packet_dir.is_dir():
        errors.append(f"packet_dir missing or unreadable: {packet_dir}")
        result = {
            "status": "blocked",
            "packet_dir": str(packet_dir),
            "run_id": run_id,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "gate_status": gate_status,
            "validated_at": validated_at,
        }
        return 1, result
    checks["packet_dir_exists"] = True

    required_paths = {rel: packet_dir / rel for rel in REQUIRED_PACKET_FILES}
    missing_required = [rel for rel in REQUIRED_PACKET_FILES if not required_paths[rel].exists()]
    if missing_required:
        for rel in missing_required:
            errors.append(f"missing required packet file: {rel}")
    else:
        checks["required_files_present"] = True

    # Extra paths are tolerated with warnings.
    allowed_dirs = {"artifacts"}
    allowed_files = set(REQUIRED_PACKET_FILES)
    extra_paths: list[str] = []
    for path in sorted(packet_dir.rglob("*"), key=lambda p: p.relative_to(packet_dir).as_posix()):
        rel = path.relative_to(packet_dir).as_posix()
        if path.is_dir():
            if rel not in allowed_dirs:
                extra_paths.append(rel)
        elif rel not in allowed_files:
            extra_paths.append(rel)
    for rel in extra_paths:
        warnings.append(f"extra path tolerated: {rel}")

    manifest_obj: dict[str, Any] | None = None
    evidence_obj: Any | None = None
    proof_obj: dict[str, Any] | None = None
    evidence_gate_status: str | None = None

    # Unreadable/invalid required JSON files are blocked.
    if required_paths["packet_manifest.json"].exists():
        parsed, err = _parse_json_file(required_paths["packet_manifest.json"], "packet_manifest")
        if err is not None:
            errors.append(err)
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        if isinstance(parsed, dict):
            manifest_obj = parsed
            checks["manifest_json_readable"] = True
        else:
            errors.append("packet_manifest must be a top-level JSON object")

    if required_paths["artifacts/evidence_summary.json"].exists():
        parsed, err = _parse_json_file(required_paths["artifacts/evidence_summary.json"], "evidence_summary")
        if err is not None:
            errors.append(err)
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        if not isinstance(parsed, dict):
            errors.append("evidence_summary must be a top-level JSON object")
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        evidence_obj = parsed
        checks["evidence_json_readable"] = True
        evidence_gate_status_value = evidence_obj.get("gate_status")
        if isinstance(evidence_gate_status_value, str):
            evidence_gate_status = evidence_gate_status_value

        if Draft202012Validator is None:
            errors.append("evidence_summary schema validator unavailable: jsonschema import failed")
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        schema_obj, schema_err = _parse_json_file(
            CALIBRATION_EVIDENCE_SCHEMA_PATH,
            "calibration_evidence_schema",
        )
        if schema_err is not None:
            errors.append(schema_err)
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        if not isinstance(schema_obj, dict):
            errors.append("calibration_evidence_schema must be a top-level JSON object")
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        try:
            validator = Draft202012Validator(schema_obj)
        except Exception as exc:
            errors.append(f"calibration_evidence_schema unusable validator setup: {exc}")
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        schema_violations = sorted(
            list(validator.iter_errors(evidence_obj)),
            key=lambda err: (
                tuple(str(part) for part in err.absolute_path),
                err.message,
                err.validator,
            ),
        )
        if schema_violations:
            errors.extend(
                [
                    f"evidence_summary schema nonconformant: {_path_to_str(list(err.absolute_path))}: {err.message}"
                    for err in schema_violations
                ]
            )

    if required_paths["artifacts/branch_commit_proof.json"].exists():
        parsed, err = _parse_json_file(required_paths["artifacts/branch_commit_proof.json"], "branch_commit_proof")
        if err is not None:
            errors.append(err)
            return 1, {
                "status": "blocked",
                "packet_dir": str(packet_dir),
                "run_id": run_id,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
                "gate_status": gate_status,
                "validated_at": validated_at,
            }
        if isinstance(parsed, dict):
            proof_obj = parsed
            checks["branch_proof_json_readable"] = True
        else:
            errors.append("branch_commit_proof must be a top-level JSON object")

    if manifest_obj is not None:
        missing_manifest = [k for k in REQUIRED_MANIFEST_FIELDS if k not in manifest_obj]
        inputs = manifest_obj.get("inputs")
        if not isinstance(inputs, dict):
            missing_manifest.extend(f"inputs.{k}" for k in REQUIRED_MANIFEST_INPUT_FIELDS)
        else:
            for key in REQUIRED_MANIFEST_INPUT_FIELDS:
                if key not in inputs:
                    missing_manifest.append(f"inputs.{key}")

        artifact_hashes = manifest_obj.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            missing_manifest.extend(f"artifact_hashes.{k}" for k in REQUIRED_MANIFEST_HASH_FIELDS)
        else:
            for key in REQUIRED_MANIFEST_HASH_FIELDS:
                if key not in artifact_hashes:
                    missing_manifest.append(f"artifact_hashes.{key}")

        if missing_manifest:
            for key in missing_manifest:
                errors.append(f"missing required manifest field: {key}")
        else:
            checks["manifest_fields_complete"] = True
            manifest_run_id_value = manifest_obj.get("run_id")
            if isinstance(manifest_run_id_value, str) and manifest_run_id_value != run_id:
                errors.append("run_id mismatch: packet_manifest.run_id and validator run_id must match exactly")
            gate_value = manifest_obj.get("gate_status")
            gate_status = gate_value if isinstance(gate_value, str) else None
            if gate_status in ALLOWED_GATE_STATUSES:
                checks["gate_status_allowed"] = True
            else:
                errors.append("manifest gate_status invalid: must be one of pass, incomplete_evidence, fail")

    if isinstance(gate_status, str) and isinstance(evidence_gate_status, str):
        if gate_status != evidence_gate_status:
            errors.append(
                "gate_status mismatch: packet_manifest.gate_status and evidence_summary.gate_status must match exactly"
            )

    if proof_obj is not None:
        missing_proof = [k for k in REQUIRED_BRANCH_PROOF_FIELDS if k not in proof_obj]
        if missing_proof:
            for key in missing_proof:
                errors.append(f"missing required branch proof field: {key}")
        else:
            checks["branch_proof_fields_complete"] = True
            sanity_errors: list[str] = []
            if proof_obj.get("source_branch") != "agent-initiative-base":
                sanity_errors.append("branch proof source_branch must equal agent-initiative-base")
            if not isinstance(proof_obj.get("initiative_commits"), list) or len(proof_obj["initiative_commits"]) == 0:
                sanity_errors.append("branch proof initiative_commits must be a non-empty array")
            if not isinstance(proof_obj.get("files_in_scope"), list) or len(proof_obj["files_in_scope"]) == 0:
                sanity_errors.append("branch proof files_in_scope must be a non-empty array")
            for key in ("local_source_head", "origin_source_head", "pre_initiative_base"):
                val = proof_obj.get(key)
                if not isinstance(val, str) or not val.strip():
                    sanity_errors.append(f"branch proof {key} must be a non-empty string")
            if (
                isinstance(proof_obj.get("local_source_head"), str)
                and isinstance(proof_obj.get("origin_source_head"), str)
                and proof_obj["local_source_head"] != proof_obj["origin_source_head"]
            ):
                sanity_errors.append(
                    "branch proof head mismatch: local_source_head and origin_source_head must match exactly"
                )
            if (
                isinstance(proof_obj.get("local_source_head"), str)
                and isinstance(proof_obj.get("initiative_commits"), list)
                and len(proof_obj["initiative_commits"]) > 0
                and proof_obj["local_source_head"] not in proof_obj["initiative_commits"]
            ):
                sanity_errors.append(
                    "branch proof commit-chain mismatch: initiative_commits must include local_source_head exactly"
                )
            if (
                isinstance(proof_obj.get("pre_initiative_base"), str)
                and isinstance(proof_obj.get("local_source_head"), str)
                and isinstance(proof_obj.get("initiative_commit_range"), str)
                and proof_obj["initiative_commit_range"]
                != f"{proof_obj['pre_initiative_base']}..{proof_obj['local_source_head']}"
            ):
                sanity_errors.append(
                    "branch proof commit-range mismatch: initiative_commit_range must equal pre_initiative_base..local_source_head exactly"
                )
            if sanity_errors:
                errors.extend(sanity_errors)
            else:
                checks["branch_proof_sanity"] = True

    if checks["manifest_fields_complete"] and checks["required_files_present"]:
        assert manifest_obj is not None
        artifact_hashes = manifest_obj["artifact_hashes"]
        hash_errors: list[str] = []
        for rel in REQUIRED_MANIFEST_HASH_FIELDS:
            expected = artifact_hashes.get(rel)
            if not isinstance(expected, str):
                hash_errors.append(f"manifest hash entry must be string: {rel}")
                continue
            actual = _sha256(required_paths[rel])
            if actual != expected:
                hash_errors.append(f"artifact hash mismatch: {rel}")
        if hash_errors:
            errors.extend(hash_errors)
        else:
            checks["artifact_hashes_match"] = True

    _ = evidence_obj

    status = "pass" if not errors else "invalid_packet"
    exit_code = 0 if status == "pass" else 2
    result = {
        "status": status,
        "packet_dir": str(packet_dir),
        "run_id": run_id,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "gate_status": gate_status,
        "validated_at": validated_at,
    }
    return exit_code, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate completeness/integrity of an initiative-lane promotion packet.")
    parser.add_argument("--run-id")
    parser.add_argument("--validated-at")
    parser.add_argument("--packet-root", default=str(DEFAULT_PACKET_ROOT))
    args, unknown_args = parser.parse_known_args()

    exit_code, payload = validate_promotion_packet(
        run_id=args.run_id,
        validated_at=args.validated_at,
        packet_root=Path(args.packet_root),
        cli_unknown_args=unknown_args,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
