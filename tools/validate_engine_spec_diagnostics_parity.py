#!/usr/bin/env python3
"""
Validate exact diagnostics required-field parity between ENGINE_SPEC and replay assessment artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "engine_spec_diagnostics_parity.v1"
PARITY_MODE = "exact_name_path_only"
ENGINE_SPEC_FIELD_PATH = "invariants.diagnostics_payload_required_fields"
ASSESSMENT_FIELD_PATH = "contract_diagnostics_required_keys"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE_SPEC_PATH = ROOT / "docs" / "ENGINE_SPEC.json"


def _empty_result(engine_spec_path: str, assessment_artifact_path: str, validated_at: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "validated_at": validated_at,
        "parity_mode": PARITY_MODE,
        "source_of_truth": {
            "engine_spec_path": engine_spec_path,
            "engine_spec_field_path": ENGINE_SPEC_FIELD_PATH,
        },
        "assessment_source": {
            "artifact_path_or_stream": assessment_artifact_path,
            "assessment_field_path": ASSESSMENT_FIELD_PATH,
        },
        "spec_required_fields": [],
        "assessment_required_fields": [],
        "matched_fields": [],
        "missing_fields": [],
        "extra_assessment_fields": [],
        "counts": {
            "spec_required_count": 0,
            "assessment_required_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "extra_count": 0,
        },
        "errors": [],
    }


def _load_json(path: Path, label: str) -> tuple[Any | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"{label} unreadable: {path} ({exc})"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{label} invalid JSON: {path} ({exc})"


def _extract_spec_required_fields(spec_obj: Any) -> tuple[list[str] | None, str | None]:
    if not isinstance(spec_obj, dict):
        return None, "engine spec must be a JSON object"
    invariants = spec_obj.get("invariants")
    if not isinstance(invariants, dict):
        return None, "missing/wrong-type path: invariants"
    required = invariants.get("diagnostics_payload_required_fields")
    if not isinstance(required, list):
        return None, "missing/wrong-type path: invariants.diagnostics_payload_required_fields"
    if len(required) == 0:
        return None, "empty path: invariants.diagnostics_payload_required_fields"

    normalized: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(required):
        if not isinstance(item, str):
            return (
                None,
                f"wrong-type value at invariants.diagnostics_payload_required_fields[{idx}]: expected string",
            )
        value = item.strip()
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if len(normalized) == 0:
        return None, "empty path after normalization: invariants.diagnostics_payload_required_fields"
    return normalized, None


def _extract_assessment_required_fields(assessment_obj: Any) -> tuple[list[str] | None, str | None]:
    if not isinstance(assessment_obj, dict):
        return None, "assessment artifact must be a JSON object"
    required = assessment_obj.get("contract_diagnostics_required_keys")
    if not isinstance(required, list):
        return None, "missing/wrong-type path: contract_diagnostics_required_keys"

    normalized: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(required):
        if not isinstance(item, str):
            return None, f"wrong-type value at contract_diagnostics_required_keys[{idx}]: expected string"
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized, None


def validate_diagnostics_parity(
    *,
    engine_spec_path: Path,
    assessment_artifact_path: Path,
    validated_at: str | None = None,
) -> tuple[int, dict[str, Any]]:
    result = _empty_result(
        engine_spec_path=str(engine_spec_path),
        assessment_artifact_path=str(assessment_artifact_path),
        validated_at=validated_at,
    )
    errors: list[str] = []

    spec_obj, spec_err = _load_json(engine_spec_path, "engine spec")
    if spec_err is not None:
        errors.append(spec_err)

    assessment_obj, assessment_err = _load_json(assessment_artifact_path, "assessment artifact")
    if assessment_err is not None:
        errors.append(assessment_err)

    if errors:
        result["errors"] = errors
        return 1, result

    spec_required, spec_extract_err = _extract_spec_required_fields(spec_obj)
    if spec_extract_err is not None:
        result["errors"] = [spec_extract_err]
        return 1, result
    assert spec_required is not None

    assessment_required, assessment_extract_err = _extract_assessment_required_fields(assessment_obj)
    if assessment_extract_err is not None:
        result["errors"] = [assessment_extract_err]
        return 1, result
    assert assessment_required is not None

    assessment_set = set(assessment_required)
    spec_set = set(spec_required)
    matched_fields = [field for field in spec_required if field in assessment_set]
    missing_fields = [
        {
            "field": field,
            "expected_in": ASSESSMENT_FIELD_PATH,
            "match_mode": PARITY_MODE,
            "reason": "missing_in_assessment_required_keys",
        }
        for field in spec_required
        if field not in assessment_set
    ]
    extra_fields = [field for field in assessment_required if field not in spec_set]

    result["spec_required_fields"] = spec_required
    result["assessment_required_fields"] = assessment_required
    result["matched_fields"] = matched_fields
    result["missing_fields"] = missing_fields
    result["extra_assessment_fields"] = extra_fields
    result["counts"] = {
        "spec_required_count": len(spec_required),
        "assessment_required_count": len(assessment_required),
        "matched_count": len(matched_fields),
        "missing_count": len(missing_fields),
        "extra_count": len(extra_fields),
    }
    result["errors"] = []

    if missing_fields:
        result["status"] = "fail"
        return 2, result
    result["status"] = "pass"
    return 0, result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact diagnostics field parity between docs/ENGINE_SPEC.json and a replay assessment JSON artifact."
        )
    )
    parser.add_argument(
        "--engine-spec-path",
        default=str(DEFAULT_ENGINE_SPEC_PATH),
        help="Path to ENGINE_SPEC JSON (default: docs/ENGINE_SPEC.json)",
    )
    parser.add_argument(
        "--assessment-artifact-path",
        required=True,
        help="Path to replay assessment JSON artifact",
    )
    parser.add_argument(
        "--validated-at",
        default=None,
        help="Optional explicit validation timestamp/string included in output payload",
    )
    args = parser.parse_args()

    exit_code, payload = validate_diagnostics_parity(
        engine_spec_path=Path(args.engine_spec_path),
        assessment_artifact_path=Path(args.assessment_artifact_path),
        validated_at=args.validated_at,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
