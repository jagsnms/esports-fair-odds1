#!/usr/bin/env python3
"""
Validate one replay summary artifact against replay_validation_summary schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - exercised in runtime environments without jsonschema
    Draft202012Validator = None


SCHEMA_VERSION = "replay_validation_summary_validator.v1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = ROOT / "tools" / "schemas" / "replay_validation_summary.schema.json"


def _empty_result(artifact_path: str, schema_path: str, validated_at: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "artifact_path": artifact_path,
        "schema_path": schema_path,
        "validated_at": validated_at,
        "errors": [],
        "violations": [],
        "summary_metadata": {
            "schema_version": None,
            "fixture_class": None,
            "replay_path": None,
        },
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


def _extract_summary_metadata(artifact_obj: Any) -> dict[str, Any]:
    out = {
        "schema_version": None,
        "fixture_class": None,
        "replay_path": None,
    }
    if not isinstance(artifact_obj, dict):
        return out
    for key in out:
        if key in artifact_obj:
            out[key] = artifact_obj.get(key)
    return out


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


def validate_replay_validation_summary(
    *,
    artifact_path: Path,
    schema_path: Path,
    validated_at: str | None,
) -> tuple[int, dict[str, Any]]:
    result = _empty_result(
        artifact_path=str(artifact_path),
        schema_path=str(schema_path),
        validated_at=validated_at,
    )

    if not isinstance(validated_at, str) or not validated_at.strip():
        result["errors"] = ["invalid validated_at: required explicit non-empty string"]
        return 1, result

    if Draft202012Validator is None:
        result["errors"] = ["schema validator unavailable: jsonschema import failed"]
        return 1, result

    artifact_obj, artifact_err = _load_json(artifact_path, "artifact")
    if artifact_err is not None:
        result["errors"] = [artifact_err]
        return 1, result
    result["summary_metadata"] = _extract_summary_metadata(artifact_obj)

    schema_obj, schema_err = _load_json(schema_path, "schema")
    if schema_err is not None:
        result["errors"] = [schema_err]
        return 1, result
    if not isinstance(schema_obj, dict):
        result["errors"] = ["schema invalid JSON shape: expected top-level object"]
        return 1, result

    try:
        validator = Draft202012Validator(schema_obj)
    except Exception as exc:
        result["errors"] = [f"schema unusable validator setup: {exc}"]
        return 1, result

    # Optional nested replay_point_source data is validated entirely by schema when present.
    violations = sorted(
        list(validator.iter_errors(artifact_obj)),
        key=lambda err: (
            tuple(str(part) for part in err.absolute_path),
            err.message,
            err.validator,
        ),
    )
    if violations:
        result["status"] = "fail"
        result["violations"] = [
            f"{_path_to_str(list(err.absolute_path))}: {err.message}" for err in violations
        ]
        return 2, result

    result["status"] = "pass"
    return 0, result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one replay summary artifact against replay_validation_summary schema."
    )
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--validated-at", required=True)
    args = parser.parse_args()

    exit_code, payload = validate_replay_validation_summary(
        artifact_path=Path(args.artifact_path),
        schema_path=Path(args.schema_path),
        validated_at=args.validated_at,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
