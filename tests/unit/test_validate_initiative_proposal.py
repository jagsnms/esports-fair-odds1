from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "automation" / "validate_initiative_proposal.py"
FIXTURES = ROOT / "tests" / "fixtures" / "initiative_proposal_validation"


def _run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_json_stdout(proc: subprocess.CompletedProcess[str]) -> dict:
    assert proc.stdout.strip(), f"expected json output, stderr={proc.stderr}"
    return json.loads(proc.stdout)


def test_draft_artifact_is_non_blocking() -> None:
    proc = _run_validator(FIXTURES / "draft_note.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 0
    assert data["status"] == "draft"
    assert data["review_ready"] is False
    assert "Draft/not-applicable" in data["message"]


def test_draft_artifact_with_marker_phrase_in_prose_is_non_blocking() -> None:
    proc = _run_validator(FIXTURES / "draft_marker_phrase_in_prose.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 0
    assert data["status"] == "draft"
    assert data["review_ready"] is False
    assert "Draft/not-applicable" in data["message"]


def test_review_ready_missing_checkpoint_fails() -> None:
    proc = _run_validator(FIXTURES / "review_ready_missing_checkpoint.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 1
    assert data["status"] == "fail"
    assert any("Missing stale-prevention checklist item" in err for err in data["errors"])


def test_review_ready_empty_findings_fails() -> None:
    proc = _run_validator(FIXTURES / "review_ready_empty_findings.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 1
    assert data["status"] == "fail"
    assert any("Empty findings field: Promoted registry findings" in err for err in data["errors"])


def test_review_ready_ambiguous_disposition_fails() -> None:
    proc = _run_validator(FIXTURES / "review_ready_ambiguous_disposition.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 1
    assert data["status"] == "fail"
    assert any("Invalid disposition selection count" in err for err in data["errors"])


def test_review_ready_complete_artifact_passes() -> None:
    proc = _run_validator(FIXTURES / "review_ready_pass.md")
    data = _parse_json_stdout(proc)
    assert proc.returncode == 0
    assert data["status"] == "pass"
    assert data["review_ready"] is True
    assert data["errors"] == []
    assert "syntactically complete" in data["message"]
    assert "does not verify truth, novelty, or git/origin branch state" in data["message"]
