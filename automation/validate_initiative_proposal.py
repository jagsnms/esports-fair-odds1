#!/usr/bin/env python3
"""
Minimal syntactic validator for review-ready initiative proposal artifacts.

This validator enforces contract completeness only. It does NOT verify truth,
novelty, branch state, or semantic duplicate detection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Iterable

REVIEW_READY_MARKER = "Review-ready: yes"
REVIEW_READY_MARKER_LINE = re.compile(rf"^{re.escape(REVIEW_READY_MARKER)}$", re.MULTILINE)

REQUIRED_CORE_SECTION_HEADINGS = (
    "## Initiative title",
    "## Why it outranks other major issues",
    "## Why it exceeds bounded-fix scope",
    "## Affected modules/files",
    "## Proposed stages",
    "## Validation checkpoints",
    "## Risks",
    "## Recommended branch plan",
)

REQUIRED_STALE_CHECKLIST_ITEMS = (
    "Checked `automation/PROMOTED_INITIATIVES.md`",
    "Checked `automation/BANKED_INITIATIVES.md`",
    "Checked shared/origin truth (`origin/agent-base` and `origin/agent-initiative-base`)",
    "Confirmed this proposal is **not already promoted/shared truth**",
)

REQUIRED_FINDINGS_FIELDS = (
    "Promoted registry findings",
    "Banked registry findings",
    "Shared/origin truth findings",
    "Non-duplication confirmation",
)

DISPOSITION_OPTIONS = (
    "Approve planning only",
    "Approve stage 1",
    "Defer",
)


@dataclass
class ValidationResult:
    status: str
    review_ready: bool
    artifact_path: str
    errors: list[str] = field(default_factory=list)
    message: str = ""


def _line_with_checkbox_present(text: str, item_text: str) -> bool:
    pat = re.compile(rf"^- \[(?: |x|X)\]\s+{re.escape(item_text)}\s*$", re.MULTILINE)
    return bool(pat.search(text))


def _findings_field_value(text: str, label: str) -> str | None:
    # Keep extraction line-scoped; do not consume newlines into the value.
    pat = re.compile(rf"^- {re.escape(label)}:[ \t]*(.*)$", re.MULTILINE)
    match = pat.search(text)
    if not match:
        return None
    return str(match.group(1))


def _count_selected_dispositions(text: str, options: Iterable[str]) -> tuple[int, list[str]]:
    missing: list[str] = []
    selected = 0
    for option in options:
        pat = re.compile(rf"^- \[(?P<mark>[ xX])\]\s+\*\*{re.escape(option)}\*\*\s+—\s+.*$", re.MULTILINE)
        match = pat.search(text)
        if not match:
            missing.append(option)
            continue
        if str(match.group("mark")).lower() == "x":
            selected += 1
    return selected, missing


def validate_proposal_markdown(text: str, artifact_path: str) -> ValidationResult:
    review_ready = bool(REVIEW_READY_MARKER_LINE.search(text))
    if not review_ready:
        return ValidationResult(
            status="draft",
            review_ready=False,
            artifact_path=artifact_path,
            message=(
                "Draft/not-applicable: review-ready marker not present. "
                "Validator only enforces review-ready proposal artifacts."
            ),
        )

    errors: list[str] = []

    for heading in REQUIRED_CORE_SECTION_HEADINGS:
        if heading not in text:
            errors.append(f"Missing core section heading: {heading}")

    if "## Recommendation / disposition" not in text:
        errors.append("Missing recommendation/disposition field heading: ## Recommendation / disposition")

    for item in REQUIRED_STALE_CHECKLIST_ITEMS:
        if not _line_with_checkbox_present(text, item):
            errors.append(f"Missing stale-prevention checklist item: {item}")

    for field_name in REQUIRED_FINDINGS_FIELDS:
        value = _findings_field_value(text, field_name)
        if value is None:
            errors.append(f"Missing findings field: {field_name}")
            continue
        if not value.strip():
            errors.append(f"Empty findings field: {field_name}")

    selected, missing_options = _count_selected_dispositions(text, DISPOSITION_OPTIONS)
    for missing_option in missing_options:
        errors.append(f"Missing disposition option line: {missing_option}")
    if not missing_options and selected != 1:
        errors.append(f"Invalid disposition selection count: expected exactly 1 selected, got {selected}")

    if errors:
        return ValidationResult(
            status="fail",
            review_ready=True,
            artifact_path=artifact_path,
            errors=errors,
            message=(
                "FAIL: review-ready artifact is missing required contract items. "
                "This validator checks syntactic completeness only; it does not verify truth, "
                "novelty, or git/origin branch state."
            ),
        )

    return ValidationResult(
        status="pass",
        review_ready=True,
        artifact_path=artifact_path,
        message=(
            "PASS: review-ready artifact is syntactically complete for the governance contract. "
            "This pass does not verify truth, novelty, or git/origin branch state."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate review-ready initiative proposal markdown for governance contract completeness."
    )
    parser.add_argument("artifact", help="Path to markdown proposal artifact")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        result = ValidationResult(
            status="fail",
            review_ready=False,
            artifact_path=str(artifact_path),
            errors=[f"Artifact not found: {artifact_path}"],
            message="FAIL: artifact path does not exist.",
        )
        print(json.dumps(asdict(result), ensure_ascii=True))
        return 1

    text = artifact_path.read_text(encoding="utf-8")
    result = validate_proposal_markdown(text=text, artifact_path=str(artifact_path))
    print(json.dumps(asdict(result), ensure_ascii=True))
    return 1 if result.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
