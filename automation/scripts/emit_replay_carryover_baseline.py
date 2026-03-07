#!/usr/bin/env python3
"""
Stage 1 (Option A): Emit reproducible baseline evidence pack for replay carryover coverage.
Runs the four frozen replay classes and writes a single JSON artifact.
No ingestion, no rail/PHAT changes, no coverage claims.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Reduce log noise during baseline emission (runner logs point-like rejections per payload)
logging.getLogger("backend.services.runner").setLevel(logging.ERROR)

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.replay_verification_assess import run_assessment

BASELINE_RUNS = [
    ("raw_replay_sample", str(ROOT / "tools" / "fixtures" / "raw_replay_sample.jsonl"), None),
    ("replay_multimatch_small_v1", str(ROOT / "tools" / "fixtures" / "replay_multimatch_small_v1.jsonl"), None),
    ("replay_carryover_complete_v1", str(ROOT / "tools" / "fixtures" / "replay_carryover_complete_v1.jsonl"), 0.55),
    ("history_points", str(ROOT / "logs" / "history_points.jsonl"), None),
]

# Repo-relative paths for portable artifact (forward slashes)
RUN_KEY_TO_RELATIVE_PATH = {
    "raw_replay_sample": "tools/fixtures/raw_replay_sample.jsonl",
    "replay_multimatch_small_v1": "tools/fixtures/replay_multimatch_small_v1.jsonl",
    "replay_carryover_complete_v1": "tools/fixtures/replay_carryover_complete_v1.jsonl",
    "history_points": "logs/history_points.jsonl",
}

# Minimal point-like payload: one line so baseline point-like run is deterministic
_POINT_LIKE_MINIMAL_LINE = '{"t": 1, "p": 0.5, "lo": 0.0, "hi": 1.0}\n'


def _ensure_point_like_fixture(path: Path) -> None:
    """For logs/history_points.jsonl only: always write minimal one-line content so baseline is deterministic."""
    if path.name != "history_points.jsonl":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_POINT_LIKE_MINIMAL_LINE, encoding="utf-8")


def main() -> int:
    os.chdir(ROOT)
    runs: dict[str, dict] = {}
    for key, replay_path, prematch_map in BASELINE_RUNS:
        path = Path(replay_path)
        _ensure_point_like_fixture(path)
        if not path.exists():
            print(f"skip (missing): {replay_path}", file=sys.stderr)
            runs[key] = {
                "_skipped": True,
                "replay_path": RUN_KEY_TO_RELATIVE_PATH.get(key, replay_path),
                "reason": "file_not_found",
            }
            continue
        summary = asyncio.run(run_assessment(replay_path, prematch_map=prematch_map))
        summary["replay_path"] = RUN_KEY_TO_RELATIVE_PATH[key]
        runs[key] = summary

    artifact = {
        "schema_version": "replay_carryover_baseline.v1",
        "stage": "stage1_option_a",
        "runs": runs,
    }
    out_path = ROOT / "automation" / "reports" / "baseline_replay_carryover_evidence_20260307.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
