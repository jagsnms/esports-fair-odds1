"""
Shared JSONL recording for external data pulls (BO3, GRID). Safe append-only; never logs secrets.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_iso_z() -> str:
    """Return UTC timestamp like 2026-02-22T21:14:05.123Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append one JSON object as a single line to path. Creates parent dirs. Never logs API keys or secrets.
    On any error, prints a short warning and returns without raising.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[record_jsonl] append failed: {path}: {e!r}")
