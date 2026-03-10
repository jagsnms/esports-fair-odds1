# GRID PROBE V1 — Small helper to inspect saved JSON (structure to limited depth).
"""
Usage:
  python -m adapters.grid_probe.inspect_grid_json [raw_grid_central_data.json|raw_grid_series_state.json]
  Default file: raw_grid_series_state.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
MAX_DEPTH = 4


def _summary(obj: object, depth: int, prefix: str = "") -> None:
    if depth <= 0:
        print(f"{prefix}...")
        return
    if obj is None:
        print(f"{prefix}null")
        return
    if isinstance(obj, bool):
        print(f"{prefix}{obj}")
        return
    if isinstance(obj, (int, float)):
        print(f"{prefix}{obj}")
        return
    if isinstance(obj, str):
        s = obj[:60] + "..." if len(obj) > 60 else obj
        print(f"{prefix}{s!r}")
        return
    if isinstance(obj, list):
        print(f"{prefix}[list len={len(obj)}]")
        for i, item in enumerate(obj[:5]):
            _summary(item, depth - 1, prefix + f"  [{i}] ")
        if len(obj) > 5:
            print(f"{prefix}  ... and {len(obj) - 5} more")
        return
    if isinstance(obj, dict):
        print(f"{prefix}{{dict keys: {list(obj.keys())[:12]}}}")
        for k, v in list(obj.items())[:10]:
            _summary(v, depth - 1, prefix + f"  {k!r}: ")
        if len(obj) > 10:
            print(f"{prefix}  ... and {len(obj) - 10} more keys")
        return
    print(f"{prefix}{type(obj).__name__}")


def main() -> None:
    # GRID PROBE V1
    name = sys.argv[1] if len(sys.argv) > 1 else "raw_grid_series_state.json"
    path = PROBE_DIR / name
    if not path.exists():
        path = Path(name)
    if not path.exists():
        print(f"File not found: {path}")
        print("Usage: python -m adapters.grid_probe.inspect_grid_json [filename]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[GRID PROBE V1] Structure of {path} (depth={MAX_DEPTH}):")
    _summary(data, MAX_DEPTH)


if __name__ == "__main__":
    main()
