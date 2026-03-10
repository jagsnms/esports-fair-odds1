"""Emit the Phase 1 seeded simulation summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.simulation.phase1 import emit_phase1_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit seeded simulation Phase 1 summary.")
    parser.add_argument("--seed", type=int, default=20260310, help="Explicit deterministic seed.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = emit_phase1_summary(seed=args.seed, output_path=args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())