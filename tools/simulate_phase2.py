"""Emit the bounded Phase 2 policy-driven simulation summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.simulation.phase2 import (
    PHASE2_SECOND_SOURCE_POLICY_PROFILE,
    PHASE2_STAGE1_POLICY_PROFILE,
    emit_phase2_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit bounded Phase 2 policy-driven simulation summary.")
    parser.add_argument("--seed", type=int, default=20260310, help="Explicit deterministic seed.")
    parser.add_argument(
        "--policy-profile",
        default=PHASE2_STAGE1_POLICY_PROFILE,
        choices=[PHASE2_STAGE1_POLICY_PROFILE, PHASE2_SECOND_SOURCE_POLICY_PROFILE],
        help="Bounded canonical Phase 2 policy profile.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = emit_phase2_summary(
        seed=args.seed,
        output_path=args.output,
        policy_profile=args.policy_profile,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
