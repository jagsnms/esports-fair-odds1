"""Policy-driven canonical simulation Phase 2 Stage 1 contract."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from tools.replay_verification_assess import run_assessment
from tools.synthetic_state_generator import (
    POLICY_FAMILIES,
    generate_synthetic_distribution_summary,
    write_synthetic_raw_replay_jsonl,
)

SIMULATION_PHASE2_SUMMARY_VERSION = "simulation_phase2_policy_summary.v1"
CANONICAL_ENGINE_PATH = "compute_bounds>compute_rails>resolve_p_hat"
PHASE2_STAGE1_POLICY_PROFILE = "balanced_v1"
PHASE2_STAGE1_ROUNDS = 32
PHASE2_STAGE1_TICKS_PER_ROUND = 4
PHASE2_STAGE1_FIXTURE_CLASS = "phase2_balanced_v1_policy_canonical"


def _sanitize_replay_summary(replay_summary: dict[str, Any], *, seed: int) -> dict[str, Any]:
    sanitized = dict(replay_summary)
    sanitized["fixture_class"] = PHASE2_STAGE1_FIXTURE_CLASS
    sanitized["replay_path"] = f"synthetic://phase2/{PHASE2_STAGE1_POLICY_PROFILE}/seed/{int(seed)}"
    sanitized["replay_path_exists"] = False
    return sanitized


def generate_phase2_summary(seed: int) -> dict[str, Any]:
    policy_distribution = generate_synthetic_distribution_summary(
        seed=int(seed),
        rounds=PHASE2_STAGE1_ROUNDS,
        ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
        policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
    )

    with tempfile.TemporaryDirectory(prefix="phase2_balanced_v1.") as tmpdir:
        replay_path = Path(tmpdir) / "phase2_balanced_v1_policy_raw.jsonl"
        generated_payload_count = write_synthetic_raw_replay_jsonl(
            replay_path,
            seed=int(seed),
            rounds=PHASE2_STAGE1_ROUNDS,
            ticks_per_round=PHASE2_STAGE1_TICKS_PER_ROUND,
            policy_profile=PHASE2_STAGE1_POLICY_PROFILE,
        )
        replay_summary = asyncio.run(run_assessment(str(replay_path)))

    replay_comparable_summary = _sanitize_replay_summary(replay_summary, seed=int(seed))
    return {
        "schema_version": SIMULATION_PHASE2_SUMMARY_VERSION,
        "seed": int(seed),
        "policy_profile": PHASE2_STAGE1_POLICY_PROFILE,
        "canonical_engine_path": CANONICAL_ENGINE_PATH,
        "round_count": PHASE2_STAGE1_ROUNDS,
        "ticks_per_round": PHASE2_STAGE1_TICKS_PER_ROUND,
        "generated_payload_count": int(generated_payload_count),
        "policy_families": list(POLICY_FAMILIES),
        "realized_family_counts": dict(policy_distribution["realized_family_counts"]),
        "structural_violations_total": int(replay_comparable_summary["structural_violations_total"]),
        "behavioral_violations_total": int(replay_comparable_summary["behavioral_violations_total"]),
        "invariant_violations_total": int(replay_comparable_summary["invariant_violations_total"]),
        "p_hat_min": replay_comparable_summary["p_hat_min"],
        "p_hat_max": replay_comparable_summary["p_hat_max"],
        "p_hat_count": int(replay_comparable_summary["p_hat_count"]),
        "p_hat_median": replay_comparable_summary["p_hat_median"],
        "policy_distribution": policy_distribution,
        "replay_comparable_summary": replay_comparable_summary,
    }


def emit_phase2_summary(seed: int, output_path: str | Path | None = None) -> dict[str, Any]:
    summary = generate_phase2_summary(seed)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
