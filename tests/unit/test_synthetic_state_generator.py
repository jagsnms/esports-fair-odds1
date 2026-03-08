from __future__ import annotations

import asyncio
from pathlib import Path

from tools.replay_verification_assess import run_assessment
from tools.synthetic_state_generator import (
    generate_synthetic_raw_replay,
    write_synthetic_raw_replay_jsonl,
)


def test_synthetic_generator_is_seed_deterministic() -> None:
    a = generate_synthetic_raw_replay(seed=123, rounds=6, ticks_per_round=4)
    b = generate_synthetic_raw_replay(seed=123, rounds=6, ticks_per_round=4)
    c = generate_synthetic_raw_replay(seed=124, rounds=6, ticks_per_round=4)
    assert a == b
    assert a != c


def test_synthetic_generator_covers_stress_axes() -> None:
    payloads = generate_synthetic_raw_replay(seed=42, rounds=8, ticks_per_round=4)
    assert len(payloads) == 32

    bomb_vals = {bool(p.get("is_bomb_planted")) for p in payloads}
    assert bomb_vals == {False, True}

    times = [int(p.get("round_time_remaining", 0)) for p in payloads]
    assert max(times) >= 100_000
    assert min(times) <= 10_000

    alive_counts: set[int] = set()
    regimes: set[str] = set()
    for p in payloads:
        regimes.add(str(p.get("synthetic_regime_team_one", "")))
        regimes.add(str(p.get("synthetic_regime_team_two", "")))
        for team_key in ("team_one", "team_two"):
            team = p.get(team_key) or {}
            states = team.get("player_states") or []
            alive_counts.add(sum(1 for row in states if bool(row.get("is_alive"))))
    assert {3, 4, 5}.issubset(alive_counts)
    assert {"eco", "half", "full"}.issubset(regimes)

    # Post-plant countdown should decrease within a round.
    by_round: dict[int, list[int]] = {}
    for p in payloads:
        if p.get("is_bomb_planted") is True:
            rn = int(p.get("round_number", 0))
            by_round.setdefault(rn, []).append(int(p.get("round_time_remaining", 0)))
    assert by_round, "expected at least one post-plant round"
    for _, tvals in by_round.items():
        if len(tvals) > 1:
            assert tvals == sorted(tvals, reverse=True)


def test_synthetic_generator_integrates_with_replay_assessment(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay_synthetic_seeded_v1.jsonl"
    n_rows = write_synthetic_raw_replay_jsonl(
        replay_path,
        seed=2026,
        rounds=6,
        ticks_per_round=4,
    )
    summary = asyncio.run(run_assessment(str(replay_path), prematch_map=0.55))

    assert n_rows == 24
    assert summary["direct_load_payload_count"] == n_rows
    assert summary["replay_payload_count_loaded"] == n_rows
    # Runner may emit additional deterministic points around segment boundaries.
    assert summary["total_points_captured"] >= n_rows
    assert summary["raw_contract_points"] == summary["total_points_captured"]
    assert summary["rail_input_v2_activated_points"] == summary["total_points_captured"]
    assert summary["rail_input_v1_fallback_points"] == 0
    assert summary["structural_violations_total"] == 0
    assert summary["invariant_violations_total"] == 0
    assert summary["contract_diagnostics_postplant_timer_points"] > 0
