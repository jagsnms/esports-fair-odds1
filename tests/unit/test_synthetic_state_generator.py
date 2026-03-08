from __future__ import annotations

import asyncio
from pathlib import Path

from tools.replay_verification_assess import run_assessment
from tools.synthetic_state_generator import (
    EXECUTE_POST_PLANT_PHASES,
    EXECUTE_PRE_PLANT_PHASES,
    POLICY_FAMILIES,
    POLICY_PHASE_ORDER,
    POLICY_ROUND_INTENTS,
    RETAKE_POST_PLANT_PHASES,
    RETAKE_PRE_PLANT_PHASES,
    generate_synthetic_raw_replay,
    write_synthetic_raw_replay_jsonl,
)


def _rows_by_round(payloads: list[dict]) -> dict[int, list[dict]]:
    by_round: dict[int, list[dict]] = {}
    for row in payloads:
        rn = int(row.get("round_number", 0))
        by_round.setdefault(rn, []).append(row)
    return by_round


def _policy_family_sequence(payloads: list[dict]) -> list[str]:
    by_round = _rows_by_round(payloads)
    return [str(by_round[rn][0].get("synthetic_policy_family", "")) for rn in sorted(by_round)]


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
    if "contract_diagnostics_postplant_timer_points" in summary:
        assert summary["contract_diagnostics_postplant_timer_points"] > 0


def test_policy_family_sequence_is_seed_deterministic() -> None:
    a = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    b = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    c = generate_synthetic_raw_replay(seed=100, rounds=16, ticks_per_round=4)
    seq_a = _policy_family_sequence(a)
    seq_b = _policy_family_sequence(b)
    seq_c = _policy_family_sequence(c)
    assert seq_a == seq_b
    assert seq_a != seq_c


def test_policy_metadata_fields_present_and_nonempty() -> None:
    payloads = generate_synthetic_raw_replay(seed=7, rounds=6, ticks_per_round=4)
    for row in payloads:
        for key in (
            "synthetic_policy_family",
            "synthetic_policy_phase",
            "synthetic_policy_round_intent",
        ):
            assert key in row
            assert isinstance(row[key], str)
            assert row[key].strip()


def test_policy_round_intent_matches_family_mapping() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=4)
    for row in payloads:
        family = str(row.get("synthetic_policy_family", ""))
        intent = str(row.get("synthetic_policy_round_intent", ""))
        assert family in POLICY_ROUND_INTENTS
        assert intent == POLICY_ROUND_INTENTS[family]


def test_policy_coverage_reference_run_hits_all_families() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    families = set(_policy_family_sequence(payloads))
    assert families == set(POLICY_FAMILIES)


def test_policy_family_is_immutable_within_round() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    for rows in by_round.values():
        labels = {str(row.get("synthetic_policy_family", "")) for row in rows}
        assert len(labels) == 1


def test_policy_phase_progression_is_monotonic_within_round() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        allowed = POLICY_PHASE_ORDER[family]
        rank = {phase: idx for idx, phase in enumerate(allowed)}
        phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
        assert all(phase in rank for phase in phases)
        phase_ranks = [rank[phase] for phase in phases]
        assert phase_ranks == sorted(phase_ranks)


def test_execute_policy_phase_planted_contract_enforced() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    execute_rounds = 0
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "execute":
            continue
        execute_rounds += 1
        planted_flags: list[bool] = []
        saw_post_plant_phase = False
        for row in rows:
            phase = str(row.get("synthetic_policy_phase", ""))
            planted = bool(row.get("is_bomb_planted"))
            if phase in EXECUTE_PRE_PLANT_PHASES:
                assert planted is False
            if phase in EXECUTE_POST_PLANT_PHASES:
                assert planted is True
                saw_post_plant_phase = True
            planted_flags.append(planted)
        assert saw_post_plant_phase
        assert any(planted_flags), "execute round cannot remain entirely pre-plant"
        seen_true = False
        for flag in planted_flags:
            if flag:
                seen_true = True
            if seen_true:
                assert flag is True
    assert execute_rounds > 0


def test_retake_policy_phase_planted_contract_enforced() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    retake_rounds = 0
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "retake":
            continue
        retake_rounds += 1
        planted_flags: list[bool] = []
        retake_attempt_started = False
        for row in rows:
            phase = str(row.get("synthetic_policy_phase", ""))
            planted = bool(row.get("is_bomb_planted"))
            if phase in RETAKE_PRE_PLANT_PHASES:
                assert planted is False
            if phase in RETAKE_POST_PLANT_PHASES:
                assert planted is True
                if phase == "retake_attempt":
                    retake_attempt_started = True
            planted_flags.append(planted)
        assert retake_attempt_started, "retake round must include retake_attempt phase"
        seen_attempt_true = False
        for row in rows:
            phase = str(row.get("synthetic_policy_phase", ""))
            planted = bool(row.get("is_bomb_planted"))
            if phase in RETAKE_POST_PLANT_PHASES and phase == "retake_attempt":
                seen_attempt_true = True
            if seen_attempt_true:
                assert planted is True
        assert any(planted_flags), "retake round must become post-plant"
    assert retake_rounds > 0


def test_policy_label_coherence_for_retake_and_clutch() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family == "retake":
            assert any(row.get("is_bomb_planted") is True for row in rows)
        if family == "clutch":
            has_low_man_tick = False
            for row in rows:
                states_a = ((row.get("team_one") or {}).get("player_states") or [])
                states_b = ((row.get("team_two") or {}).get("player_states") or [])
                alive_a = sum(1 for p in states_a if p.get("is_alive") is True)
                alive_b = sum(1 for p in states_b if p.get("is_alive") is True)
                if min(alive_a, alive_b) <= 2:
                    has_low_man_tick = True
                    break
            assert has_low_man_tick
