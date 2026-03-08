from __future__ import annotations

import asyncio
from pathlib import Path

from tools.replay_verification_assess import run_assessment
from tools.synthetic_state_generator import (
    EXECUTE_POST_PLANT_PHASES,
    EXECUTE_PRE_PLANT_PHASES,
    POLICY_FAMILIES,
    POLICY_PROFILE_QUOTAS_32,
    POLICY_PROFILES,
    POLICY_PHASE_ORDER,
    POLICY_ROUND_INTENTS,
    RETAKE_POST_PLANT_PHASES,
    RETAKE_PRE_PLANT_PHASES,
    generate_synthetic_distribution_summary,
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


def _team_totals(row: dict, team_key: str) -> tuple[int, int]:
    states = ((row.get(team_key) or {}).get("player_states") or [])
    cash = sum(max(0, int(p.get("balance", 0) or 0)) for p in states)
    loadout = sum(max(0, int(p.get("equipment_value", 0) or 0)) for p in states)
    return cash, loadout


def _eco_force_ratios(row: dict) -> tuple[float, float]:
    cash_a, load_a = _team_totals(row, "team_one")
    cash_b, load_b = _team_totals(row, "team_two")
    cash_ratio = max(cash_a, cash_b) / max(1, min(cash_a, cash_b))
    load_ratio = max(load_a, load_b) / max(1, min(load_a, load_b))
    return cash_ratio, load_ratio


def _eco_force_winner_side_by_cash(row: dict) -> str:
    cash_a, _ = _team_totals(row, "team_one")
    cash_b, _ = _team_totals(row, "team_two")
    if cash_a == cash_b:
        return "tie"
    return "team_one" if cash_a > cash_b else "team_two"


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


def test_execute_retake_contract_with_literal_phase_names() -> None:
    """Hard contract check using literal phase names to prevent shared-constant drift."""
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    execute_phase_planted = {
        "setup": False,
        "commit": False,
        "pressure": True,
        "resolution": True,
    }
    retake_phase_planted = {
        "site_loss": False,
        "retake_setup": False,
        "retake_attempt": True,
        "retake_resolution": True,
    }
    saw_execute = False
    saw_retake = False
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family == "execute":
            saw_execute = True
            phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
            assert set(phases) == set(execute_phase_planted)
            seen_true = False
            for row in rows:
                phase = str(row.get("synthetic_policy_phase", ""))
                planted = bool(row.get("is_bomb_planted"))
                assert phase in execute_phase_planted
                assert planted is execute_phase_planted[phase]
                if planted:
                    seen_true = True
                if seen_true:
                    assert planted is True
        if family == "retake":
            saw_retake = True
            phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
            assert set(phases) == set(retake_phase_planted)
            seen_true = False
            for row in rows:
                phase = str(row.get("synthetic_policy_phase", ""))
                planted = bool(row.get("is_bomb_planted"))
                assert phase in retake_phase_planted
                assert planted is retake_phase_planted[phase]
                if phase == "retake_attempt":
                    seen_true = True
                if seen_true:
                    assert planted is True
    assert saw_execute
    assert saw_retake


def test_low_tick_policy_guard_behavior_is_explicit() -> None:
    """Low-tick behavior guard: ticks=2 constrains families; ticks=3 keeps retake semantics."""
    payloads_2 = generate_synthetic_raw_replay(seed=99, rounds=8, ticks_per_round=2)
    by_round_2 = _rows_by_round(payloads_2)
    families_2 = {str(rows[0].get("synthetic_policy_family", "")) for rows in by_round_2.values()}
    assert families_2.issubset({"execute", "eco_force"})
    assert "retake" not in families_2
    assert "clutch" not in families_2
    for rows in by_round_2.values():
        if str(rows[0].get("synthetic_policy_family", "")) == "execute":
            phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
            planted = [bool(row.get("is_bomb_planted")) for row in rows]
            assert phases == ["setup", "pressure"]
            assert planted == [False, True]

    payloads_3 = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=3)
    by_round_3 = _rows_by_round(payloads_3)
    families_3 = {str(rows[0].get("synthetic_policy_family", "")) for rows in by_round_3.values()}
    assert "clutch" not in families_3
    assert "execute" in families_3
    assert "retake" in families_3
    for rows in by_round_3.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
        planted = [bool(row.get("is_bomb_planted")) for row in rows]
        if family == "execute":
            assert phases == ["setup", "commit", "pressure"]
            assert planted == [False, False, True]
        if family == "retake":
            assert phases == ["site_loss", "retake_setup", "retake_attempt"]
            assert planted == [False, False, True]


def test_clutch_policy_alive_contract_enforced() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=16, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    clutch_rounds = 0
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "clutch":
            continue
        clutch_rounds += 1
        alive_series: list[tuple[int, int]] = []
        phase_seen: list[str] = []
        for row in rows:
            phase = str(row.get("synthetic_policy_phase", ""))
            phase_seen.append(phase)
            states_a = ((row.get("team_one") or {}).get("player_states") or [])
            states_b = ((row.get("team_two") or {}).get("player_states") or [])
            alive_a = sum(1 for p in states_a if p.get("is_alive") is True)
            alive_b = sum(1 for p in states_b if p.get("is_alive") is True)
            alive_series.append((alive_a, alive_b))
            if phase == "isolation":
                assert alive_a >= 2 and alive_b >= 2
        assert phase_seen == ["isolation", "duel_setup", "duel_resolution", "terminal"]
        # No resurrection artifacts: alive counts are non-increasing per team.
        for i in range(1, len(alive_series)):
            prev_a, prev_b = alive_series[i - 1]
            cur_a, cur_b = alive_series[i]
            assert cur_a <= prev_a
            assert cur_b <= prev_b
        duel_res_a, duel_res_b = alive_series[2]
        terminal_a, terminal_b = alive_series[3]
        assert min(duel_res_a, duel_res_b) <= 2
        assert min(terminal_a, terminal_b) <= 2
        assert terminal_a <= duel_res_a
        assert terminal_b <= duel_res_b
    assert clutch_rounds > 0


def test_eco_force_asymmetry_contract_enforced() -> None:
    payloads = generate_synthetic_raw_replay(seed=99, rounds=20, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    eco_rounds = 0
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "eco_force":
            continue
        eco_rounds += 1
        assert [str(r.get("synthetic_policy_phase", "")) for r in rows] == [
            "economy_setup",
            "contact",
            "trade_or_save",
            "terminal",
        ]
        phase_map = {str(r.get("synthetic_policy_phase", "")): r for r in rows}
        setup_cash_ratio, setup_load_ratio = _eco_force_ratios(phase_map["economy_setup"])
        contact_cash_ratio, contact_load_ratio = _eco_force_ratios(phase_map["contact"])
        trade_cash_ratio, trade_load_ratio = _eco_force_ratios(phase_map["trade_or_save"])
        assert (setup_cash_ratio >= 1.5) or (setup_load_ratio >= 1.5)
        assert (contact_cash_ratio >= 1.5) or (contact_load_ratio >= 1.5)
        assert (trade_cash_ratio >= 1.2) or (trade_load_ratio >= 1.2)
        # Forbidden early inversion: setup winner side must match contact winner side.
        setup_winner = _eco_force_winner_side_by_cash(phase_map["economy_setup"])
        contact_winner = _eco_force_winner_side_by_cash(phase_map["contact"])
        assert setup_winner != "tie"
        assert contact_winner == setup_winner
        # Terminal may narrow but must not invert relative to setup.
        terminal_winner = _eco_force_winner_side_by_cash(phase_map["terminal"])
        assert terminal_winner in (setup_winner, "tie")
    assert eco_rounds > 0


def test_clutch_eco_literal_phase_contract_hardening() -> None:
    """Literal phase-name contract hardening for Stage 2B."""
    payloads = generate_synthetic_raw_replay(seed=99, rounds=20, ticks_per_round=4)
    by_round = _rows_by_round(payloads)
    saw_clutch = False
    saw_eco = False
    for rows in by_round.values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        phases = [str(r.get("synthetic_policy_phase", "")) for r in rows]
        if family == "clutch":
            saw_clutch = True
            assert phases == ["isolation", "duel_setup", "duel_resolution", "terminal"]
            alive_min = []
            for row in rows:
                states_a = ((row.get("team_one") or {}).get("player_states") or [])
                states_b = ((row.get("team_two") or {}).get("player_states") or [])
                alive_a = sum(1 for p in states_a if p.get("is_alive") is True)
                alive_b = sum(1 for p in states_b if p.get("is_alive") is True)
                alive_min.append(min(alive_a, alive_b))
            assert alive_min[2] <= 2
            assert alive_min[3] <= 2
        if family == "eco_force":
            saw_eco = True
            assert phases == ["economy_setup", "contact", "trade_or_save", "terminal"]
            setup_ratio = _eco_force_ratios(rows[0])
            contact_ratio = _eco_force_ratios(rows[1])
            trade_ratio = _eco_force_ratios(rows[2])
            assert (setup_ratio[0] >= 1.5) or (setup_ratio[1] >= 1.5)
            assert (contact_ratio[0] >= 1.5) or (contact_ratio[1] >= 1.5)
            assert (trade_ratio[0] >= 1.2) or (trade_ratio[1] >= 1.2)
    assert saw_clutch
    assert saw_eco


def test_eco_force_low_tick_behavior_is_explicit() -> None:
    payloads_2 = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=2)
    for rows in _rows_by_round(payloads_2).values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "eco_force":
            continue
        phases = [str(r.get("synthetic_policy_phase", "")) for r in rows]
        assert phases == ["economy_setup", "trade_or_save"]
        setup_ratio = _eco_force_ratios(rows[0])
        trade_ratio = _eco_force_ratios(rows[1])
        assert (setup_ratio[0] >= 1.5) or (setup_ratio[1] >= 1.5)
        assert (trade_ratio[0] >= 1.2) or (trade_ratio[1] >= 1.2)

    payloads_3 = generate_synthetic_raw_replay(seed=99, rounds=12, ticks_per_round=3)
    for rows in _rows_by_round(payloads_3).values():
        family = str(rows[0].get("synthetic_policy_family", ""))
        if family != "eco_force":
            continue
        phases = [str(r.get("synthetic_policy_phase", "")) for r in rows]
        assert phases == ["economy_setup", "contact", "trade_or_save"]
        setup_ratio = _eco_force_ratios(rows[0])
        contact_ratio = _eco_force_ratios(rows[1])
        trade_ratio = _eco_force_ratios(rows[2])
        assert (setup_ratio[0] >= 1.5) or (setup_ratio[1] >= 1.5)
        assert (contact_ratio[0] >= 1.5) or (contact_ratio[1] >= 1.5)
        assert (trade_ratio[0] >= 1.2) or (trade_ratio[1] >= 1.2)


def test_stage3a_distribution_summary_same_seed_is_identical() -> None:
    a = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=4,
        policy_profile="balanced_v1",
    )
    b = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=4,
        policy_profile="balanced_v1",
    )
    c = generate_synthetic_distribution_summary(
        seed=100,
        rounds=32,
        ticks_per_round=4,
        policy_profile="balanced_v1",
    )
    assert a == b
    assert a["family_sequence"] != c["family_sequence"]


def test_stage3a_profile_quota_conformance_reference_run() -> None:
    for profile in POLICY_PROFILES:
        summary = generate_synthetic_distribution_summary(
            seed=99,
            rounds=32,
            ticks_per_round=4,
            policy_profile=profile,
        )
        expected = POLICY_PROFILE_QUOTAS_32[profile]
        assert summary["policy_profile"] == profile
        assert summary["target_family_quotas"] == expected
        assert summary["effective_family_quotas"] == expected
        assert summary["realized_family_counts"] == expected
        assert all(int(v) == 0 for v in summary["quota_delta"].values())
        assert summary["feasibility_adjustment_applied"] is False
        assert summary["feasibility_adjustment_reason"] == ""
        for key in (
            "one_family_per_round_violations",
            "family_immutability_violations",
            "phase_monotonicity_violations",
            "intent_family_mismatch_violations",
            "execute_retake_contract_violations",
            "clutch_contract_violations",
            "eco_force_contract_violations",
        ):
            assert int(summary[key]) == 0


def test_stage3a_profile_redistribution_is_deterministic_and_reported() -> None:
    summary_ticks_2 = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=2,
        policy_profile="balanced_v1",
    )
    summary_ticks_2_again = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=2,
        policy_profile="balanced_v1",
    )
    assert summary_ticks_2 == summary_ticks_2_again
    assert summary_ticks_2["feasibility_adjustment_applied"] is True
    assert "reduced by" in str(summary_ticks_2["feasibility_adjustment_reason"])
    assert int(summary_ticks_2["effective_family_quotas"]["retake"]) == 0
    assert int(summary_ticks_2["effective_family_quotas"]["clutch"]) == 0
    assert summary_ticks_2["realized_family_counts"] == summary_ticks_2["effective_family_quotas"]

    summary_ticks_3 = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=3,
        policy_profile="eco_bias_v1",
    )
    assert summary_ticks_3["feasibility_adjustment_applied"] is True
    assert int(summary_ticks_3["effective_family_quotas"]["clutch"]) == 0
    assert int(summary_ticks_3["effective_family_quotas"]["retake"]) > 0
    assert summary_ticks_3["realized_family_counts"] == summary_ticks_3["effective_family_quotas"]


def test_stage3a_distribution_summary_fields_present_and_typed() -> None:
    summary = generate_synthetic_distribution_summary(
        seed=99,
        rounds=32,
        ticks_per_round=4,
        policy_profile="execute_bias_v1",
    )
    required = {
        "seed",
        "rounds",
        "ticks_per_round",
        "policy_profile",
        "target_family_quotas",
        "effective_family_quotas",
        "realized_family_counts",
        "realized_family_rates",
        "quota_delta",
        "family_sequence",
        "feasibility_adjustment_applied",
        "feasibility_adjustment_reason",
        "one_family_per_round_violations",
        "family_immutability_violations",
        "phase_monotonicity_violations",
        "intent_family_mismatch_violations",
        "execute_retake_contract_violations",
        "clutch_contract_violations",
        "eco_force_contract_violations",
    }
    assert required.issubset(summary.keys())
    assert isinstance(summary["family_sequence"], list)
    assert len(summary["family_sequence"]) == 32
    assert set(summary["realized_family_counts"]) == set(POLICY_FAMILIES)
    assert set(summary["realized_family_rates"]) == set(POLICY_FAMILIES)
    assert set(summary["quota_delta"]) == set(POLICY_FAMILIES)


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
