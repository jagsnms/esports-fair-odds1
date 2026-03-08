#!/usr/bin/env python3
"""
Seeded synthetic raw BO3 replay generator for simulation/stress coverage.

Produces raw-snapshot JSONL compatible with replay raw-contract path
(`team_one`/`team_two` + `round_phase`) and encodes:
- alive trajectories (5->4->3),
- economy/loadout regimes (eco/half/full),
- bomb planted transitions + countdown,
- timer pressure near zero,
- carryover context swings via regime shifts round-to-round.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


REGIMES = ("eco", "half", "full")
POLICY_FAMILIES = ("execute", "retake", "clutch", "eco_force")
POLICY_PROFILES = ("balanced_v1", "execute_bias_v1", "eco_bias_v1")
POLICY_ROUND_INTENTS: dict[str, str] = {
    "execute": "site_take_attempt",
    "retake": "postplant_retake_attempt",
    "clutch": "man_disadvantage_conversion",
    "eco_force": "economy_pressure_round",
}
POLICY_PROFILE_QUOTAS_32: dict[str, dict[str, int]] = {
    "balanced_v1": {"execute": 8, "retake": 8, "clutch": 8, "eco_force": 8},
    "execute_bias_v1": {"execute": 14, "retake": 6, "clutch": 6, "eco_force": 6},
    "eco_bias_v1": {"execute": 8, "retake": 5, "clutch": 5, "eco_force": 14},
}
POLICY_PROFILE_PROPORTIONS: dict[str, dict[str, float]] = {
    profile: {family: (float(quota) / 32.0) for family, quota in quotas.items()}
    for profile, quotas in POLICY_PROFILE_QUOTAS_32.items()
}
POLICY_PHASE_ORDER: dict[str, tuple[str, ...]] = {
    "execute": ("setup", "commit", "pressure", "resolution"),
    "retake": ("site_loss", "retake_setup", "retake_attempt", "retake_resolution"),
    "clutch": ("isolation", "duel_setup", "duel_resolution", "terminal"),
    "eco_force": ("economy_setup", "contact", "trade_or_save", "terminal"),
}
EXECUTE_PRE_PLANT_PHASES = ("setup", "commit")
EXECUTE_POST_PLANT_PHASES = ("pressure", "resolution")
RETAKE_PRE_PLANT_PHASES = ("site_loss", "retake_setup")
RETAKE_POST_PLANT_PHASES = ("retake_attempt", "retake_resolution")
ECO_FORCE_STRICT_RATIO = 1.5
ECO_FORCE_MILD_RATIO = 1.2
FAMILY_ASSIGNMENT_PRIORITY = ("retake", "clutch", "execute", "eco_force")


def _policy_profile_priority(profile: str) -> tuple[str, ...]:
    if profile == "eco_bias_v1":
        return ("eco_force", "execute", "retake", "clutch")
    if profile == "execute_bias_v1":
        return ("execute", "eco_force", "retake", "clutch")
    return ("execute", "eco_force", "retake", "clutch")


def _normalize_policy_profile(profile: str) -> str:
    out = str(profile or "balanced_v1")
    if out not in POLICY_PROFILES:
        raise ValueError(f"unknown policy profile: {out}")
    return out


def _regime_from_index(idx: int) -> str:
    return REGIMES[idx % len(REGIMES)]


def _allowed_policy_families_for_round(round_idx: int, ticks_per_round: int) -> tuple[str, ...]:
    """Allowed metadata families for a round shape; excludes obviously incoherent labels."""
    allow_retake = (ticks_per_round >= 3) and (round_idx % 2 == 0)
    allow_clutch = ticks_per_round >= 4
    out = ["execute", "eco_force"]
    if allow_clutch:
        out.append("clutch")
    if allow_retake:
        out.append("retake")
    return tuple(out)


def _target_family_quotas(*, policy_profile: str, rounds: int, ticks_per_round: int) -> dict[str, int]:
    profile = _normalize_policy_profile(policy_profile)
    n_rounds = max(1, int(rounds))
    ticks = max(2, int(ticks_per_round))
    if n_rounds == 32 and ticks >= 4:
        return {family: int(POLICY_PROFILE_QUOTAS_32[profile][family]) for family in POLICY_FAMILIES}
    proportions = POLICY_PROFILE_PROPORTIONS[profile]
    quotas = {family: int(math.floor(proportions[family] * n_rounds)) for family in POLICY_FAMILIES}
    remainder = n_rounds - sum(quotas.values())
    if remainder > 0:
        ranked = sorted(
            POLICY_FAMILIES,
            key=lambda fam: (proportions[fam] * n_rounds) - float(quotas[fam]),
            reverse=True,
        )
        for idx in range(remainder):
            quotas[ranked[idx % len(ranked)]] += 1
    return quotas


def _effective_family_quotas(
    *,
    target_quotas: dict[str, int],
    rounds: int,
    ticks_per_round: int,
    policy_profile: str,
) -> tuple[dict[str, int], bool, str]:
    n_rounds = max(1, int(rounds))
    ticks = max(2, int(ticks_per_round))
    profile = _normalize_policy_profile(policy_profile)
    capacities = {
        family: sum(
            1
            for round_idx in range(n_rounds)
            if family in _allowed_policy_families_for_round(round_idx, ticks)
        )
        for family in POLICY_FAMILIES
    }
    effective = {family: int(max(0, target_quotas.get(family, 0))) for family in POLICY_FAMILIES}
    reasons: list[str] = []
    overflow = 0
    for family in POLICY_FAMILIES:
        cap = capacities[family]
        if effective[family] > cap:
            clipped = effective[family] - cap
            overflow += clipped
            effective[family] = cap
            reasons.append(f"{family} reduced by {clipped} due to feasibility")
    priority = _policy_profile_priority(profile)
    while overflow > 0:
        moved = False
        for family in priority:
            cap_left = capacities[family] - effective[family]
            if cap_left <= 0:
                continue
            effective[family] += 1
            overflow -= 1
            moved = True
            if overflow <= 0:
                break
        if not moved:
            raise ValueError("unable to redistribute infeasible policy quota")
    if sum(effective.values()) != n_rounds:
        raise ValueError("effective quotas must sum to rounds")
    applied = bool(reasons)
    reason = "; ".join(reasons) if reasons else ""
    return effective, applied, reason


def _select_policy_family_sequence(
    *,
    rng: random.Random,
    rounds: int,
    ticks_per_round: int,
    policy_profile: str,
) -> tuple[list[str], dict[str, int], dict[str, int], bool, str]:
    """
    Deterministically select one policy family per round from seeded RNG.

    Guarantees coverage for bounded runs when feasible:
    - if rounds >= len(POLICY_FAMILIES), tries to place each family at least once
      using coherence-constrained candidate rounds.
    """
    n_rounds = max(1, int(rounds))
    ticks = max(2, int(ticks_per_round))
    profile = _normalize_policy_profile(policy_profile)
    target_quotas = _target_family_quotas(
        policy_profile=profile,
        rounds=n_rounds,
        ticks_per_round=ticks,
    )
    effective_quotas, feasibility_adjustment_applied, feasibility_adjustment_reason = _effective_family_quotas(
        target_quotas=target_quotas,
        rounds=n_rounds,
        ticks_per_round=ticks,
        policy_profile=profile,
    )
    families: list[str | None] = [None] * n_rounds
    unassigned = set(range(n_rounds))
    for family in FAMILY_ASSIGNMENT_PRIORITY:
        quota = int(effective_quotas.get(family, 0))
        if quota <= 0:
            continue
        candidates = [
            idx
            for idx in sorted(unassigned)
            if family in _allowed_policy_families_for_round(idx, ticks)
        ]
        if len(candidates) < quota:
            raise ValueError(f"not enough candidate rounds for family={family} quota={quota}")
        chosen = rng.sample(candidates, quota) if quota < len(candidates) else candidates
        for idx in chosen:
            families[idx] = family
            unassigned.remove(idx)
    if unassigned:
        raise ValueError("family assignment left unassigned rounds")
    out = [str(fam) for fam in families]
    realized = {family: sum(1 for fam in out if fam == family) for family in POLICY_FAMILIES}
    if realized != effective_quotas:
        raise ValueError("realized distribution does not match effective quotas")
    return out, target_quotas, effective_quotas, feasibility_adjustment_applied, feasibility_adjustment_reason


def _policy_phase_for_tick(
    *,
    family: str,
    tick_idx: int,
    ticks_per_round: int,
) -> str:
    """Monotonic finite-state phase label for metadata-only Stage 1."""
    phases = POLICY_PHASE_ORDER.get(family, POLICY_PHASE_ORDER["execute"])
    total_ticks = max(2, int(ticks_per_round))
    bucket = min(len(phases) - 1, int((max(0, int(tick_idx)) * len(phases)) / total_ticks))
    return str(phases[bucket])


def _shape_planted_state_for_policy(*, family: str, phase: str, base_planted: bool) -> bool:
    """
    Stage 2A bounded policy shaping:
    - execute: setup/commit pre-plant; pressure/resolution post-plant.
    - retake: site_loss/retake_setup pre-plant; retake_attempt/resolution post-plant.
    - clutch/eco_force: metadata-only (no behavior mutation in Stage 2A).
    """
    if family == "execute":
        if phase in EXECUTE_PRE_PLANT_PHASES:
            return False
        if phase in EXECUTE_POST_PLANT_PHASES:
            return True
    if family == "retake":
        if phase in RETAKE_PRE_PLANT_PHASES:
            return False
        if phase in RETAKE_POST_PLANT_PHASES:
            return True
    return bool(base_planted)


def _shape_alive_counts_for_policy(
    *,
    family: str,
    phase: str,
    winner_team_one: bool,
    base_alive_one: int,
    base_alive_two: int,
) -> tuple[int, int]:
    """
    Stage 2B clutch shaping:
    - isolation: both teams start at >=2 alive.
    - duel_setup: no increase from isolation baseline.
    - duel_resolution + terminal: low-man condition must exist and persist.
    """
    if family != "clutch":
        return base_alive_one, base_alive_two
    if phase == "isolation":
        return 3, 3
    if phase == "duel_setup":
        return (3, 2) if winner_team_one else (2, 3)
    if phase in ("duel_resolution", "terminal"):
        return (2, 1) if winner_team_one else (1, 2)
    return base_alive_one, base_alive_two


def _team_state_totals(states: list[dict[str, Any]]) -> tuple[int, int]:
    cash_total = sum(max(0, int(row.get("balance", 0) or 0)) for row in states)
    loadout_total = sum(max(0, int(row.get("equipment_value", 0) or 0)) for row in states)
    return cash_total, loadout_total


def _team_totals_from_payload(row: dict[str, Any], team_key: str) -> tuple[int, int]:
    states = ((row.get(team_key) or {}).get("player_states") or [])
    return _team_state_totals(states)


def _eco_force_ratios_from_payload(row: dict[str, Any]) -> tuple[float, float]:
    cash_a, load_a = _team_totals_from_payload(row, "team_one")
    cash_b, load_b = _team_totals_from_payload(row, "team_two")
    cash_ratio = max(cash_a, cash_b) / max(1, min(cash_a, cash_b))
    load_ratio = max(load_a, load_b) / max(1, min(load_a, load_b))
    return cash_ratio, load_ratio


def _eco_force_winner_side_by_cash(row: dict[str, Any]) -> str:
    cash_a, _ = _team_totals_from_payload(row, "team_one")
    cash_b, _ = _team_totals_from_payload(row, "team_two")
    if cash_a == cash_b:
        return "tie"
    return "team_one" if cash_a > cash_b else "team_two"


def _policy_round_rows(payloads: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in payloads:
        rn = int(row.get("round_number", 0))
        out.setdefault(rn, []).append(row)
    return out


def _compute_policy_integrity_counters(payloads: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "one_family_per_round_violations": 0,
        "family_immutability_violations": 0,
        "phase_monotonicity_violations": 0,
        "intent_family_mismatch_violations": 0,
        "execute_retake_contract_violations": 0,
        "clutch_contract_violations": 0,
        "eco_force_contract_violations": 0,
    }
    by_round = _policy_round_rows(payloads)
    for rows in by_round.values():
        if not rows:
            counters["one_family_per_round_violations"] += 1
            continue
        families = [str(row.get("synthetic_policy_family", "")) for row in rows]
        unique_families = {family for family in families if family}
        if len(unique_families) != 1:
            counters["one_family_per_round_violations"] += 1
        if len(unique_families) > 1:
            counters["family_immutability_violations"] += 1
            continue
        family = families[0]
        expected_intent = POLICY_ROUND_INTENTS.get(family, "")
        if any(str(row.get("synthetic_policy_round_intent", "")) != expected_intent for row in rows):
            counters["intent_family_mismatch_violations"] += 1
        allowed_phases = POLICY_PHASE_ORDER.get(family, POLICY_PHASE_ORDER["execute"])
        rank = {phase: idx for idx, phase in enumerate(allowed_phases)}
        phases = [str(row.get("synthetic_policy_phase", "")) for row in rows]
        phase_ranks: list[int] = []
        phase_invalid = False
        for phase in phases:
            if phase not in rank:
                phase_invalid = True
                break
            phase_ranks.append(rank[phase])
        if phase_invalid or phase_ranks != sorted(phase_ranks):
            counters["phase_monotonicity_violations"] += 1
        if family == "execute":
            violated = False
            planted_flags = [bool(row.get("is_bomb_planted")) for row in rows]
            for row in rows:
                phase = str(row.get("synthetic_policy_phase", ""))
                planted = bool(row.get("is_bomb_planted"))
                if phase in EXECUTE_PRE_PLANT_PHASES and planted:
                    violated = True
                if phase in EXECUTE_POST_PLANT_PHASES and not planted:
                    violated = True
            seen_true = False
            for flag in planted_flags:
                if flag:
                    seen_true = True
                if seen_true and not flag:
                    violated = True
            if violated:
                counters["execute_retake_contract_violations"] += 1
        elif family == "retake":
            violated = False
            for row in rows:
                phase = str(row.get("synthetic_policy_phase", ""))
                planted = bool(row.get("is_bomb_planted"))
                if phase in RETAKE_PRE_PLANT_PHASES and planted:
                    violated = True
                if phase in RETAKE_POST_PLANT_PHASES and not planted:
                    violated = True
            seen_attempt = False
            for row in rows:
                phase = str(row.get("synthetic_policy_phase", ""))
                planted = bool(row.get("is_bomb_planted"))
                if phase == "retake_attempt":
                    seen_attempt = True
                if seen_attempt and not planted:
                    violated = True
            if violated:
                counters["execute_retake_contract_violations"] += 1
        elif family == "clutch":
            violated = False
            alive_series: list[tuple[int, int]] = []
            for row in rows:
                states_a = ((row.get("team_one") or {}).get("player_states") or [])
                states_b = ((row.get("team_two") or {}).get("player_states") or [])
                alive_a = sum(1 for p in states_a if p.get("is_alive") is True)
                alive_b = sum(1 for p in states_b if p.get("is_alive") is True)
                alive_series.append((alive_a, alive_b))
                if str(row.get("synthetic_policy_phase", "")) == "isolation":
                    if alive_a < 2 or alive_b < 2:
                        violated = True
            for idx in range(1, len(alive_series)):
                prev_a, prev_b = alive_series[idx - 1]
                cur_a, cur_b = alive_series[idx]
                if cur_a > prev_a or cur_b > prev_b:
                    violated = True
            phase_to_alive = {
                str(row.get("synthetic_policy_phase", "")): alive_series[idx]
                for idx, row in enumerate(rows)
            }
            duel_resolution = phase_to_alive.get("duel_resolution")
            terminal = phase_to_alive.get("terminal")
            if duel_resolution is None or terminal is None:
                violated = True
            else:
                if min(duel_resolution) > 2:
                    violated = True
                if min(terminal) > 2:
                    violated = True
                if terminal[0] > duel_resolution[0] or terminal[1] > duel_resolution[1]:
                    violated = True
            if violated:
                counters["clutch_contract_violations"] += 1
        elif family == "eco_force":
            violated = False
            phase_map = {str(row.get("synthetic_policy_phase", "")): row for row in rows}
            setup = phase_map.get("economy_setup")
            trade = phase_map.get("trade_or_save")
            if setup is None or trade is None:
                violated = True
            else:
                setup_cash_ratio, setup_load_ratio = _eco_force_ratios_from_payload(setup)
                trade_cash_ratio, trade_load_ratio = _eco_force_ratios_from_payload(trade)
                if not ((setup_cash_ratio >= ECO_FORCE_STRICT_RATIO) or (setup_load_ratio >= ECO_FORCE_STRICT_RATIO)):
                    violated = True
                if not ((trade_cash_ratio >= ECO_FORCE_MILD_RATIO) or (trade_load_ratio >= ECO_FORCE_MILD_RATIO)):
                    violated = True
                contact = phase_map.get("contact")
                if contact is not None:
                    contact_cash_ratio, contact_load_ratio = _eco_force_ratios_from_payload(contact)
                    if not (
                        (contact_cash_ratio >= ECO_FORCE_STRICT_RATIO)
                        or (contact_load_ratio >= ECO_FORCE_STRICT_RATIO)
                    ):
                        violated = True
                setup_winner = _eco_force_winner_side_by_cash(setup)
                if setup_winner == "tie":
                    violated = True
                contact = phase_map.get("contact")
                if contact is not None:
                    contact_winner = _eco_force_winner_side_by_cash(contact)
                    if contact_winner != setup_winner:
                        violated = True
                terminal = phase_map.get("terminal")
                if terminal is not None:
                    terminal_winner = _eco_force_winner_side_by_cash(terminal)
                    if terminal_winner not in (setup_winner, "tie"):
                        violated = True
            if violated:
                counters["eco_force_contract_violations"] += 1
    return counters


def summarize_policy_distribution(
    *,
    payloads: list[dict[str, Any]],
    seed: int,
    rounds: int,
    ticks_per_round: int,
    policy_profile: str,
    target_family_quotas: dict[str, int],
    effective_family_quotas: dict[str, int],
    feasibility_adjustment_applied: bool,
    feasibility_adjustment_reason: str,
) -> dict[str, Any]:
    by_round = _policy_round_rows(payloads)
    family_sequence = [str(by_round[rn][0].get("synthetic_policy_family", "")) for rn in sorted(by_round)]
    realized_family_counts = {
        family: int(sum(1 for fam in family_sequence if fam == family))
        for family in POLICY_FAMILIES
    }
    n_rounds = max(1, int(rounds))
    realized_family_rates = {
        family: float(realized_family_counts[family]) / float(n_rounds)
        for family in POLICY_FAMILIES
    }
    quota_delta = {
        family: int(realized_family_counts[family] - int(effective_family_quotas.get(family, 0)))
        for family in POLICY_FAMILIES
    }
    summary: dict[str, Any] = {
        "seed": int(seed),
        "rounds": int(rounds),
        "ticks_per_round": int(ticks_per_round),
        "policy_profile": _normalize_policy_profile(policy_profile),
        "target_family_quotas": {family: int(target_family_quotas.get(family, 0)) for family in POLICY_FAMILIES},
        "effective_family_quotas": {family: int(effective_family_quotas.get(family, 0)) for family in POLICY_FAMILIES},
        "realized_family_counts": realized_family_counts,
        "realized_family_rates": realized_family_rates,
        "quota_delta": quota_delta,
        "family_sequence": family_sequence,
        "feasibility_adjustment_applied": bool(feasibility_adjustment_applied),
        "feasibility_adjustment_reason": str(feasibility_adjustment_reason),
    }
    summary.update(_compute_policy_integrity_counters(payloads))
    return summary


def _ensure_ratio_for_key(
    *,
    advantaged_states: list[dict[str, Any]],
    disadvantaged_states: list[dict[str, Any]],
    key: str,
    min_ratio: float,
) -> None:
    adv_total = sum(max(0, int(row.get(key, 0) or 0)) for row in advantaged_states)
    dis_total = sum(max(0, int(row.get(key, 0) or 0)) for row in disadvantaged_states)
    target = int(math.ceil(float(min_ratio) * max(1, dis_total)))
    if adv_total >= target:
        return
    if not advantaged_states:
        return
    if adv_total <= 0:
        base = max(1, target // len(advantaged_states))
        for row in advantaged_states:
            row[key] = base
    else:
        scale = float(target) / float(max(1, adv_total))
        for row in advantaged_states:
            cur = max(0, int(row.get(key, 0) or 0))
            row[key] = int(math.ceil(cur * scale))
    new_total = sum(max(0, int(row.get(key, 0) or 0)) for row in advantaged_states)
    if new_total < target:
        advantaged_states[0][key] = int(max(0, int(advantaged_states[0].get(key, 0) or 0)) + (target - new_total))


def _shape_eco_force_team_states(
    *,
    phase: str,
    team_one_states: list[dict[str, Any]],
    team_two_states: list[dict[str, Any]],
    advantaged_team_one: bool,
) -> None:
    """
    Stage 2B eco_force shaping:
    - economy_setup/contact: strict asymmetry (>=1.5 on cash/loadout ratios).
    - trade_or_save: mild asymmetry (>=1.2).
    - terminal: no inversion versus setup winner side (>=1.0).
    """
    if phase in ("economy_setup", "contact"):
        min_ratio = ECO_FORCE_STRICT_RATIO
    elif phase == "trade_or_save":
        min_ratio = ECO_FORCE_MILD_RATIO
    elif phase == "terminal":
        min_ratio = 1.0
    else:
        min_ratio = ECO_FORCE_MILD_RATIO
    advantaged_states = team_one_states if advantaged_team_one else team_two_states
    disadvantaged_states = team_two_states if advantaged_team_one else team_one_states
    _ensure_ratio_for_key(
        advantaged_states=advantaged_states,
        disadvantaged_states=disadvantaged_states,
        key="balance",
        min_ratio=min_ratio,
    )
    _ensure_ratio_for_key(
        advantaged_states=advantaged_states,
        disadvantaged_states=disadvantaged_states,
        key="equipment_value",
        min_ratio=min_ratio,
    )


def _loadout_range_for_regime(regime: str) -> tuple[int, int]:
    if regime == "eco":
        return (350, 900)
    if regime == "half":
        return (1300, 2600)
    return (3200, 5200)


def _cash_range_for_regime(regime: str) -> tuple[int, int]:
    if regime == "eco":
        return (300, 1800)
    if regime == "half":
        return (1800, 4200)
    return (3500, 8000)


def _alive_pattern_for_winner(winner_team_one: bool, tick_idx: int) -> tuple[int, int]:
    # Fixed trajectories ensure 5->4->3 coverage while preserving winner tilt late.
    if winner_team_one:
        seq = ((5, 5), (5, 4), (4, 3), (3, 2))
    else:
        seq = ((5, 5), (4, 5), (3, 4), (2, 3))
    return seq[min(max(tick_idx, 0), len(seq) - 1)]


def _hp_for_player(rng: random.Random, alive: bool, low_bias: float = 0.0) -> int:
    if not alive:
        return 0
    lo = int(max(25, 40 + low_bias))
    hi = int(min(100, 100 - low_bias))
    return int(rng.randint(lo, hi))


def _build_team_player_states(
    *,
    rng: random.Random,
    team_prefix: str,
    alive_count: int,
    regime: str,
    hp_low_bias: float = 0.0,
) -> list[dict[str, Any]]:
    """Build 5-player BO3-like state rows for one team."""
    lo_load, hi_load = _loadout_range_for_regime(regime)
    lo_cash, hi_cash = _cash_range_for_regime(regime)
    states: list[dict[str, Any]] = []
    for idx in range(5):
        is_alive = idx < max(0, min(5, int(alive_count)))
        equipment_value = int(rng.randint(lo_load, hi_load)) if is_alive else 0
        balance = int(rng.randint(lo_cash, hi_cash))
        states.append(
            {
                "nickname": f"{team_prefix}{idx + 1}",
                "is_alive": bool(is_alive),
                "balance": balance,
                "equipment_value": equipment_value,
                    "health": _hp_for_player(rng, bool(is_alive), low_bias=hp_low_bias),
                "armor": int(rng.randint(0, 100)) if is_alive else 0,
                "has_helmet": bool(is_alive and rng.random() < 0.55),
                "has_kevlar": bool(is_alive and rng.random() < 0.8),
            }
        )
    return states


def generate_synthetic_raw_replay(
    *,
    seed: int = 1337,
    rounds: int = 10,
    ticks_per_round: int = 4,
    policy_profile: str = "balanced_v1",
) -> list[dict[str, Any]]:
    """
    Generate raw BO3 snapshots with deterministic seeded RNG.

    Returns list of raw payload dicts (not wrapped entries).
    """
    rng = random.Random(int(seed))
    ticks = max(2, int(ticks_per_round))
    n_rounds = max(1, int(rounds))
    out: list[dict[str, Any]] = []
    policy_families_by_round, _, _, _, _ = _select_policy_family_sequence(
        rng=rng,
        rounds=n_rounds,
        ticks_per_round=ticks,
        policy_profile=policy_profile,
    )
    score_one = 0
    score_two = 0
    prev_round_winner_one: bool | None = None

    # Timer rails include early/mid/late pressure and near-zero states.
    timer_template_ms = [115_000, 80_000, 42_000, 9_000]

    for r in range(n_rounds):
        policy_family = policy_families_by_round[r]
        policy_round_intent = POLICY_ROUND_INTENTS.get(policy_family, "site_take_attempt")
        eco_force_advantaged_team_one = bool(rng.random() < 0.5)
        # Carryover swing pattern: winner previous round tends to keep stronger regime.
        if prev_round_winner_one is None:
            regime_one = _regime_from_index(r)
            regime_two = _regime_from_index(r + 1)
        elif prev_round_winner_one:
            regime_one = "full" if r % 2 == 0 else "half"
            regime_two = "half" if r % 2 == 0 else "eco"
        else:
            regime_one = "half" if r % 2 == 0 else "eco"
            regime_two = "full" if r % 2 == 0 else "half"

        # Deterministic alternating winner with small seeded perturbation.
        winner_one = (r % 2 == 0)
        if rng.random() < 0.15:
            winner_one = not winner_one

        for t in range(ticks):
            policy_phase = _policy_phase_for_tick(
                family=policy_family,
                tick_idx=t,
                ticks_per_round=ticks,
            )
            base_planted = bool((t >= 2) and (r % 2 == 0))
            planted = _shape_planted_state_for_policy(
                family=policy_family,
                phase=policy_phase,
                base_planted=base_planted,
            )
            timer_ms = timer_template_ms[min(t, len(timer_template_ms) - 1)]
            if planted:
                timer_ms = min(timer_ms, 35_000 if t == 2 else 9_000)
            alive_one, alive_two = _alive_pattern_for_winner(winner_one, min(t, 3))
            alive_one, alive_two = _shape_alive_counts_for_policy(
                family=policy_family,
                phase=policy_phase,
                winner_team_one=winner_one,
                base_alive_one=alive_one,
                base_alive_two=alive_two,
            )
            # Encourage late fragility pressure.
            hp_bias = 5.0 * t
            team_one_states = _build_team_player_states(
                rng=rng,
                team_prefix="a",
                alive_count=alive_one,
                regime=regime_one,
                hp_low_bias=hp_bias if winner_one else hp_bias + 5.0,
            )
            team_two_states = _build_team_player_states(
                rng=rng,
                team_prefix="b",
                alive_count=alive_two,
                regime=regime_two,
                hp_low_bias=hp_bias if not winner_one else hp_bias + 5.0,
            )
            if policy_family == "eco_force":
                _shape_eco_force_team_states(
                    phase=policy_phase,
                    team_one_states=team_one_states,
                    team_two_states=team_two_states,
                    advantaged_team_one=eco_force_advantaged_team_one,
                )
            payload: dict[str, Any] = {
                "team_one": {
                    "name": "Synthetic A",
                    "id": 1001,
                    "score": score_one,
                    "match_score": 0,
                    "side": "T" if (r % 2 == 0) else "CT",
                    "player_states": team_one_states,
                },
                "team_two": {
                    "name": "Synthetic B",
                    "id": 2002,
                    "score": score_two,
                    "match_score": 0,
                    "side": "CT" if (r % 2 == 0) else "T",
                    "player_states": team_two_states,
                },
                "bo_type": 3,
                "game_number": 1,
                "round_number": r + 1,
                "round_phase": "IN_PROGRESS",
                "round_time_remaining": int(timer_ms),
                "is_bomb_planted": bool(planted),
                # Synthetic diagnostics hints (ignored by core parser).
                "synthetic_regime_team_one": regime_one,
                "synthetic_regime_team_two": regime_two,
                "synthetic_target_winner_team_one": bool(winner_one),
                "synthetic_policy_family": policy_family,
                "synthetic_policy_phase": policy_phase,
                "synthetic_policy_round_intent": policy_round_intent,
            }
            out.append(payload)

        # Advance score once at round boundary.
        if winner_one:
            score_one += 1
        else:
            score_two += 1
        prev_round_winner_one = winner_one

    return out


def generate_synthetic_distribution_summary(
    *,
    seed: int = 1337,
    rounds: int = 10,
    ticks_per_round: int = 4,
    policy_profile: str = "balanced_v1",
) -> dict[str, Any]:
    profile = _normalize_policy_profile(policy_profile)
    n_rounds = max(1, int(rounds))
    ticks = max(2, int(ticks_per_round))
    target_quotas = _target_family_quotas(
        policy_profile=profile,
        rounds=n_rounds,
        ticks_per_round=ticks,
    )
    effective_quotas, feasibility_adjustment_applied, feasibility_adjustment_reason = _effective_family_quotas(
        target_quotas=target_quotas,
        rounds=n_rounds,
        ticks_per_round=ticks,
        policy_profile=profile,
    )
    payloads = generate_synthetic_raw_replay(
        seed=seed,
        rounds=n_rounds,
        ticks_per_round=ticks,
        policy_profile=profile,
    )
    return summarize_policy_distribution(
        payloads=payloads,
        seed=seed,
        rounds=n_rounds,
        ticks_per_round=ticks,
        policy_profile=profile,
        target_family_quotas=target_quotas,
        effective_family_quotas=effective_quotas,
        feasibility_adjustment_applied=feasibility_adjustment_applied,
        feasibility_adjustment_reason=feasibility_adjustment_reason,
    )


def write_synthetic_raw_replay_jsonl(
    path: str | Path,
    *,
    seed: int = 1337,
    rounds: int = 10,
    ticks_per_round: int = 4,
    policy_profile: str = "balanced_v1",
) -> int:
    """Generate replay and write JSONL payload lines. Returns line count."""
    payloads = generate_synthetic_raw_replay(
        seed=seed,
        rounds=rounds,
        ticks_per_round=ticks_per_round,
        policy_profile=policy_profile,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in payloads:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")
    return len(payloads)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate seeded synthetic raw BO3 replay JSONL.")
    ap.add_argument("--out", default="tools/fixtures/replay_synthetic_seeded_v1.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for deterministic RNG")
    ap.add_argument("--rounds", type=int, default=10, help="Number of synthetic rounds")
    ap.add_argument("--ticks_per_round", type=int, default=4, help="Ticks per round")
    ap.add_argument(
        "--policy_profile",
        default="balanced_v1",
        choices=list(POLICY_PROFILES),
        help="Distribution profile for policy-family selection",
    )
    args = ap.parse_args()
    n = write_synthetic_raw_replay_jsonl(
        args.out,
        seed=args.seed,
        rounds=args.rounds,
        ticks_per_round=args.ticks_per_round,
        policy_profile=args.policy_profile,
    )
    print(f"wrote {n} synthetic payloads to {args.out}")


if __name__ == "__main__":
    main()
