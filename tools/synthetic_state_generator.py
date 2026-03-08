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
import random
from pathlib import Path
from typing import Any


REGIMES = ("eco", "half", "full")
POLICY_FAMILIES = ("execute", "retake", "clutch", "eco_force")
POLICY_ROUND_INTENTS: dict[str, str] = {
    "execute": "site_take_attempt",
    "retake": "postplant_retake_attempt",
    "clutch": "man_disadvantage_conversion",
    "eco_force": "economy_pressure_round",
}
POLICY_PHASE_ORDER: dict[str, tuple[str, ...]] = {
    "execute": ("setup", "commit", "pressure", "resolution"),
    "retake": ("site_loss", "retake_setup", "retake_attempt", "retake_resolution"),
    "clutch": ("isolation", "duel_setup", "duel_resolution", "terminal"),
    "eco_force": ("economy_setup", "contact", "trade_or_save", "terminal"),
}


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


def _select_policy_family_sequence(
    *,
    rng: random.Random,
    rounds: int,
    ticks_per_round: int,
) -> list[str]:
    """
    Deterministically select one policy family per round from seeded RNG.

    Guarantees coverage for bounded runs when feasible:
    - if rounds >= len(POLICY_FAMILIES), tries to place each family at least once
      using coherence-constrained candidate rounds.
    """
    n_rounds = max(1, int(rounds))
    families: list[str | None] = [None] * n_rounds
    if n_rounds >= len(POLICY_FAMILIES):
        mandatory = list(POLICY_FAMILIES)
        rng.shuffle(mandatory)
        for family in mandatory:
            candidates = [
                idx
                for idx in range(n_rounds)
                if families[idx] is None
                and family in _allowed_policy_families_for_round(idx, ticks_per_round)
            ]
            if candidates:
                families[rng.choice(candidates)] = family
    for idx in range(n_rounds):
        if families[idx] is not None:
            continue
        allowed = _allowed_policy_families_for_round(idx, ticks_per_round)
        families[idx] = rng.choice(list(allowed))
    return [str(fam) for fam in families]


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
) -> list[dict[str, Any]]:
    """
    Generate raw BO3 snapshots with deterministic seeded RNG.

    Returns list of raw payload dicts (not wrapped entries).
    """
    rng = random.Random(int(seed))
    ticks = max(2, int(ticks_per_round))
    n_rounds = max(1, int(rounds))
    out: list[dict[str, Any]] = []
    policy_families_by_round = _select_policy_family_sequence(
        rng=rng,
        rounds=n_rounds,
        ticks_per_round=ticks,
    )
    score_one = 0
    score_two = 0
    prev_round_winner_one: bool | None = None

    # Timer rails include early/mid/late pressure and near-zero states.
    timer_template_ms = [115_000, 80_000, 42_000, 9_000]

    for r in range(n_rounds):
        policy_family = policy_families_by_round[r]
        policy_round_intent = POLICY_ROUND_INTENTS.get(policy_family, "site_take_attempt")
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
            planted = bool((t >= 2) and (r % 2 == 0))
            timer_ms = timer_template_ms[min(t, len(timer_template_ms) - 1)]
            if planted:
                timer_ms = min(timer_ms, 35_000 if t == 2 else 9_000)
            alive_one, alive_two = _alive_pattern_for_winner(winner_one, min(t, 3))
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


def write_synthetic_raw_replay_jsonl(
    path: str | Path,
    *,
    seed: int = 1337,
    rounds: int = 10,
    ticks_per_round: int = 4,
) -> int:
    """Generate replay and write JSONL payload lines. Returns line count."""
    payloads = generate_synthetic_raw_replay(
        seed=seed,
        rounds=rounds,
        ticks_per_round=ticks_per_round,
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
    args = ap.parse_args()
    n = write_synthetic_raw_replay_jsonl(
        args.out,
        seed=args.seed,
        rounds=args.rounds,
        ticks_per_round=args.ticks_per_round,
    )
    print(f"wrote {n} synthetic payloads to {args.out}")


if __name__ == "__main__":
    main()
