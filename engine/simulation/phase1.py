"""Seeded simulation Phase 1 contract harness."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.resolve import resolve_p_hat
from engine.diagnostics.invariants import compute_corridor_invariants
from engine.models import Config, Frame, State

SIMULATION_PHASE1_SUMMARY_VERSION = "simulation_phase1_summary.v1"
CANONICAL_ENGINE_PATH = "compute_bounds>compute_rails>resolve_p_hat"
SIMULATION_PHASE1_FAMILIES = (
    "players_alive_trajectory",
    "loadout_regime_shift",
    "bomb_plant_countdown_transition",
    "near_zero_timer_pressure_ramp",
    "carryover_economy_transition",
)


def _bomb_phase(*, round_phase: str, is_bomb_planted: bool, round_number: int, timer_s: float | None) -> dict[str, Any]:
    return {
        "round_phase": round_phase,
        "is_bomb_planted": is_bomb_planted,
        "round_number": round_number,
        "phase_time_remaining_s": timer_s,
    }


def _make_frame(
    *,
    timestamp: float,
    scores: tuple[int, int],
    series_score: tuple[int, int],
    alive_counts: tuple[int, int],
    hp_totals: tuple[float, float],
    cash_totals: tuple[float, float],
    loadout_totals: tuple[float, float],
    armor_totals: tuple[float, float],
    round_time_remaining_s: float,
    round_phase: str,
    is_bomb_planted: bool,
    round_number: int,
    a_side: str,
) -> Frame:
    wealth_totals = (
        float(cash_totals[0]) + float(loadout_totals[0]),
        float(cash_totals[1]) + float(loadout_totals[1]),
    )
    return Frame(
        timestamp=timestamp,
        teams=("A", "B"),
        scores=scores,
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=wealth_totals,
        cash_totals=cash_totals,
        loadout_totals=loadout_totals,
        wealth_totals=wealth_totals,
        armor_totals=armor_totals,
        bomb_phase_time_remaining=_bomb_phase(
            round_phase=round_phase,
            is_bomb_planted=is_bomb_planted,
            round_number=round_number,
            timer_s=round_time_remaining_s,
        ),
        round_time_remaining_s=round_time_remaining_s,
        map_index=0,
        series_score=series_score,
        map_name="synthetic_cs2_phase1",
        series_fmt="bo3",
        a_side=a_side,
    )


def _build_phase1_trajectories(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prematch_bias = 0.46 + (0.08 * rng.random())
    base_series = (rng.randint(0, 1), rng.randint(0, 1))
    a_side = rng.choice(("T", "CT"))

    return [
        {
            "family": "players_alive_trajectory",
            "prematch_map": prematch_bias,
            "frames": [
                _make_frame(
                    timestamp=1.0,
                    scores=(6, 6),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(14500.0, 14300.0),
                    loadout_totals=(23500.0, 23200.0),
                    armor_totals=(500.0, 500.0),
                    round_time_remaining_s=78.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=13,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=2.0,
                    scores=(6, 6),
                    series_score=base_series,
                    alive_counts=(4, 5),
                    hp_totals=(355.0, 420.0),
                    cash_totals=(14500.0, 14300.0),
                    loadout_totals=(23500.0, 23200.0),
                    armor_totals=(430.0, 500.0),
                    round_time_remaining_s=51.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=13,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=3.0,
                    scores=(6, 6),
                    series_score=base_series,
                    alive_counts=(3, 4),
                    hp_totals=(240.0, 310.0),
                    cash_totals=(14500.0, 14300.0),
                    loadout_totals=(23500.0, 23200.0),
                    armor_totals=(250.0, 360.0),
                    round_time_remaining_s=29.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=13,
                    a_side=a_side,
                ),
            ],
        },
        {
            "family": "loadout_regime_shift",
            "prematch_map": prematch_bias,
            "frames": [
                _make_frame(
                    timestamp=4.0,
                    scores=(2, 2),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(1800.0, 2000.0),
                    loadout_totals=(5200.0, 5600.0),
                    armor_totals=(120.0, 140.0),
                    round_time_remaining_s=115.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=5,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=5.0,
                    scores=(2, 2),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(6400.0, 6100.0),
                    loadout_totals=(14200.0, 13800.0),
                    armor_totals=(360.0, 340.0),
                    round_time_remaining_s=112.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=5,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=6.0,
                    scores=(2, 2),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(10500.0, 10100.0),
                    loadout_totals=(24200.0, 23600.0),
                    armor_totals=(500.0, 500.0),
                    round_time_remaining_s=109.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=5,
                    a_side=a_side,
                ),
            ],
        },
        {
            "family": "bomb_plant_countdown_transition",
            "prematch_map": prematch_bias,
            "frames": [
                _make_frame(
                    timestamp=7.0,
                    scores=(8, 7),
                    series_score=base_series,
                    alive_counts=(3, 3),
                    hp_totals=(255.0, 250.0),
                    cash_totals=(6200.0, 6000.0),
                    loadout_totals=(18800.0, 18400.0),
                    armor_totals=(310.0, 300.0),
                    round_time_remaining_s=34.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=True,
                    round_number=16,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=8.0,
                    scores=(8, 7),
                    series_score=base_series,
                    alive_counts=(3, 2),
                    hp_totals=(220.0, 130.0),
                    cash_totals=(6200.0, 6000.0),
                    loadout_totals=(18800.0, 18400.0),
                    armor_totals=(260.0, 180.0),
                    round_time_remaining_s=18.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=True,
                    round_number=16,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=9.0,
                    scores=(8, 7),
                    series_score=base_series,
                    alive_counts=(2, 2),
                    hp_totals=(140.0, 110.0),
                    cash_totals=(6200.0, 6000.0),
                    loadout_totals=(18800.0, 18400.0),
                    armor_totals=(150.0, 120.0),
                    round_time_remaining_s=7.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=True,
                    round_number=16,
                    a_side=a_side,
                ),
            ],
        },
        {
            "family": "near_zero_timer_pressure_ramp",
            "prematch_map": prematch_bias,
            "frames": [
                _make_frame(
                    timestamp=10.0,
                    scores=(10, 10),
                    series_score=base_series,
                    alive_counts=(4, 4),
                    hp_totals=(310.0, 290.0),
                    cash_totals=(7800.0, 7900.0),
                    loadout_totals=(20100.0, 19800.0),
                    armor_totals=(400.0, 390.0),
                    round_time_remaining_s=9.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=21,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=11.0,
                    scores=(10, 10),
                    series_score=base_series,
                    alive_counts=(4, 3),
                    hp_totals=(250.0, 180.0),
                    cash_totals=(7800.0, 7900.0),
                    loadout_totals=(20100.0, 19800.0),
                    armor_totals=(320.0, 250.0),
                    round_time_remaining_s=4.0,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=21,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=12.0,
                    scores=(10, 10),
                    series_score=base_series,
                    alive_counts=(3, 2),
                    hp_totals=(170.0, 80.0),
                    cash_totals=(7800.0, 7900.0),
                    loadout_totals=(20100.0, 19800.0),
                    armor_totals=(180.0, 90.0),
                    round_time_remaining_s=1.5,
                    round_phase="IN_PROGRESS",
                    is_bomb_planted=False,
                    round_number=21,
                    a_side=a_side,
                ),
            ],
        },
        {
            "family": "carryover_economy_transition",
            "prematch_map": prematch_bias,
            "frames": [
                _make_frame(
                    timestamp=13.0,
                    scores=(9, 9),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(5400.0, 5500.0),
                    loadout_totals=(14800.0, 14900.0),
                    armor_totals=(330.0, 335.0),
                    round_time_remaining_s=115.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=19,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=14.0,
                    scores=(10, 9),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(9800.0, 3400.0),
                    loadout_totals=(23800.0, 12400.0),
                    armor_totals=(500.0, 240.0),
                    round_time_remaining_s=114.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=20,
                    a_side=a_side,
                ),
                _make_frame(
                    timestamp=15.0,
                    scores=(10, 10),
                    series_score=base_series,
                    alive_counts=(5, 5),
                    hp_totals=(500.0, 500.0),
                    cash_totals=(3600.0, 9600.0),
                    loadout_totals=(12600.0, 23600.0),
                    armor_totals=(250.0, 500.0),
                    round_time_remaining_s=114.0,
                    round_phase="BUY_TIME",
                    is_bomb_planted=False,
                    round_number=21,
                    a_side=a_side,
                ),
            ],
        },
    ]


def _build_config(prematch_map: float) -> Config:
    config = Config(
        source="REPLAY",
        contract_scope="map",
        series_fmt="bo3",
        prematch_map=float(prematch_map),
        invariant_diagnostics=True,
    )
    setattr(config, "contract_testing_mode", True)
    return config



def generate_phase1_summary(seed: int) -> dict[str, Any]:
    trajectories = _build_phase1_trajectories(seed)
    trajectory_family_counts = {item["family"]: 1 for item in trajectories}
    trajectory_tick_counts = {item["family"]: len(item["frames"]) for item in trajectories}

    p_hats: list[float] = []
    rail_lows: list[float] = []
    rail_highs: list[float] = []
    structural_violations_total = 0
    behavioral_violations_total = 0
    invariant_violations_total = 0
    total_ticks_evaluated = 0

    for trajectory in trajectories:
        config = _build_config(trajectory["prematch_map"])
        state = State(config=config, map_index=0, last_total_rounds=0, last_map_index=0)
        for frame in trajectory["frames"]:
            total_ticks_evaluated += 1
            bound_low, bound_high, _ = compute_bounds(frame, config, state)
            rail_low, rail_high, _ = compute_rails(frame, config, state, (bound_low, bound_high))
            p_hat, resolve_debug = resolve_p_hat(frame, config, state, (rail_low, rail_high))
            contract_diag = resolve_debug.get("contract_diagnostics") or {}
            corridor_diag = compute_corridor_invariants(
                series_low=bound_low,
                series_high=bound_high,
                map_low=rail_low,
                map_high=rail_high,
                p_hat=p_hat,
                testing_mode=True,
            )

            structural_violations_total += len(contract_diag.get("structural_violations") or [])
            structural_violations_total += len(corridor_diag.get("invariant_structural_violations") or [])
            behavioral_violations_total += len(contract_diag.get("behavioral_violations") or [])
            behavioral_violations_total += len(corridor_diag.get("invariant_behavioral_violations") or [])
            invariant_violations_total += len(corridor_diag.get("invariant_violations") or [])

            p_hats.append(float(p_hat))
            rail_lows.append(float(rail_low))
            rail_highs.append(float(rail_high))

            state.last_frame = frame
            state.last_total_rounds = int(frame.scores[0]) + int(frame.scores[1])
            state.last_series_score = frame.series_score
            state.last_map_index = frame.map_index

    return {
        "schema_version": SIMULATION_PHASE1_SUMMARY_VERSION,
        "seed": int(seed),
        "canonical_engine_path": CANONICAL_ENGINE_PATH,
        "trajectory_family_counts": trajectory_family_counts,
        "trajectory_tick_counts": trajectory_tick_counts,
        "total_ticks_evaluated": total_ticks_evaluated,
        "structural_violations_total": structural_violations_total,
        "behavioral_violations_total": behavioral_violations_total,
        "invariant_violations_total": invariant_violations_total,
        "p_hat_min": min(p_hats),
        "p_hat_max": max(p_hats),
        "p_hat_count": len(p_hats),
        "rail_low_min": min(rail_lows),
        "rail_low_max": max(rail_lows),
        "rail_high_min": min(rail_highs),
        "rail_high_max": max(rail_highs),
    }



def emit_phase1_summary(seed: int, output_path: str | Path | None = None) -> dict[str, Any]:
    summary = generate_phase1_summary(seed)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
