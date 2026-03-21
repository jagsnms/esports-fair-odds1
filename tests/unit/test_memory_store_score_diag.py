"""
Unit tests for score-space diagnostics (history_score_points.jsonl).
Verifies: new file created when HISTORY_SCORE_RECORD_ENABLED=true; schema has score_raw and term_contribs;
reconstruction score_raw Ã¢â€°Ë† base_intercept + sum(term_contribs); history_points.jsonl behavior unchanged.
"""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from engine.models import HistoryPoint

from backend.store.memory_store import (
    _make_score_diag_record,
    MemoryStore,
)


def test_make_score_diag_record_reconstruction() -> None:
    """Score diag v2: score_raw = contrib_sum + residual_contrib (exact reconstruction)."""
    base = 0.0
    term_alive = 0.02
    term_hp = -0.01
    term_loadout = 0.005
    term_bomb = 0.0
    term_cash = 0.0
    score_raw = term_alive + term_hp + term_loadout + term_bomb + term_cash
    point = HistoryPoint(
        time=1000.5,
        p_hat=0.52,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.4,
        rail_high=0.6,
        game_number=1,
        map_index=0,
        round_number=3,
        explain={
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.52, "clamp_reason": None},
            "score_raw": score_raw,
            "term_contribs": {
                "term_alive": term_alive,
                "term_hp": term_hp,
                "term_loadout": term_loadout,
                "term_bomb": term_bomb,
                "term_cash": term_cash,
            },
            "base_intercept": base,
            "p_unshaped": 0.51,
        },
    )
    rec = _make_score_diag_record(point)
    assert rec is not None
    assert rec["schema"] == "score_diag_v2"
    assert "score_raw" in rec
    assert "term_contribs" in rec
    assert "base_intercept" in rec
    assert "contrib_sum" in rec
    assert "residual_contrib" in rec
    score_raw_out = rec["score_raw"]
    contrib_sum = rec["contrib_sum"]
    residual_contrib = rec["residual_contrib"]
    assert abs(score_raw_out - (contrib_sum + residual_contrib)) < 1e-6, (
        f"score_raw={score_raw_out} != contrib_sum + residual_contrib = {contrib_sum + residual_contrib}"
    )
    assert rec["phase"] == "IN_PROGRESS"
    assert rec["p_hat_final"] == 0.52


def test_make_score_diag_record_skips_idle() -> None:
    """Idle phase ticks are not emitted."""
    point = HistoryPoint(
        time=0,
        p_hat=0.5,
        explain={
            "phase": "idle",
            "final": {"p_hat_final": 0.5, "clamp_reason": None},
            "score_raw": 0.0,
            "term_contribs": {},
            "base_intercept": 0.0,
        },
    )
    assert _make_score_diag_record(point) is None


def test_make_score_diag_record_v2_residual_makes_reconstruction_exact() -> None:
    """When score_raw != sum(term_contribs) (e.g. armor), residual_contrib captures the gap; reconstruction exact."""
    term_alive = 0.02
    term_hp = -0.01
    term_loadout = 0.0
    term_bomb = 0.0
    term_cash = 0.0
    contrib_sum = term_alive + term_hp + term_loadout + term_bomb + term_cash  # 0.01
    residual = 0.039  # e.g. armor or other missing term
    score_raw = contrib_sum + residual
    point = HistoryPoint(
        time=0,
        p_hat=0.5,
        bound_low=0.0,
        bound_high=1.0,
        rail_low=0.3,
            rail_high=0.7,
            match_id=444,
            explain={
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.5, "clamp_reason": None},
            "score_raw": score_raw,
            "term_contribs": {
                "term_alive": term_alive,
                "term_hp": term_hp,
                "term_loadout": term_loadout,
                "term_bomb": term_bomb,
                "term_cash": term_cash,
            },
            "base_intercept": 0.0,
            "p_unshaped": 0.52,
        },
    )
    rec = _make_score_diag_record(point)
    assert rec is not None
    assert rec["schema"] == "score_diag_v2"
    assert abs(rec["contrib_sum"] - contrib_sum) < 1e-9
    assert abs(rec["residual_contrib"] - residual) < 1e-9
    assert abs(rec["score_raw"] - (rec["contrib_sum"] + rec["residual_contrib"])) < 1e-6


def test_make_score_diag_record_includes_term_raw_term_coef() -> None:
    """When explain has term_raw and term_coef, record includes them; contrib Ã¢â€°Ë† raw * coef for linear terms."""
    # Synthetic: alive_raw=1.0, coef=0.035 -> term_alive=0.035
    term_raw = {"alive": 1.0, "hp": 0.1, "loadout": 0.5, "bomb": 0.0, "cash": 0.0}
    term_coef = {"alive": 0.035, "hp": 0.04, "loadout": 0.012, "bomb": 0.06, "cash": 0.0}
    term_alive = term_raw["alive"] * term_coef["alive"]
    term_hp = term_raw["hp"] * term_coef["hp"]
    term_loadout = term_raw["loadout"] * term_coef["loadout"]
    term_bomb = 0.0
    term_cash = 0.0
    score_raw = term_alive + term_hp + term_loadout + term_bomb + term_cash
    point = HistoryPoint(
        time=0,
        p_hat=0.5,
        bound_low=0.0,
        bound_high=1.0,
        rail_low=0.3,
            rail_high=0.7,
            match_id=444,
            explain={
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.5, "clamp_reason": None},
            "score_raw": score_raw,
            "term_contribs": {
                "term_alive": term_alive,
                "term_hp": term_hp,
                "term_loadout": term_loadout,
                "term_bomb": term_bomb,
                "term_cash": term_cash,
            },
            "base_intercept": 0.0,
            "term_raw": term_raw,
            "term_coef": term_coef,
            "p_unshaped": 0.53,
        },
    )
    rec = _make_score_diag_record(point)
    assert rec is not None
    assert "term_raw" in rec
    assert "term_coef" in rec
    assert rec["term_raw"]["alive"] == 1.0
    assert rec["term_coef"]["alive"] == 0.035
    # Linear check: contrib Ã¢â€°Ë† raw * coef
    assert abs(rec["term_contribs"]["term_alive"] - rec["term_raw"]["alive"] * rec["term_coef"]["alive"]) < 1e-6
    assert abs(rec["term_contribs"]["term_hp"] - rec["term_raw"]["hp"] * rec["term_coef"]["hp"]) < 1e-6
    assert abs(rec["term_contribs"]["term_loadout"] - rec["term_raw"]["loadout"] * rec["term_coef"]["loadout"]) < 1e-6


def test_make_score_diag_record_includes_team_identity_when_present() -> None:
    """When point has team identity fields, score_diag_v2 record includes them (witness CSV / inversion audit)."""
    point = HistoryPoint(
        time=0,
        p_hat=0.5,
        bound_low=0.0,
        bound_high=1.0,
        rail_low=0.35,
        rail_high=0.65,
        explain={
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.5, "clamp_reason": None},
            "score_raw": 0.0,
            "term_contribs": {"term_alive": 0.0, "term_hp": 0.0, "term_loadout": 0.0, "term_bomb": 0.0, "term_cash": 0.0},
            "base_intercept": 0.0,
            "p_unshaped": 0.5,
        },
        team_one_id=1001,
        team_two_id=1002,
        team_one_provider_id="bo3:1001",
        team_two_provider_id="bo3:1002",
        team_a_is_team_one=True,
        a_side="CT",
    )
    rec = _make_score_diag_record(point)
    assert rec is not None
    assert rec.get("team_one_id") == 1001
    assert rec.get("team_two_id") == 1002
    assert rec.get("team_one_provider_id") == "bo3:1001"
    assert rec.get("team_two_provider_id") == "bo3:1002"
    assert rec.get("team_a_is_team_one") is True
    assert rec.get("a_side") == "CT"


def test_make_score_diag_record_includes_match_identity_and_movement_fields() -> None:
    """Unified calibration rows keep match identity and the minimum movement diagnostics."""
    point = HistoryPoint(
        time=42.0,
        p_hat=0.61,
        bound_low=0.1,
        bound_high=0.9,
        rail_low=0.35,
        rail_high=0.75,
        match_id=777,
        game_number=2,
        map_index=1,
        round_number=14,
        explain={
            "phase": "IN_PROGRESS",
            "round_phase": "IN_PROGRESS",
            "q_intra_total": 0.73,
            "alive_counts": (4, 2),
            "hp_totals": (287.0, 120.0),
            "loadout_totals": (15400.0, 9800.0),
            "target_p_hat": 0.69,
            "p_hat_prev": 0.58,
            "movement_confidence": 0.25,
            "expected_p_hat_after_movement": 0.6075,
            "movement_gap_abs": 0.0025,
            "final": {"p_hat_final": 0.61, "clamp_reason": None},
            "score_raw": 0.13,
            "term_contribs": {
                "term_alive": 0.06,
                "term_hp": 0.04,
                "term_loadout": 0.03,
                "term_bomb": 0.0,
                "term_cash": 0.0,
            },
            "base_intercept": 0.0,
            "p_unshaped": 0.73,
        },
    )
    rec = _make_score_diag_record(point)
    assert rec is not None
    assert rec["match_id"] == 777
    assert rec["game_number"] == 2
    assert rec["map_index"] == 1
    assert rec["round_number"] == 14
    assert rec["round_phase"] == "IN_PROGRESS"
    assert rec["q_intra_total"] == 0.73
    assert rec["alive_counts"] == (4, 2)
    assert rec["hp_totals"] == (287.0, 120.0)
    assert rec["loadout_totals"] == (15400.0, 9800.0)
    assert rec["target_p_hat"] == 0.69
    assert rec["p_hat_prev"] == 0.58
    assert rec["p_hat_final"] == 0.61
    assert rec["movement_confidence"] == 0.25
    assert rec["expected_p_hat_after_movement"] == 0.6075
    assert rec["movement_gap_abs"] == 0.0025


def test_make_score_diag_record_skips_missing_score_raw() -> None:
    """Ticks without score_raw in explain are not emitted."""
    point = HistoryPoint(
        time=0,
        p_hat=0.5,
        explain={
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.5, "clamp_reason": None},
            "term_contribs": {},
        },
    )
    assert _make_score_diag_record(point) is None


@pytest.mark.asyncio
async def test_score_diag_file_created_when_enabled(tmp_path: object) -> None:
    """When HISTORY_SCORE_RECORD_ENABLED=true, append_point writes to the score diag file."""
    score_path = os.path.join(str(tmp_path), "history_score_points.jsonl")
    with (
        patch("backend.store.memory_store._HISTORY_SCORE_RECORD_ENABLED", True),
        patch("backend.store.memory_store._HISTORY_SCORE_RECORD_JSONL_PATH", score_path),
        patch("backend.store.memory_store._HISTORY_RECORD_ENABLED", False),
    ):
        store = MemoryStore(max_history=100)
        point = HistoryPoint(
            time=2000.0,
            p_hat=0.55,
            bound_low=0.2,
            bound_high=0.8,
            rail_low=0.45,
            rail_high=0.65,
            game_number=1,
            map_index=0,
            round_number=5,
            explain={
                "phase": "IN_PROGRESS",
                "final": {"p_hat_final": 0.55, "clamp_reason": None},
                "score_raw": 0.1,
                "term_contribs": {
                    "term_alive": 0.05,
                    "term_hp": 0.03,
                    "term_loadout": 0.02,
                    "term_bomb": 0.0,
                    "term_cash": 0.0,
                },
                "base_intercept": 0.0,
                "p_unshaped": 0.53,
            },
        )
        from engine.models import Derived, State
        from engine.config import Config
        state = State(config=Config(poll_interval_s=5.0))
        derived = Derived(p_hat=0.55, bound_low=0.2, bound_high=0.8, rail_low=0.45, rail_high=0.65, kappa=0.0, debug={})
        await store.append_point(point, state, derived)

    assert os.path.isfile(score_path)
    with open(score_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["schema"] == "score_diag_v2"
    assert rec["score_raw"] == 0.1
    assert rec["term_contribs"]["term_alive"] == 0.05
    assert "contrib_sum" in rec
    assert "residual_contrib" in rec
    assert abs(rec["score_raw"] - (rec["contrib_sum"] + rec["residual_contrib"])) < 1e-6


@pytest.mark.asyncio
async def test_history_points_schema_unchanged(tmp_path: object) -> None:
    """With both recorders enabled, history_points.jsonl still gets the same wire schema (unchanged)."""
    history_path = os.path.join(str(tmp_path), "history_points.jsonl")
    score_path = os.path.join(str(tmp_path), "history_score_points.jsonl")
    with (
        patch("backend.store.memory_store._HISTORY_RECORD_ENABLED", True),
        patch("backend.store.memory_store._HISTORY_RECORD_JSONL_PATH", history_path),
        patch("backend.store.memory_store._HISTORY_SCORE_RECORD_ENABLED", True),
        patch("backend.store.memory_store._HISTORY_SCORE_RECORD_JSONL_PATH", score_path),
    ):
        store = MemoryStore(max_history=100)
        point = HistoryPoint(
            time=3000.0,
            p_hat=0.5,
            bound_low=0.0,
            bound_high=1.0,
            rail_low=0.3,
            rail_high=0.7,
            match_id=444,
            explain={
                "phase": "IN_PROGRESS",
                "final": {"p_hat_final": 0.5, "clamp_reason": None},
                "score_raw": 0.0,
                "term_contribs": {"term_alive": 0.0, "term_hp": 0.0, "term_loadout": 0.0, "term_bomb": 0.0, "term_cash": 0.0},
                "base_intercept": 0.0,
            },
        )
        from engine.models import Derived, State
        from engine.config import Config
        state = State(config=Config(poll_interval_s=5.0))
        derived = Derived(p_hat=0.5, bound_low=0.0, bound_high=1.0, rail_low=0.3, rail_high=0.7, kappa=0.0, debug={})
        await store.append_point(point, state, derived)

    assert os.path.isfile(history_path)
    with open(history_path, "r", encoding="utf-8") as f:
        wire = json.loads(f.read())
    # Existing schema: t, p, lo, hi, rail_low, rail_high, explain, etc. (no score_diag-specific keys in wire)
    assert "t" in wire
    assert "p" in wire
    assert "lo" in wire
    assert "hi" in wire
    assert "explain" in wire
    assert wire["match_id"] == 444
    assert wire["explain"]["final"]["p_hat_final"] == 0.5
    # Score diag file is separate
    assert os.path.isfile(score_path)
    with open(score_path, "r", encoding="utf-8") as f:
        score_rec = json.loads(f.read())
    assert "score_raw" in score_rec
    assert "term_contribs" in score_rec
