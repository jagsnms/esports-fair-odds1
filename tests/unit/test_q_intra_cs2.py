"""
Unit tests for engine.compute.q_intra_cs2 (CS2 intra-round advantage signal v1).
"""
from __future__ import annotations

from typing import Any

from engine.compute.q_intra_cs2 import compute_q_intra_cs2
from engine.models import Frame


def _frame(
    alive_counts: tuple[int, int] | None = None,
    hp_totals: tuple[float, float] | None = None,
    bomb_phase_time_remaining: Any = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=alive_counts if alive_counts is not None else (0, 0),
        hp_totals=hp_totals if hp_totals is not None else (0.0, 0.0),
        bomb_phase_time_remaining=bomb_phase_time_remaining,
    )


def test_q_in_zero_one() -> None:
    """q is always in [0, 1] for any frame."""
    cases = [
        _frame(alive_counts=(5, 5), hp_totals=(500.0, 500.0)),
        _frame(alive_counts=(5, 0), hp_totals=(500.0, 0.0)),
        _frame(alive_counts=(0, 5), hp_totals=(0.0, 500.0)),
        _frame(alive_counts=(3, 2), hp_totals=(300.0, 400.0)),
        _frame(),  # missing microstate -> 0.5
    ]
    for f in cases:
        q, debug = compute_q_intra_cs2(f)
        assert 0.0 <= q <= 1.0, f"q={q} out of range for {f}"
        assert debug["q_intra_round_win_a"] == q


def test_missing_inputs_q_half_and_reason() -> None:
    """When required microstate inputs are missing, return q=0.5 and reason in debug."""
    # Frame with no usable alive or hp: use defaults that are "missing" by having neither
    # actually present. Our impl treats (0,0) as present (we can read them). So we need
    # a frame where alive and hp are not usable. We don't have a way to pass "missing"
    # except by not having the attr. Frame() has alive_counts=(0,0), hp_totals=(0.0,0.0)
    # so both are "present" (tuple len 2). So we need to pass something that fails
    # the "present" check: e.g. alive_counts=() or a frame with no alive/hp.
    # In our implementation, alive_present is True if isinstance tuple/list and len>=2.
    # So (0,0) gives alive_present=True. To get missing we need to not have both.
    # We can't easily construct a Frame with alive_counts being a single element tuple
    # (len 1) - that would make alive_present False. Same for hp.
    # So: Frame with alive_counts=(1,) or (None, None)? Our code does len(alive) >= 2
    # so (1,) -> False. And int(None) would raise - we catch and return False, 0.0.
    # Actually for (None, None) we have len 2, then int(None) raises ValueError -> return False, 0.0.
    # So alive_present=False for (None, None). Same for hp (None, None). So both False -> missing_microstate.
    f = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(None, None),  # type: ignore[arg-type]
        hp_totals=(None, None),    # type: ignore[arg-type]
    )
    q, debug = compute_q_intra_cs2(f)
    assert q == 0.5
    assert debug.get("reason") == "missing_microstate"
    assert debug["inputs_present"]["alive"] is False
    assert debug["inputs_present"]["hp"] is False
    assert debug["used_econ"] is False
    assert debug["used_bomb_direction"] is False


def test_missing_inputs_single_tuple_length() -> None:
    """Frame with only one element in alive (invalid) and no hp -> missing_microstate."""
    # Build frame with alive not usable: (5,) has len 1
    f = _frame(alive_counts=(5,), hp_totals=(0.0, 0.0))  # type: ignore[arg-type]
    # Actually Frame expects tuple[int, int]. So we need a frame that our getters
    # treat as missing. In _get_alive_delta, len(alive) < 2 -> present=False.
    # So alive_counts=(1,) would need to be passed. Frame type says tuple[int,int]
    # but we can pass (1,) in Python. So create Frame with (5,) for alive and (0,0) for hp.
    # Then alive_present=False (len 1), hp_present=True. So we have hp -> we don't return
    # missing_microstate. So for true missing we need both absent. Easiest: Frame with
    # alive_counts that fail the len check. So (1,) and hp_totals that fail: (1.0,) or ().
    f = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(1,),  # type: ignore[arg-type]
        hp_totals=(1.0,),   # type: ignore[arg-type]
    )
    q, debug = compute_q_intra_cs2(f)
    assert q == 0.5
    assert debug.get("reason") == "missing_microstate"


def test_monotonic_alive_increases_q() -> None:
    """Increasing alive for A (or decreasing for B) increases q, others fixed."""
    base = _frame(alive_counts=(3, 3), hp_totals=(300.0, 300.0))
    q_base, _ = compute_q_intra_cs2(base)
    # More A alive
    more_a = _frame(alive_counts=(4, 3), hp_totals=(300.0, 300.0))
    q_more_a, _ = compute_q_intra_cs2(more_a)
    assert q_more_a > q_base
    # Fewer B alive
    fewer_b = _frame(alive_counts=(3, 2), hp_totals=(300.0, 300.0))
    q_fewer_b, _ = compute_q_intra_cs2(fewer_b)
    assert q_fewer_b > q_base


def test_monotonic_hp_increases_q() -> None:
    """Increasing HP for A (or decreasing for B) increases q, others fixed."""
    base = _frame(alive_counts=(3, 3), hp_totals=(300.0, 300.0))
    q_base, _ = compute_q_intra_cs2(base)
    more_hp_a = _frame(alive_counts=(3, 3), hp_totals=(400.0, 300.0))
    q_more, _ = compute_q_intra_cs2(more_hp_a)
    assert q_more > q_base
    less_hp_b = _frame(alive_counts=(3, 3), hp_totals=(300.0, 200.0))
    q_less_b, _ = compute_q_intra_cs2(less_hp_b)
    assert q_less_b > q_base


def test_time_gating_no_time_term_used_false() -> None:
    """When no numeric time remaining, time_term_used is False."""
    f = _frame(alive_counts=(3, 2), hp_totals=(300.0, 250.0), bomb_phase_time_remaining=None)
    q, debug = compute_q_intra_cs2(f)
    assert debug["time_term_used"] is False
    assert debug["t_remaining"] is None


def test_time_gating_with_plausible_time() -> None:
    """When round_time_remaining is numeric and in range, time_term_used is True."""
    f = _frame(
        alive_counts=(3, 2),
        hp_totals=(300.0, 250.0),
        bomb_phase_time_remaining={"round_time_remaining": 45.0},
    )
    q, debug = compute_q_intra_cs2(f)
    assert debug["time_term_used"] is True
    assert debug["t_remaining"] == 45.0


def test_debug_keys_present() -> None:
    """Debug dict contains all required keys."""
    f = _frame(alive_counts=(4, 2), hp_totals=(400.0, 200.0))
    q, debug = compute_q_intra_cs2(f)
    assert "q_intra_round_win_a" in debug
    assert "raw_score" in debug
    assert "alive_delta" in debug
    assert "hp_delta" in debug
    assert "bomb_term_used" in debug
    assert "time_term_used" in debug
    assert "t_remaining" in debug
    assert "used_econ" in debug
    assert debug["used_econ"] is False
    assert "used_bomb_direction" in debug
    assert debug["used_bomb_direction"] is False
    assert "inputs_present" in debug
    ip = debug["inputs_present"]
    assert "alive" in ip and "hp" in ip and "bomb" in ip and "time" in ip
