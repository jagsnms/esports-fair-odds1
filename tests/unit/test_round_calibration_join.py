"""Unit test: synthetic tick + round_result join for round-level calibration."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing tools from repo root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.round_level_calibration import run_calibration


def test_round_result_join_synthetic_tick() -> None:
    """A non-event tick with map_index and round_number matching a round_result event joins and gets correct label."""
    # One round_result: (map_index=0, round_number=5), winner team A
    round_result_line = {
        "t": 1000.0,
        "p": 0.5,
        "event": {
            "event_type": "round_result",
            "map_index": 0,
            "round_number": 5,
            "round_winner_team_id": 1,
            "round_winner_is_team_a": True,
        },
    }
    # One non-event tick: same map_index and round_number, computed (phase IN_PROGRESS, no clamp)
    tick_line = {
        "t": 999.0,
        "p": 0.48,
        "map_index": 0,
        "round_number": 5,
        "explain": {
            "phase": "IN_PROGRESS",
            "final": {"p_hat_final": 0.48, "clamp_reason": None},
            "rails": {"corridor_width": 0.2},
            "q_terms": {"term_alive": 0.0},
            "micro_adj": {"alive_adj": 0.0, "hp_adj": 0.0, "econ_adj": 0.0},
        },
    }
    lines = [round_result_line, tick_line]
    summary, used_rows, bin_tables = run_calibration(lines)
    assert summary["n_round_result_events"] == 1
    assert summary["n_rounds_with_label"] == 1
    assert summary["n_used_ticks"] == 1, "tick should join on (map_index, round_number) and be included"
    assert len(used_rows) == 1
    assert used_rows[0]["y"] == 1, "label should be round_winner_is_team_a True -> y=1"
    assert used_rows[0]["map_index"] == 0 and used_rows[0]["round_number"] == 5
