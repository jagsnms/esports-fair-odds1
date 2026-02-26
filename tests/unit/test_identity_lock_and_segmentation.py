"""
Unit tests for identity lock and transition segmentation (reducer).
Also: BO3 normalize a_side (Team A side T/CT) from snapshot with mapping.
"""
from __future__ import annotations

import unittest
from typing import Optional

from engine.models import Config, Frame, State
from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
from engine.state.reducer import reduce_state


def _frame(
    scores: tuple[int, int] = (0, 0),
    series_score: tuple[int, int] = (0, 0),
    map_index: int = 0,
    teams: tuple[str, str] = ("TeamA", "TeamB"),
    team_one_provider_id: Optional[str] = None,
    team_two_provider_id: Optional[str] = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=teams,
        scores=scores,
        series_score=series_score,
        series_fmt="bo3",
        map_index=map_index,
        team_one_provider_id=team_one_provider_id,
        team_two_provider_id=team_two_provider_id,
    )


def _config(lock_team_mapping: bool = False, team_a_is_team_one: bool = True) -> Config:
    c = Config()
    c.lock_team_mapping = lock_team_mapping
    c.team_a_is_team_one = team_a_is_team_one
    return c


def test_segment_id_increments_on_series_score_change() -> None:
    """Two frames with different series_score yield segment_id increment."""
    config = _config()
    state0 = State(config=config, segment_id=0, last_series_score=None, last_map_index=None)
    frame1 = _frame(series_score=(0, 0), map_index=0)
    state1 = reduce_state(state0, frame1, config)
    assert state1.segment_id == 0
    assert state1.last_series_score == (0, 0)
    assert state1.last_map_index == 0

    frame2 = _frame(series_score=(1, 0), map_index=0)  # map won by team A
    state2 = reduce_state(state1, frame2, config)
    assert state2.segment_id == 1, "segment_id should increment when series_score changes"
    assert state2.last_series_score == (1, 0)


def test_segment_id_increments_on_map_index_change() -> None:
    """Two frames with different map_index yield segment_id increment."""
    config = _config()
    state0 = State(config=config, segment_id=0, last_series_score=(0, 0), last_map_index=0)
    frame_next_map = _frame(series_score=(0, 0), map_index=1)
    state1 = reduce_state(state0, frame_next_map, config)
    assert state1.segment_id == 1, "segment_id should increment when map_index changes"
    assert state1.last_map_index == 1


def test_lock_team_mapping_keeps_mapping_unchanged() -> None:
    """When lock_team_mapping=True, reducer does not change team_mapping after initialization."""
    config = _config(lock_team_mapping=True, team_a_is_team_one=True)
    state0 = State(config=config, team_mapping={})
    frame1 = _frame(team_one_provider_id="p1", team_two_provider_id="p2")
    state1 = reduce_state(state0, frame1, config)
    assert state1.team_mapping.get("team_one_key") == "p1"
    assert state1.team_mapping.get("team_two_key") == "p2"
    assert state1.team_mapping.get("a_is_team_one") is True

    # Change config and frame so that unlocked would produce different mapping
    config2 = _config(lock_team_mapping=True, team_a_is_team_one=False)
    frame2 = _frame(team_one_provider_id="other1", team_two_provider_id="other2")
    state2 = reduce_state(state1, frame2, config2)
    # Should still have original mapping (lock keeps it)
    assert state2.team_mapping.get("team_one_key") == "p1"
    assert state2.team_mapping.get("team_two_key") == "p2"
    assert state2.team_mapping.get("a_is_team_one") is True


def test_unlocked_team_mapping_updates_with_config() -> None:
    """When lock_team_mapping=False, team_mapping reflects current config + frame keys."""
    config = _config(lock_team_mapping=False, team_a_is_team_one=True)
    state0 = State(config=config, team_mapping={})
    frame1 = _frame(team_one_provider_id="pid1", team_two_provider_id="pid2")
    state1 = reduce_state(state0, frame1, config)
    assert state1.team_mapping.get("team_one_key") == "pid1"
    assert state1.team_mapping.get("team_two_key") == "pid2"

    config2 = _config(lock_team_mapping=False, team_a_is_team_one=False)
    frame2 = _frame()  # no provider_id -> use names (TeamA, TeamB)
    state2 = reduce_state(state1, frame2, config2)
    # team_one = TeamB, team_two = TeamA when team_a_is_team_one is False
    assert state2.team_mapping.get("team_one_key") == "TeamB"
    assert state2.team_mapping.get("team_two_key") == "TeamA"
    assert state2.team_mapping.get("a_is_team_one") is False


def test_bo3_a_side_from_snapshot_team_a_is_team_one() -> None:
    """With team_a_is_team_one=True, frame.a_side is team_one.side normalized to T/CT."""
    raw = {
        "team_one": {"name": "Team1", "score": 5, "match_score": 1, "side": "T"},
        "team_two": {"name": "Team2", "score": 4, "match_score": 0, "side": "CT"},
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.a_side == "T"
    frame2 = bo3_snapshot_to_frame(
        {"team_one": {"name": "T1", "score": 0, "match_score": 0, "side": "CT"}, "team_two": {"name": "T2", "score": 0, "match_score": 0, "side": "T"}},
        team_a_is_team_one=True,
    )
    assert frame2.a_side == "CT"


def test_bo3_a_side_from_snapshot_team_a_is_team_two() -> None:
    """With team_a_is_team_one=False, frame.a_side is team_two.side (Team A = team_two)."""
    raw = {
        "team_one": {"name": "Team1", "score": 4, "match_score": 0, "side": "T"},
        "team_two": {"name": "Team2", "score": 5, "match_score": 1, "side": "CT"},
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=False)
    assert frame.a_side == "CT"
    raw2 = {
        "team_one": {"name": "T1", "score": 0, "match_score": 0, "side": "CT"},
        "team_two": {"name": "T2", "score": 0, "match_score": 0, "side": "T"},
    }
    frame2 = bo3_snapshot_to_frame(raw2, team_a_is_team_one=False)
    assert frame2.a_side == "T"


def test_bo3_a_side_stable_and_normalized() -> None:
    """a_side is stable for same mapping; lowercase/whitespace normalized to T/CT; unknown -> None."""
    raw_t = {"team_one": {"name": "A", "score": 0, "match_score": 0, "side": " t "}, "team_two": {"name": "B", "score": 0, "match_score": 0, "side": "ct"}}
    frame = bo3_snapshot_to_frame(raw_t, team_a_is_team_one=True)
    assert frame.a_side == "T"
    raw_no_side = {"team_one": {"name": "A", "score": 0, "match_score": 0}, "team_two": {"name": "B", "score": 0, "match_score": 0}}
    frame_none = bo3_snapshot_to_frame(raw_no_side, team_a_is_team_one=True)
    assert frame_none.a_side is None


class TestIdentityLockAndSegmentation(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_segment_id_increments_on_series_score_change(self) -> None:
        test_segment_id_increments_on_series_score_change()

    def test_segment_id_increments_on_map_index_change(self) -> None:
        test_segment_id_increments_on_map_index_change()

    def test_lock_team_mapping_keeps_mapping_unchanged(self) -> None:
        test_lock_team_mapping_keeps_mapping_unchanged()

    def test_unlocked_team_mapping_updates_with_config(self) -> None:
        test_unlocked_team_mapping_updates_with_config()

    def test_bo3_a_side_from_snapshot_team_a_is_team_one(self) -> None:
        test_bo3_a_side_from_snapshot_team_a_is_team_one()

    def test_bo3_a_side_from_snapshot_team_a_is_team_two(self) -> None:
        test_bo3_a_side_from_snapshot_team_a_is_team_two()

    def test_bo3_a_side_stable_and_normalized(self) -> None:
        test_bo3_a_side_stable_and_normalized()


if __name__ == "__main__":
    unittest.main()
