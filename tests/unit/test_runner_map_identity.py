"""
Unit tests for per-map canonical Team A identity cache in Runner.
Ensures round_result labels and score ticks use the same Team A definition per (game_number, map_index).
"""
from __future__ import annotations

import pytest

from engine.models import Config, Frame

from backend.services.runner import (
    Runner,
    _team_identity_from_cache,
    _normalize_side_raw,
)


def test_normalize_side_raw() -> None:
    assert _normalize_side_raw("T") == "T"
    assert _normalize_side_raw("CT") == "CT"
    assert _normalize_side_raw(" t ") == "T"
    assert _normalize_side_raw(None) is None
    assert _normalize_side_raw("") is None
    assert _normalize_side_raw("x") is None


def test_team_identity_from_cache_empty() -> None:
    assert _team_identity_from_cache(None) == {}
    assert _team_identity_from_cache({}) == {}


def test_team_identity_from_cache_full() -> None:
    entry = {
        "team_one_id": 101,
        "team_two_id": 102,
        "team_one_provider_id": "p1",
        "team_two_provider_id": "p2",
        "team_a_is_team_one": True,
        "a_side": "CT",
    }
    out = _team_identity_from_cache(entry)
    assert out["team_one_id"] == 101
    assert out["team_two_id"] == 102
    assert out["team_one_provider_id"] == "p1"
    assert out["team_two_provider_id"] == "p2"
    assert out["team_a_is_team_one"] is True
    assert out["a_side"] == "CT"


def test_ensure_map_identity_from_raw_populates_cache() -> None:
    """First call for (game_number, map_index) populates cache; second call returns same entry."""
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_state = lambda: None
    store.get_current = lambda: None
    store.get_history = lambda **kw: []
    store.set_current = lambda s, d: None
    store.append_point = lambda p, s, d: None
    runner = Runner(store=store, broadcaster=MagicMock())

    raw = {
        "team_one": {"id": 1, "provider_id": "team_a_prov"},
        "team_two": {"id": 2, "provider_id": "team_b_prov"},
    }
    config = Config(source="BO3", match_id=99, team_a_is_team_one=True)

    entry1 = runner._ensure_map_identity_from_raw(raw, config, game_number=1, map_index=0)
    assert entry1 is not None
    assert entry1["team_one_id"] == 1
    assert entry1["team_two_id"] == 2
    assert entry1["team_one_provider_id"] == "team_a_prov"
    assert entry1["team_two_provider_id"] == "team_b_prov"
    assert entry1["team_a_is_team_one"] is True

    # Same (game, map) returns same cached entry
    entry2 = runner._ensure_map_identity_from_raw(raw, config, game_number=1, map_index=0)
    assert entry2 is entry1

    # Different (game, map) creates new entry
    entry3 = runner._ensure_map_identity_from_raw(raw, config, game_number=2, map_index=1)
    assert entry3 is not None
    assert (1, 0) in runner._map_identity_cache
    assert (2, 1) in runner._map_identity_cache


def test_ensure_map_identity_from_raw_uses_config_team_a_is_team_one() -> None:
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_state = lambda: None
    store.get_current = lambda: None
    store.get_history = lambda **kw: []
    store.set_current = lambda s, d: None
    store.append_point = lambda p, s, d: None
    runner = Runner(store=store, broadcaster=MagicMock())

    raw = {
        "team_one": {"id": 10},
        "team_two": {"id": 20},
    }
    config_team_a_one = Config(source="BO3", match_id=1, team_a_is_team_one=True)
    config_team_a_two = Config(source="BO3", match_id=1, team_a_is_team_one=False)

    entry1 = runner._ensure_map_identity_from_raw(raw, config_team_a_one, game_number=1, map_index=0)
    assert entry1["team_a_is_team_one"] is True

    runner._map_identity_cache.clear()
    entry2 = runner._ensure_map_identity_from_raw(raw, config_team_a_two, game_number=1, map_index=0)
    assert entry2["team_a_is_team_one"] is False


def test_ensure_map_identity_from_frame_populates_from_frame() -> None:
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_state = lambda: None
    store.get_current = lambda: None
    store.get_history = lambda **kw: []
    store.set_current = lambda s, d: None
    store.append_point = lambda p, s, d: None
    runner = Runner(store=store, broadcaster=MagicMock())

    frame = Frame(
        timestamp=0.0,
        teams=("Team1", "Team2"),
        scores=(5, 3),
        team_one_id=100,
        team_two_id=200,
        team_one_provider_id="prov_1",
        team_two_provider_id="prov_2",
        a_side="T",
    )
    config = Config(source="BO3", match_id=1, team_a_is_team_one=True)

    entry = runner._ensure_map_identity_from_frame(frame, config, game_number=1, map_index=0)
    assert entry is not None
    assert entry["team_one_id"] == 100
    assert entry["team_two_id"] == 200
    assert entry["team_one_provider_id"] == "prov_1"
    assert entry["team_two_provider_id"] == "prov_2"
    assert entry["team_a_is_team_one"] is True
    assert entry["a_side"] == "T"


def test_reset_outcome_trackers_clears_map_identity_cache() -> None:
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_state = lambda: None
    store.get_current = lambda: None
    store.get_history = lambda **kw: []
    store.set_current = lambda s, d: None
    store.append_point = lambda p, s, d: None
    runner = Runner(store=store, broadcaster=MagicMock())

    raw = {"team_one": {"id": 1}, "team_two": {"id": 2}}
    config = Config(source="BO3", match_id=1)
    runner._ensure_map_identity_from_raw(raw, config, game_number=1, map_index=0)
    assert len(runner._map_identity_cache) == 1

    runner._reset_outcome_trackers()
    assert len(runner._map_identity_cache) == 0


def test_ensure_map_identity_returns_none_when_game_or_map_none() -> None:
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_state = lambda: None
    store.get_current = lambda: None
    store.get_history = lambda **kw: []
    store.set_current = lambda s, d: None
    store.append_point = lambda p, s, d: None
    runner = Runner(store=store, broadcaster=MagicMock())

    raw = {"team_one": {"id": 1}, "team_two": {"id": 2}}
    config = Config(source="BO3", match_id=1)

    assert runner._ensure_map_identity_from_raw(raw, config, None, 0) is None
    assert runner._ensure_map_identity_from_raw(raw, config, 1, None) is None
    assert runner._ensure_map_identity_from_frame(None, config, 1, 0) is None
    assert runner._ensure_map_identity_from_frame(Frame(timestamp=0, teams=("A", "B"), scores=(0, 0)), config, None, 0) is None
