"""
Unit tests for BO3 normalize first-class microstate fields: cash_totals, loadout_totals, armor_totals.
"""
from __future__ import annotations

import unittest

from engine.normalize.bo3_normalize import bo3_snapshot_to_frame


def test_player_states_present_cash_and_loadout_non_none() -> None:
    """When player_states present for both teams, cash_totals and loadout_totals are non-None and length-2 numeric tuples."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 5,
            "match_score": 1,
            "player_states": [
                {"is_alive": True, "balance": 1000, "equipment_value": 2000, "health": 100},
                {"is_alive": True, "balance": 500, "equipment_value": 1500, "health": 80},
                {"is_alive": False, "balance": 200, "equipment_value": 0, "health": 0},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 4,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 800, "equipment_value": 1000, "health": 100},
                {"is_alive": True, "balance": 400, "equipment_value": 500, "health": 60},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals is not None
    assert frame.loadout_totals is not None
    assert len(frame.cash_totals) == 2
    assert len(frame.loadout_totals) == 2
    # Team A (team_one): alive only -> 1000+500=1500 cash, 2000+1500=3500 loadout
    assert frame.cash_totals[0] == 1500.0
    assert frame.loadout_totals[0] == 3500.0
    # Team B (team_two): 800+400=1200 cash, 1000+500=1500 loadout
    assert frame.cash_totals[1] == 1200.0
    assert frame.loadout_totals[1] == 1500.0


def test_player_states_present_armor_when_provided() -> None:
    """When player_states present and armor is in payload, armor_totals is non-None and length-2."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 0, "equipment_value": 0, "health": 100, "armor": 100},
                {"is_alive": True, "balance": 0, "equipment_value": 0, "health": 100, "armor": 50},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 0, "equipment_value": 0, "health": 100, "armor": 80},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.armor_totals is not None
    assert len(frame.armor_totals) == 2
    assert frame.armor_totals[0] == 150.0  # 100+50
    assert frame.armor_totals[1] == 80.0


def test_player_states_missing_microstate_none() -> None:
    """When player_states missing for either team, cash_totals and loadout_totals remain None."""
    raw = {
        "team_one": {"name": "Team1", "score": 0, "match_score": 0},
        "team_two": {"name": "Team2", "score": 0, "match_score": 0},
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals is None
    assert frame.loadout_totals is None
    assert frame.armor_totals is None


def test_player_states_one_team_missing_microstate_none() -> None:
    """When only one team has player_states, microstate fields remain None."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [{"is_alive": True, "balance": 1000, "equipment_value": 500, "health": 100}],
        },
        "team_two": {"name": "Team2", "score": 0, "match_score": 0},
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals is None
    assert frame.loadout_totals is None
    assert frame.armor_totals is None


def test_cash_loadout_totals_unchanged() -> None:
    """cash_loadout_totals still populated as before (all players, cash+equipment combined)."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 1000, "equipment_value": 2000, "health": 100},
                {"is_alive": False, "balance": 500, "equipment_value": 0, "health": 0},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 800, "equipment_value": 500, "health": 100},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    # cash_loadout_totals = sum(balance) + sum(equipment_value) for ALL players
    # Team1: 1000+2000 + 500+0 = 3500; Team2: 800+500 = 1300
    assert frame.cash_loadout_totals[0] == 3500.0
    assert frame.cash_loadout_totals[1] == 1300.0
    # cash_totals = alive-only balance: Team1 1000, Team2 800
    assert frame.cash_totals == (1000.0, 800.0)
    assert frame.loadout_totals == (2000.0, 500.0)


def test_armor_missing_armor_totals_none() -> None:
    """When no player has armor field, armor_totals remains None."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [{"is_alive": True, "balance": 0, "equipment_value": 0, "health": 100}],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [{"is_alive": True, "balance": 0, "equipment_value": 0, "health": 100}],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals is not None
    assert frame.loadout_totals is not None
    assert frame.armor_totals is None


def test_player_rows_team_mapping_and_alive_ordering() -> None:
    """players_a/players_b map team_one/team_two correctly and order alive players first."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 3,
            "match_score": 1,
            "player_states": [
                {"nickname": "one_alive_1", "is_alive": True, "health": 90, "balance": 1000, "equipment_value": 2000},
                {"nickname": "one_dead", "is_alive": False, "health": 0, "balance": 0, "equipment_value": 0},
                {"nickname": "one_alive_2", "is_alive": True, "health": 80, "balance": 500, "equipment_value": 1500},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 5,
            "match_score": 2,
            "player_states": [
                {"nickname": "two_alive", "is_alive": True, "health": 100},
                {"nickname": "two_dead", "is_alive": False, "health": 0},
            ],
        },
    }
    frame_a = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    names_a = [p.name for p in frame_a.players_a]
    names_b = [p.name for p in frame_a.players_b]
    # Team A = team_one => players_a from team_one, alive first (one_alive_1, one_alive_2, one_dead)
    assert names_a == ["one_alive_1", "one_alive_2", "one_dead"]
    # Team B = team_two
    assert names_b == ["two_alive", "two_dead"]
    # Alive flags preserved
    assert [p.alive for p in frame_a.players_a] == [True, True, False]

    frame_b = bo3_snapshot_to_frame(raw, team_a_is_team_one=False)
    names_a_b = [p.name for p in frame_b.players_a]
    names_b_b = [p.name for p in frame_b.players_b]
    # When team_a_is_team_one=False, sides swap: Team A = team_two, Team B = team_one
    assert names_a_b == ["two_alive", "two_dead"]
    assert names_b_b == ["one_alive_1", "one_alive_2", "one_dead"]


class TestBo3NormalizeMicrostateFields(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_player_states_present_cash_and_loadout_non_none(self) -> None:
        test_player_states_present_cash_and_loadout_non_none()

    def test_player_states_present_armor_when_provided(self) -> None:
        test_player_states_present_armor_when_provided()

    def test_player_states_missing_microstate_none(self) -> None:
        test_player_states_missing_microstate_none()

    def test_player_states_one_team_missing_microstate_none(self) -> None:
        test_player_states_one_team_missing_microstate_none()

    def test_cash_loadout_totals_unchanged(self) -> None:
        test_cash_loadout_totals_unchanged()

    def test_armor_missing_armor_totals_none(self) -> None:
        test_armor_missing_armor_totals_none()

    def test_wealth_decomposition_ev_zeros_weapons_present(self) -> None:
        test_wealth_decomposition_ev_zeros_weapons_present()

    def test_wealth_decomposition_mixed_ev_and_estimate(self) -> None:
        test_wealth_decomposition_mixed_ev_and_estimate()

    def test_alive_only_filter_respected(self) -> None:
        test_alive_only_filter_respected()


def test_wealth_decomposition_ev_zeros_weapons_present() -> None:
    """equipment_value all zeros but weapons present -> loadout_totals > 0 from estimate, wealth_totals = cash + loadout."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 2000, "equipment_value": 0, "health": 100, "primary_weapon": "ak-47", "secondary_weapon": "glock-18", "has_kevlar": True, "has_helmet": True},
                {"is_alive": True, "balance": 1000, "equipment_value": 0, "health": 100, "primary_weapon": "m4a4", "secondary_weapon": "usp-s"},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 500, "equipment_value": 0, "health": 100, "primary_weapon": "awp", "secondary_weapon": "p250"},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals is not None
    assert frame.loadout_totals is not None
    assert frame.wealth_totals is not None
    # Team A: cash 2000+1000=3000; loadout estimated (ak 2700+glock 200+1000 vesthelm) + (m4a4 3100+usp 200) = 3900+3300 = 7200 approx
    assert frame.cash_totals[0] == 3000.0
    assert frame.loadout_totals[0] > 0
    assert frame.wealth_totals[0] == frame.cash_totals[0] + frame.loadout_totals[0]
    assert frame.wealth_totals[1] == frame.cash_totals[1] + frame.loadout_totals[1]
    assert frame.loadout_source == "weapon_est"


def test_wealth_decomposition_mixed_ev_and_estimate() -> None:
    """Mixed ev/estimate -> totals combine; loadout_source is 'mixed'."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 1000, "equipment_value": 4000, "health": 100},
                {"is_alive": True, "balance": 500, "equipment_value": 0, "health": 100, "primary_weapon": "ak-47", "secondary_weapon": "glock-18"},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 800, "equipment_value": 3000, "health": 100},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals == (1500.0, 800.0)
    assert frame.loadout_totals[0] > 4000.0  # 4000 ev + estimate for second player
    assert frame.loadout_totals[1] == 3000.0
    assert frame.wealth_totals is not None
    assert frame.wealth_totals[0] == frame.cash_totals[0] + frame.loadout_totals[0]
    assert frame.wealth_totals[1] == frame.cash_totals[1] + frame.loadout_totals[1]
    assert frame.loadout_source == "mixed"
    assert frame.loadout_ev_count_a == 1
    assert frame.loadout_est_count_a == 1
    assert frame.loadout_ev_count_b == 1
    assert frame.loadout_est_count_b == 0


def test_alive_only_filter_respected() -> None:
    """Dead players are excluded from cash_totals, loadout_totals, wealth_totals."""
    raw = {
        "team_one": {
            "name": "Team1",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": True, "balance": 3000, "equipment_value": 2000, "health": 100},
                {"is_alive": False, "balance": 5000, "equipment_value": 5000, "health": 0},
            ],
        },
        "team_two": {
            "name": "Team2",
            "score": 0,
            "match_score": 0,
            "player_states": [
                {"is_alive": False, "balance": 10000, "equipment_value": 10000, "health": 0},
            ],
        },
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.cash_totals == (3000.0, 0.0)
    assert frame.loadout_totals == (2000.0, 0.0)
    assert frame.wealth_totals == (5000.0, 0.0)
