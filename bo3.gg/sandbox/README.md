# BO3.gg CS2 API Sandbox

Standalone probe for live CS2 data via the `cs2api` Python package. **Do not modify files outside this folder.**

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python probe_bo3_live.py
```

## Outputs

- **raw_live_matches.json** – Full response from `get_live_matches()`.
- **raw_live_snapshot.json** – Full response from `get_live_match_snapshot(match_id)` for the first live match.

## Snapshot field mapping (confirmed)

| Concept        | Where in snapshot |
|----------------|--------------------|
| **Map score**  | `team_one.score`, `team_two.score` (current map). `match_fixture.team_one_score` / `team_two_score` (series, may be 0 in snapshot). |
| **Series score** | `team_one.match_score`, `team_two.match_score`. |
| **Current map**  | `map_name` (e.g. `"de_inferno"`), `game_number`. |
| **Team sides**   | `team_one.side`, `team_two.side` (`"CT"` / `"TERRORIST"`). Per-player: `team_one.player_states[].side`. |
| **Round state**  | `round_number`, `round_phase` (e.g. `"FINISHED"`, `"IN_PROGRESS"`), `round_time`, `round_time_remaining`, `is_bomb_planted`, `round_time_period`. |
| **Alive/dead**   | `team_one.player_states[].is_alive`, `team_two.player_states[].is_alive`. |
| **Economy/money** | Per player: `team_one/team_two.player_states[].balance`, `equipment_value`. Team-level: `economy_level`, `average_player_economy_level` (in snapshot team objects). |

Live matches list also has `live_updates` with `game_score`, `match_score`, `side`, `economy_level`, `equipment_value`, `map_name`, `round_phase`, `round_number` per team.
