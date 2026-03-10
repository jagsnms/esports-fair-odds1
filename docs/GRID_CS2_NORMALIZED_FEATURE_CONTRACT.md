# CS2 Normalized Feature Contract

**Sources:** `docs/GRID_API_SPEC.md`, `docs/GRID_CS2_LIVE_STATE_FEATURE_INVENTORY.md`.  
**Purpose:** Define the normalized feature contract consumed by app logic from any live source (BO3 or GRID). GRID is the richer source; BO3 provides a sparse subset. Terminology aligns with existing app concepts: **round-state rails**, **mid-round** (intraround), **p_hat corridor**.

---

## 1) Context / reliability fields

Used to gate or damp probability updates (e.g. reject stale data, require `valid`, respect `paused`).

| Normalized field | Type | Required | Description |
|------------------|------|----------|-------------|
| `source` | string | **Required** | `"bo3"` \| `"grid"` — identifies provider. |
| `series_id` | string | **Required** | Unique series identifier. |
| `valid` | bool | **Required** | Provider marks data accurate; if false, gate or damp live model. |
| `updated_at` | string (ISO DateTime) | **Required** | Last update time; used for staleness and damping. |
| `version` | string | Optional | Schema/data version (e.g. GRID `version`). |
| `started` | bool | **Required** | Series has started. |
| `finished` | bool | **Required** | Series over; terminal. |
| `forfeited` | bool | Optional | Forfeit; special terminal handling. |
| `game_index` | int | **Required** | Current game (map) index in series (0-based or 1-based; document choice). |
| `game_started` | bool | **Required** | Current game has started. |
| `game_finished` | bool | **Required** | Current game over. |
| `game_paused` | bool | Optional | Game paused; optionally freeze or damp mid-round adjustment. |

---

## 2) Round-state rail fields (latched per round)

Update at round boundaries; drive **round-state rails** and resets for p_hat corridor. Used for map-level score and “if A wins round / if B wins round” bands.

| Normalized field | Type | Required | Description |
|------------------|------|----------|-------------|
| `rounds_a` | int | **Required** | Round score team A (current map). |
| `rounds_b` | int | **Required** | Round score team B (current map). |
| `map_name` | string | **Required** | Current map identifier (e.g. `mirage`, `ancient`). |
| `round_index` | int | Optional | Current round number in game (if available, e.g. segments). |
| `team_a_id` | string | **Required** | Team A identity (for side mapping). |
| `team_b_id` | string | **Required** | Team B identity. |
| `team_a_name` | string | Optional | Display name team A. |
| `team_b_name` | string | Optional | Display name team B. |
| `team_a_side` | string | Optional | T / CT for team A this half. |
| `team_b_side` | string | Optional | T / CT for team B. |
| `team_a_won_series` | bool | Optional | Team A won the series (terminal). |
| `team_b_won_series` | bool | Optional | Team B won the series. |
| `team_a_kills` | int | Optional | Kills this game/round for team A. |
| `team_b_kills` | int | Optional | Kills for team B. |
| `team_a_deaths` | int | Optional | Deaths for team A. |
| `team_b_deaths` | int | Optional | Deaths for team B. |
| `team_a_first_kill` | bool | Optional | First blood this round for team A. |
| `team_b_first_kill` | bool | Optional | First blood for team B. |

---

## 3) Mid-round dynamic fields (tick-updated)

Updated during the round; used for **mid-round** (intraround) adjustment **inside the p_hat corridor** (e.g. alive count, HP, economy, clock).

| Normalized field | Type | Required | Description |
|------------------|------|----------|-------------|
| `clock_seconds` | int | Optional | Game/round clock seconds (count-up or count-down per `clock_ticks_backwards`). |
| `clock_ticking` | bool | Optional | Clock is running (vs paused). |
| `clock_type` | string | Optional | e.g. `gameClock`, `bombTimer` — which phase. |
| `clock_ticks_backwards` | bool | Optional | True = countdown. |
| `team_a_money` | int | Optional | Team A total cash. |
| `team_b_money` | int | Optional | Team B total cash. |
| `team_a_loadout_value` | int | Optional | Team A equipment value. |
| `team_b_loadout_value` | int | Optional | Team B equipment value. |
| `team_a_net_worth` | int | Optional | Team A money + loadout (or equivalent). |
| `team_b_net_worth` | int | Optional | Team B money + loadout. |
| `players_a` | list of player object | Optional | Team A players (see below). |
| `players_b` | list of player object | Optional | Team B players. |

**Player object (per player in `players_a` / `players_b`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | **Required** (if players present) | Player ID. |
| `alive` | bool | **Required** | Alive this round. |
| `current_health` | int | Optional | Current HP. |
| `current_armor` | int | Optional | Current armor. |
| `money` | int | Optional | Cash. |
| `loadout_value` | int | Optional | Equipment value. |
| `net_worth` | int | Optional | money + loadout. |
| `kills` | int | Optional | Kills this round/game. |
| `deaths` | int | Optional | Deaths. |
| `name` | string | Optional | Display. |

---

## 4) Optional advanced fields (bomb / objective / structures)

For richer mid-round modeling (bomb phase, site state, objectives). All optional in contract.

| Normalized field | Type | Required | Description |
|------------------|------|----------|-------------|
| `bomb_timer_seconds` | int | Optional | Bomb countdown when planted (from NPC respawnClock or dedicated clock). |
| `bomb_planted` | bool | Optional | Bomb is planted. |
| `bomb_carrier_side` | string | Optional | T / CT carrying bomb (if applicable). |
| `objectives_a` | list of objective | Optional | Team A objectives (e.g. plant/defuse). |
| `objectives_b` | list of objective | Optional | Team B objectives. |
| `structures` | list of structure | Optional | Sites/structures (type, destroyed, etc.). |
| `npc_bomb` | object or null | Optional | Bomb NPC (type, position, respawnClock). |

**Objective:** `id`, `type` (e.g. plant, defuse), `completion_count`, `completed_first`.  
**Structure:** `id`, `type`, `side`, `team_id`, `destroyed`, `current_health`, `respawn_clock_seconds`.

---

## 5) Derived aggregates (computed from normalized fields)

App can compute these from the contract for p_hat and corridor logic.

| Derived field | Formula / rule | Use |
|---------------|----------------|--------|
| `alive_count_a` | Sum of `player.alive` true for team A | Mid-round; 1vX, clutch. |
| `alive_count_b` | Sum of `player.alive` true for team B | Same. |
| `hp_total_a` | Sum of `player.current_health` for alive A | Mid-round strength. |
| `hp_total_b` | Sum of `player.current_health` for alive B | Same. |
| `armor_total_a` | Sum of `player.current_armor` for A | Optional. |
| `armor_total_b` | Sum of `player.current_armor` for B | Optional. |
| `loadout_total_a` | Sum of `player.loadout_value` for A, or team-level `team_a_loadout_value` | Economy / strength. |
| `loadout_total_b` | Same for B | Same. |
| `completeness_score` | 0.0–1.0: fraction of required fields present and non-null (e.g. rails + clock + teams + players). | Gate or weight updates. |
| `staleness_seconds` | Now − `updated_at` (parsed). | Damp or reject if above threshold. |
| `has_players` | True if `players_a` and `players_b` are non-empty lists. | Chooses player-level vs team-level fallback. |
| `has_clock` | True if `clock_seconds` or `clock_type` present. | Chooses clock-based vs no-clock fallback. |

---

## 6) Required vs optional summary

- **Required (minimum viable contract):**  
  `source`, `series_id`, `valid`, `updated_at`, `started`, `finished`, `game_index`, `game_started`, `game_finished`, `rounds_a`, `rounds_b`, `map_name`, `team_a_id`, `team_b_id`.

- **Required for full round-state rails:**  
  Above + `team_a_side`, `team_b_side` (or infer from context), and optionally `team_a_kills`, `team_b_kills`, `team_a_deaths`, `team_b_deaths` when available.

- **Required for mid-round adjustment:**  
  Either player-level (`players_a` / `players_b` with `alive`, and ideally `current_health`) or team-level (`team_a_money`, `team_b_money`, `team_a_loadout_value`, `team_b_loadout_value`). Clock (`clock_seconds`, `clock_ticking`, `clock_type`) optional but recommended.

- **Optional everywhere:**  
  Names, first kill, objectives, structures, NPCs, bomb timer, damage stats, position.

---

## 7) Fallback behavior rules

- **If no players (`players_a` / `players_b` empty or missing):**  
  Use **team-level only**: `team_a_money`, `team_b_money`, `team_a_loadout_value`, `team_b_loadout_value`, `team_a_net_worth`, `team_b_net_worth`. Do not derive alive counts or HP from players; mid-round adjustment may be disabled or use economy-only.

- **If no clock (`clock_seconds` / `clock_type` missing):**  
  **Damp mid-round adjustment** (e.g. narrower corridor or lower weight for intraround updates). Do not assume round time; time-based features (e.g. bomb) unavailable.

- **If `valid` is false:**  
  **Gate** live model: do not apply live-state updates, or apply with strong damping.

- **If staleness exceeds threshold (e.g. `staleness_seconds` > 60):**  
  **Damp or reject** updates; optionally keep last valid state with a “stale” flag.

- **If `game_finished` true:**  
  **Latch** final round score; no further mid-round updates for that game.

- **If `rounds_a` / `rounds_b` change:**  
  **Reset** round-state rail and re-latch; mid-round state applies to the new round.

- **Completeness score below threshold:**  
  **Damp** probability update or use only the subset of features that are present (e.g. rails only, no mid-round).

---

## 8) Mapping notes

### BO3 → normalized (sparse)

- **Context:** `source = "bo3"`, `series_id` from BO3 session, `valid` = true if session live, `updated_at` from last scrape/time.
- **Rails:** `rounds_a`, `rounds_b`, `map_name`, `game_index`, `game_started`, `game_finished` from BO3 state; `team_a_id` / `team_b_id` from match/team IDs.
- **Mid-round:** Often only **alive counts** and optionally **HP totals** (if BO3 exposes them); rarely per-player. `team_a_money` / `team_b_money` or loadout may be absent; use 0 or omit. Clock often absent → damp mid-round.
- **Advanced:** Bomb/objectives/structures typically absent in BO3.

### GRID → normalized (rich)

- **Context:** `source = "grid"`, `series_id` = `seriesState.id`, `valid` = `seriesState.valid`, `updated_at` = `seriesState.updatedAt`, `version` = `seriesState.version`, `game_index` from `games[].sequenceNumber`, `game_started` / `game_finished` / `game_paused` from current game.
- **Rails:** `rounds_a` / `rounds_b` from `games[].teams[].score` (map to A/B by side or order); `map_name` from `games[].map.name`; `team_a_id` / `team_b_id` from `games[].teams[].id`; `team_a_side` / `team_b_side` from `games[].teams[].side`; kills/deaths from `games[].teams[].kills` / `deaths`; optional `round_index` from `games[].segments[].sequenceNumber`.
- **Mid-round:** `clock_*` from `games[].clock`; `team_*_money`, `team_*_loadout_value`, `team_*_net_worth` from `games[].teams[]`; `players_a` / `players_b` from `games[].teams[].players[]` with `alive`, `currentHealth`, `currentArmor`, `money`, `loadoutValue`, `netWorth`, `kills`, `deaths`.
- **Advanced:** Bomb from `games[].nonPlayerCharacters[]` (type = bomb), `respawnClock` → `bomb_timer_seconds`; objectives from `games[].teams[].objectives` / `players[].objectives`; structures from `games[].structures[]`.

---

## 9) V2 implementation subset (minimal robust upgrade)

Exact fields to wire first for a minimal, robust upgrade.

**Exact field list (wire first):**

- **Context:** `source`, `series_id`, `valid`, `updated_at`, `started`, `finished`, `game_index`, `game_started`, `game_finished`
- **Round-state rail:** `rounds_a`, `rounds_b`, `map_name`, `team_a_id`, `team_b_id`
- **Mid-round (player path):** `players_a`, `players_b` — each player: `id`, `alive`, `current_health`; optional: `money`, `loadout_value`
- **Mid-round (team fallback):** `team_a_money`, `team_b_money`, `team_a_loadout_value`, `team_b_loadout_value`
- **Mid-round (clock, optional):** `clock_seconds`, `clock_ticking`, `clock_type`
- **Derived (compute in app):** `alive_count_a`, `alive_count_b`, `hp_total_a`, `hp_total_b`, `completeness_score`, `staleness_seconds`

**Reliability rules:**  
Require `valid === true` and `staleness_seconds` below threshold (e.g. 90s); else gate or damp. If no players, use team-level economy only; if no clock, damp mid-round adjustment.
