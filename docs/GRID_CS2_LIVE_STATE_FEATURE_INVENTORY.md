# CS2 Live-State Feature Inventory (GRID Schema)

**Source of truth:** `docs/GRID_API_SPEC.md` (from gridapi.docx).  
**Scope:** Concrete CSGO types used by CS2: `SeriesTeamStateCsgo`, `GameTeamStateCsgo`, `GamePlayerStateCsgo`, plus `GameState`, `ClockState`, `SegmentState`, `StructureState`, `NonPlayerCharacterState`, `Objective`.

Features are grouped for the betting model into: **round-state rail** (latched per round), **mid-round dynamic** (tick-updated inside corridor), and **reliability/completeness** (gate or damp adjustments).

---

## 1) Round-state rail features (latched per round)

Features that update at round boundaries; use for round-level win probability and rail resets.

### Series level (seriesState)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.teams` (SeriesTeamStateCsgo) | `score` | Int! | Series score (e.g. games won in Bo3); anchors map/game priors | **Required** |
| `seriesState.teams` | `won` | Boolean! | Series winner; terminal state | **Required** |
| `seriesState.teams` | `id` | ID! | Team identity for side mapping | **Required** |
| `seriesState.teams` | `name` | String! | Human-readable team; optional for display | Optional |

### Game level (seriesState.games[])

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].sequenceNumber` | sequenceNumber | Int! | Game index in series (map number) | **Required** |
| `seriesState.games[].map` | id, name | MapState | Map identity; map-specific win rates | **Required** (at least name) |
| `seriesState.games[].started` | started | Boolean! | Game has begun; gates live model | **Required** |
| `seriesState.games[].finished` | finished | Boolean! | Round/game over; latch final score | **Required** |
| `seriesState.games[].forfeited` | forfeited | Boolean! | Forfeit; special terminal state | Optional |

### Game team level (seriesState.games[].teams — GameTeamStateCsgo)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].teams[].score` | score | Int! | Round score this game; primary round-state rail | **Required** |
| `seriesState.games[].teams[].won` | won | Boolean! | Game/round winner; terminal for that game | **Required** |
| `seriesState.games[].teams[].id` | id | ID! | Team identity | **Required** |
| `seriesState.games[].teams[].side` | side | String! | T/CT; needed for round-half and economy interpretation | **Required** |
| `seriesState.games[].teams[].firstKill` | firstKill | Boolean! | First blood; predictive of round win | Optional |
| `seriesState.games[].teams[].kills` | kills | Int! | Round kills; strong round-outcome signal | **Required** |
| `seriesState.games[].teams[].deaths` | deaths | Int! | Round deaths; mirror of opponent kills | **Required** |
| `seriesState.games[].teams[].objectives` | objectives | [Objective!]! | Plant/defuse; round outcome and phase | Optional (list) |

### Segment/round level (seriesState.games[].segments[] — SegmentState)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].segments[].id` | id | ID! | Round identity | Optional |
| `seriesState.games[].segments[].sequenceNumber` | sequenceNumber | Int! | Round number in game | **Required** (if using segments) |
| `seriesState.games[].segments[].type` | type | String! | Round type (e.g. regulation/OT) | Optional |
| `seriesState.games[].segments[].started` | started | Boolean! | Round has started | Optional |
| `seriesState.games[].segments[].finished` | finished | Boolean! | Round over; latch round outcome | **Required** (if using segments) |
| `seriesState.games[].segments[].startedAt` | startedAt | DateTime | Round start time | Optional |
| `seriesState.games[].segments[].duration` | duration | Duration! | Round length | Optional |
| `seriesState.games[].segments[].teams` | teams | [SegmentTeamState!]! | Per-segment team scores/outcomes | Optional (SegmentTeamStateCsgo if available) |

### Objective (for round rail — plant/defuse)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `objectives[].id` | id | ID! | Objective identity | Optional |
| `objectives[].type` | type | String! | e.g. plant, defuse; drives round phase | **Required** (for mid-round) |
| `objectives[].completedFirst` | completedFirst | Boolean! | First completion on level | Optional |
| `objectives[].completionCount` | completionCount | Int! | Count of completions | Optional |

---

## 2) Mid-round dynamic features (tick-updated inside corridor)

Features that update during the round (tick or event stream); use for in-round win probability and corridor logic.

### Game clock (seriesState.games[].clock — ClockState)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].clock.currentSeconds` | currentSeconds | Int | Round/game clock; time pressure, bomb timer proxy | **Required** |
| `seriesState.games[].clock.ticking` | ticking | Boolean | Clock running; distinguishes live vs paused | **Required** |
| `seriesState.games[].clock.type` | type | String | e.g. gameClock, bombTimer; which phase | **Required** |
| `seriesState.games[].clock.ticksBackwards` | ticksBackwards | Boolean | Countdown vs count-up | Optional |
| `seriesState.games[].clock.id` | id | String | Clock identity | Optional |

### Game team — economy & aggregate (GameTeamStateCsgo)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].teams[].money` | money | Int! | Team cash; buy capability next round | **Required** |
| `seriesState.games[].teams[].loadoutValue` | loadoutValue | Int! | Equipment value this round | **Required** |
| `seriesState.games[].teams[].netWorth` | netWorth | Int! | money + loadout; strength proxy | **Required** |

### Game player (seriesState.games[].teams[].players[] — GamePlayerStateCsgo)

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].teams[].players[].alive` | alive | Boolean! | Alive count drives 1vX and clutch | **Required** |
| `seriesState.games[].teams[].players[].currentHealth` | currentHealth | Int! | HP; survivability and trade potential | **Required** |
| `seriesState.games[].teams[].players[].currentArmor` | currentArmor | Int! | Armor; effective HP | **Required** |
| `seriesState.games[].teams[].players[].money` | money | Int! | Per-player economy | **Required** |
| `seriesState.games[].teams[].players[].loadoutValue` | loadoutValue | Int! | Per-player equipment value | **Required** |
| `seriesState.games[].teams[].players[].netWorth` | netWorth | Int! | Per-player strength proxy | **Required** |
| `seriesState.games[].teams[].players[].position` | position | Coordinates | Player position; positioning and rotations | Optional |
| `seriesState.games[].teams[].players[].id` | id | ID! | Player identity | **Required** |
| `seriesState.games[].teams[].players[].name` | name | String! | Display | Optional |
| `seriesState.games[].teams[].players[].kills` | kills | Int! | In-round kills (can tick up mid-round) | **Required** |
| `seriesState.games[].teams[].players[].deaths` | deaths | Int! | In-round deaths | **Required** |
| `seriesState.games[].teams[].players[].damageDealt` | damageDealt | Int! | Damage output this round | Optional |
| `seriesState.games[].teams[].players[].damageTaken` | damageTaken | Int! | Damage received | Optional |
| `seriesState.games[].teams[].players[].objectives` | objectives | [Objective!]! | Plant/defuse by player; bomb carrier / defuser | Optional |

### Structures (seriesState.games[].structures[] — StructureState) — bomb sites / breakables

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].structures[].id` | id | ID! | Structure identity (e.g. site A/B) | Optional |
| `seriesState.games[].structures[].type` | type | String! | Site type; bomb plant target | **Required** (for phase) |
| `seriesState.games[].structures[].side` | side | String! | Which side controls / site affiliation | Optional |
| `seriesState.games[].structures[].teamId` | teamId | ID! | Controlling team | Optional |
| `seriesState.games[].structures[].currentHealth` | currentHealth | Int! | Site health (e.g. breakable) | Optional |
| `seriesState.games[].structures[].destroyed` | destroyed | Boolean! | Site destroyed; post-plant state | Optional |
| `seriesState.games[].structures[].respawnClock` | respawnClock | ClockState | Respawn timer | Optional |
| `seriesState.games[].structures[].position` | position | Coordinates | Site position | Optional |

### Non-player characters (seriesState.games[].nonPlayerCharacters[] — NonPlayerCharacterState) — bomb entity

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].nonPlayerCharacters[].id` | id | ID! | NPC identity (e.g. bomb) | Optional |
| `seriesState.games[].nonPlayerCharacters[].type` | type | String! | e.g. bomb; drives phase (planted/carried) | **Required** (for phase) |
| `seriesState.games[].nonPlayerCharacters[].side` | side | String! | Carrier side or neutral | Optional |
| `seriesState.games[].nonPlayerCharacters[].position` | position | Coordinates | Bomb location; site/rotation | Optional |
| `seriesState.games[].nonPlayerCharacters[].alive` | alive | Boolean! | Bomb active (e.g. carried vs planted) | Optional |
| `seriesState.games[].nonPlayerCharacters[].respawnClock` | respawnClock | ClockState | Bomb timer when planted | **Required** (for mid-round when type=bomb) |

---

## 3) Reliability / completeness features (gate or damp adjustments)

Used to decide whether to trust or dampen live-state-driven probability adjustments.

### Series level

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.valid` | valid | Boolean! | Data marked accurate by provider; gate live model | **Required** |
| `seriesState.updatedAt` | updatedAt | DateTime! | Freshness; staleness damping or reject | **Required** |
| `seriesState.version` | version | Version! | Schema version; compatibility check | Optional |
| `seriesState.started` | started | Boolean! | Series has started; gate any series-level use | **Required** |
| `seriesState.finished` | finished | Boolean! | Series over; terminal | **Required** |
| `seriesState.forfeited` | forfeited | Boolean! | Forfeit; special handling | Optional |

### Game level

| GraphQL path | Field | Type | Why it matters for win probability | Normalized contract |
|-------------|------|------|------------------------------------|----------------------|
| `seriesState.games[].started` | started | Boolean! | Game live; gate game-level model | **Required** |
| `seriesState.games[].finished` | finished | Boolean! | Game over | **Required** |
| `seriesState.games[].paused` | paused | Boolean! | Paused; optionally freeze or damp updates | Optional |

---

## 4) Bomb / phase / clock / objective fields for mid-round modeling

Summary of spec fields that can improve mid-round modeling (bomb phase, time pressure, objectives).

| Location | Field | Type | Use for mid-round |
|----------|------|------|--------------------|
| **GameState** | `clock` | ClockState | Main round/game timer; bomb countdown if type indicates it |
| **ClockState** | `currentSeconds` | Int | Time remaining (e.g. bomb or round clock) |
| **ClockState** | `type` | String | Distinguish gameClock vs bomb timer vs respawn |
| **ClockState** | `ticking` | Boolean | Whether timer is running |
| **ClockState** | `ticksBackwards` | Boolean | Countdown vs count-up |
| **GameState** | `structures` | [StructureState!]! | Bomb sites (type); destroyed = post-plant state |
| **GameState** | `nonPlayerCharacters` | [NonPlayerCharacterState!]! | Bomb entity: type, position, alive, respawnClock (bomb timer) |
| **StructureState** | `destroyed` | Boolean! | Site destroyed (e.g. plant completed) |
| **StructureState** | `respawnClock` | ClockState | E.g. site “respawn” or next phase |
| **NonPlayerCharacterState** | `respawnClock` | ClockState | Bomb timer when planted |
| **Team / Player** | `objectives` | [Objective!]! | Objective type (plant/defuse); completionCount, completedFirst |
| **Objective** | `type` | String! | e.g. plant, defuse |
| **Objective** | `completionCount` | Int! | How many times completed |
| **SegmentState** | `finished` | Boolean! | Round over; latch before next round |
| **SegmentState** | `sequenceNumber` | Int! | Round index for rail alignment |

**Suggested minimal set for mid-round:**  
`seriesState.games[].clock` (currentSeconds, type, ticking), `seriesState.games[].structures[].type` and `destroyed`, `seriesState.games[].nonPlayerCharacters[].type` and `respawnClock`, and `seriesState.games[].teams[].objectives` / `players[].objectives` (type, completionCount) for plant/defuse phase.

---

## 5) Normalized contract summary

- **Required for round-state rail:** Series and game team `score`, `won`, `id`, `side`; game `sequenceNumber`, `map`, `started`, `finished`; (if using segments) segment `sequenceNumber`, `finished`; team `kills`, `deaths`.
- **Required for mid-round dynamic:** Game `clock` (currentSeconds, type, ticking); game team `money`, `loadoutValue`, `netWorth`; player `alive`, `currentHealth`, `currentArmor`, `money`, `loadoutValue`, `netWorth`, `id`, `kills`, `deaths`.
- **Required for reliability:** `seriesState.valid`, `updatedAt`, `started`, `finished`; `games[].started`, `finished`.
- **Optional but valuable:** `firstKill`, `objectives` (team/player), `position`, `structures` (type, destroyed), `nonPlayerCharacters` (type, respawnClock), segment-level state, damage fields.

No code or repo files were modified; this is a plan only.
