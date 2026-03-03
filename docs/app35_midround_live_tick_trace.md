# CS2 live tick / resolver path and midround parity oracle (`legacy/app/app35_ml.py`)

Traced by following the **final assignment to p_hat** in the live loop and walking backward to the midround function(s) and their inputs. Not inferred from function names alone.

---

## 1. Live tick entry and initial p_hat

- **Location:** ~6507
- **Code:** `(p_hat, p_hat_map_raw) = _compute_cs2_live_fair_for_state(...)`
- **Meaning:** Initial p_hat is the map-level fair probability from round score, econ, pistol, map_name, a_side, lock, contract_scope, series state. **No intraround microstate (alive/hp/bomb/time) is used here.**

---

## 2. Round transition handling (before midround)

- **Locations:** ~6604–6668
- If `current_round_key != prev_round_key`, p_hat can be overwritten with:
  - `prev_band_if_a` (A just won the round), or
  - `prev_band_if_b` (B just won),
  and session state is updated for the new round (frozen rails, band_if_a/b, etc.).
- So the value that enters the midround block is either this transition-resolved p_hat or the initial `_compute_cs2_live_fair_for_state` result.

---

## 3. Frozen round context (inputs to midround)

- **Locations:** ~7164–7179
- `p_base_frozen`, `frozen_lo`, `frozen_hi`, `band_if_a_round_frozen`, `band_if_b_round_frozen` come from session state (set at round start / rail computation).
- `p_round_context` = base for this round (with optional round_context_offset). Used as p_base for V1 and, with frozen_a/frozen_b, for V2 mixture endpoints.

---

## 4. IN_PROGRESS midround block (where final p_hat is set)

- **Location:** 7220–7322

Condition to run midround:

- `cs2_midround_mode != "OFF"`
- `frozen_lo is not None and frozen_hi is not None and frozen_hi > frozen_lo`
- Not `use_snap_this_tick` (so we are not on the single transition snap tick)

There is **no explicit IN_PROGRESS check** in this block; round phase is only used inside `_compute_cs2_midround_features` (`is_live_round_phase` from `round_phase`). The live tick that populates `cs2_live_*` is the same path that runs this block when the condition above holds.

### Step A: Feature computation (always when midround != OFF and rails valid)

- **Function:** `_compute_cs2_midround_features`
- **Call site:** 7226–7245

**Arguments (all from session state or caller):**

| Argument | Session / source |
|----------|-------------------|
| team_a_alive_count | cs2_live_team_a_alive_count |
| team_b_alive_count | cs2_live_team_b_alive_count |
| team_a_hp_alive_total | cs2_live_team_a_hp_alive_total |
| team_b_hp_alive_total | cs2_live_team_b_hp_alive_total |
| bomb_planted | cs2_live_bomb_planted |
| round_time_remaining_s | cs2_live_round_time_remaining_s |
| round_phase | cs2_live_round_phase |
| a_side | Caller variable (UI: Team A side T/CT) |
| team_a_armor_alive_total | cs2_live_team_a_armor_alive_total |
| team_b_armor_alive_total | cs2_live_team_b_armor_alive_total |
| team_a_alive_loadout_total | cs2_live_team_a_alive_loadout_total |
| team_b_alive_loadout_total | cs2_live_team_b_alive_loadout_total |
| live_source | cs2_live_source |
| grid_used_reduced_features | cs2_grid_used_reduced_features |
| grid_completeness_score | cs2_grid_completeness_score |
| grid_staleness_seconds | cs2_grid_staleness_seconds |
| grid_has_players | cs2_grid_has_players |
| grid_has_clock | cs2_grid_has_clock |

**Returns:** dict with `feature_ok`, `alive_diff`, `hp_diff_alive`, `bomb_planted`, `time_remaining_s`, `time_progress`, `is_live_round_phase`, `armor_diff_alive`, `loadout_diff_alive`, `reliability_mult`, etc.

If `not features.get("feature_ok")` (e.g. alive missing): **p_hat = p_round_context** (7248); no adjustment.

### Step B: Final p_hat assignment (when feature_ok)

**V2 (mixture)** — UI default (`cs2_midround_mode == "V2 (mixture)"`, default 5974):

- **Function:** `_apply_cs2_midround_adjustment_v2_mixture`
- **Call site:** 7254–7256  
  `midround_result = _apply_cs2_midround_adjustment_v2_mixture(float(frozen_a), float(frozen_b), features, settings={"a_side": a_side})`
- **Final p_hat assignment:** **7259** — `p_hat = float(midround_result["p_mid_clamped"])`

**V1 (additive):**

- **Function:** `_apply_cs2_midround_adjustment_v1`
- **Call site:** 7266–7268  
  `midround_result = _apply_cs2_midround_adjustment_v1(float(p_round_context), float(frozen_lo), float(frozen_hi), features, settings={"a_side": a_side})`
- **Final p_hat assignment:** **7272** — `p_hat = float(midround_result["p_mid_clamped"])`

So the **actual final assignment to p_hat** during IN_PROGRESS (midround on, feature_ok) is:

- **7259** when mode is V2 (mixture),
- **7272** when mode is V1 (additive).

---

## 5. Midround function details (inputs / outputs)

### _compute_cs2_midround_features (3383–3476)

- **Inputs:** listed in table above (alive, hp, bomb, time, phase, a_side, armor, loadout, grid/reliability flags).
- **Outputs:** `feature_ok` (False if alive missing), `alive_diff`, `hp_diff_alive`, `bomb_planted`, `time_remaining_s`, `time_progress`, `armor_diff_alive`, `loadout_diff_alive`, `reliability_mult`, etc.
- **Econ:** Not “cash” in the sense of live tick econ_a/econ_b; loadout is “alive loadout” total (equipment value of alive players).

### _apply_cs2_midround_adjustment_v1 (3544–3598)

- **Inputs:** `p_base`, `band_lo`, `band_hi`, `features`, `settings` (includes `a_side`).
- **Logic:** Additive adjustment from alive, hp, bomb (signed by a_side: T +bomb, CT −bomb), scaled by time_progress; clip to corridor.
- **Output:** `p_mid_clamped` (and `p_mid_raw`, mid_adj_*, etc.).

### _apply_cs2_midround_adjustment_v2_mixture (3676–3755)

- **Inputs:** `p_if_a`, `p_if_b` (frozen_a, frozen_b), `features`, `settings` (includes `a_side`).
- **Logic:**
  - Intra score from: alive (per-player weight), hp (normalized by hp sum or per-100), loadout (normalized by load sum or per-1000), bomb (signed by a_side), then **urgency = 0.15 + 0.85 * time_progress**, **raw_score = (score_alive + score_hp + score_loadout + score_bomb) * urgency**, **q = sigmoid(raw_score, temp)**.
  - **p_mid_raw = p_if_b + q * (p_if_a - p_if_b)**; clamp to [lo_ep, hi_ep] between the two endpoints.
- **Output:** `p_mid_clamped`, `q`, `raw_score`, component scores, `temp`, `urgency`.
- **Econ in q:** No live cash; loadout is “alive loadout” only.

---

## 6. Summary: parity oracle for “midround mixture parity”

- **Final p_hat in the live loop** (during IN_PROGRESS, midround on, feature_ok) is set at **7259** (V2) or **7272** (V1).
- **Default mode** in the UI is **"V2 (mixture)"** (5974).
- The **canonical midround path** that produces the displayed p_hat in normal use is:
  1. **`_compute_cs2_midround_features`** — builds intraround features (alive, hp, bomb, time, phase, a_side, armor, loadout, grid/reliability).
  2. **`_apply_cs2_midround_adjustment_v2_mixture`** — computes q from those features, then **p_hat = p_if_b + q*(p_if_a - p_if_b)** clamped to the endpoint band.

So the **parity oracle for “midround mixture parity”** in the new engine should be:

- **`_apply_cs2_midround_adjustment_v2_mixture`** (and the **q_intra** it computes via its internal score from alive, hp, loadout, bomb, time), with features supplied by **`_compute_cs2_midround_features`** (or an engine equivalent that provides the same logical inputs: alive, hp, bomb, time, a_side, and optionally armor/loadout/reliability).

The new engine’s **compute_q_intra_cs2** is a debug-only v1 that uses only alive/hp and optionally bomb (narrowing) and time (gating). Parity with app35’s **V2 mixture** would require either:

- Extending the engine’s q_intra to include loadout (and optionally armor) and a_side for bomb sign, and matching the V2 scoring/urgency/temp formula, or  
- Documenting that the current engine midround blend is “v1 parity” (alive/hp/bomb/time only) and that full V2 mixture parity is a later step.

---

## 7. Exact call sites (line references)

| Item | Line(s) |
|------|--------|
| Initial p_hat | 6507 |
| Round transition overwrite of p_hat | 6631, 6646 |
| Frozen context / p_round_context | 7167–7179 |
| _compute_cs2_midround_features call | 7226–7245 |
| feature_ok false → p_hat = p_round_context | 7248 |
| _apply_cs2_midround_adjustment_v2_mixture call | 7254–7256 |
| **Final p_hat (V2 mixture)** | **7259** |
| _apply_cs2_midround_adjustment_v1 call | 7266–7268 |
| **Final p_hat (V1 additive)** | **7272** |
| Default cs2_midround_mode | 5974 ("V2 (mixture)") |
