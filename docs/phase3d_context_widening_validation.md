# Contextual widening patch – validation summary

## 1) Files changed

| Path | Change |
|------|--------|
| `engine/compute/rails_cs2.py` | Modified: context_risk, widening, debug, return 3-tuple |
| `engine/compute/rails.py` | Modified: return (rail_low, rail_high, rails_debug) |
| `backend/services/runner.py` | Modified: unpack rails_debug, merge into derived.debug |
| `tests/unit/test_compute_slice1.py` | Modified: unpack 3 values from compute_rails |
| `tests/unit/test_rails_cs2_basic.py` | Modified: unpack 3 values from compute_rails_cs2 |
| `tests/unit/test_rails_cs2_context_widening.py` | **New** (widening regression + debug keys) |

Untracked (pre-existing): `scripts/replay_anchor_dump.py`.

---

## 2) Behavioral changes

- **Map corridor (rails) is now contextually widened** before being passed to resolve. Base rails are unchanged; then a `context_risk` in [0,1] is computed from:
  - **Leverage:** close score, late round, match-point-ish (leader ≥ 11).
  - **Fragility:** low or asymmetric `loadout_totals` (alive-only).
  - **Missingness:** key microstate (alive + hp or loadout) missing.
- **Widening formula:** `widened_halfwidth = current_halfwidth * (1 + beta * context_risk)` (beta=1), same center, then clamp to series corridor (bounds) and [0,1]. So higher risk → wider map corridor (less false certainty).
- **Series corridor (bounds)** is unchanged; only the map corridor is widened.
- **compute_rails** / **compute_rails_cs2** now return a third value (debug dict); all call sites updated.

---

## 3) Debug outputs added

- **context_risk** (float [0,1])
- **context_risk_components** (dict): `leverage_risk`, `leverage_closeness`, `leverage_lateness`, `leverage_match_point_ish`, `fragility_risk`, `fragility_note` (if loadout missing), `missingness_risk`, `inputs_present` (alive, hp, loadout)
- **uncertainty_multiplier** (float [1.0, 2.0])
- **map_width_before** (float)
- **map_width_after** (float)

These are merged into `derived.debug` by the runner (so appear in `/api/v1/state/current` and stream).

---

## 4) Unit tests run

| Command | Result |
|---------|--------|
| `python tests/unit/test_bounds_cs2_basic.py` | PASS |
| `python tests/unit/test_rails_cs2_basic.py` | PASS |
| `python tests/unit/test_resolve_micro_adj.py` | PASS |
| `python tests/unit/test_midround_v2_cs2.py` | PASS |
| `python tests/unit/test_state_corridor_labels.py` | PASS |
| `python tests/unit/test_no_merged_econ_in_cs2_compute.py` | PASS |
| `python tests/unit/test_rails_cs2_context_widening.py` | PASS |

---

## 5) Replay run details

- **match_id used:** 111632 (present in `logs/bo3_pulls.jsonl`, 2487 entries).
- **max_ticks:** 60  
- **anchors:** 0, 5, 15, 30, 45  

### Two anchor blocks (fragile vs more neutral)

**Anchor 15 (more neutral – BUY_TIME, balanced loadout)**  
- round_phase=BUY_TIME, t_remaining=8.0, bomb=False  
- alive_counts=(5, 5), hp_totals=(500,500), loadout_totals=(27650,27850)  
- rails low=0.5097 high=0.7179 → **map width = 0.2082**  
- p_hat_final=0.6110  

**Anchor 30 (fragile – large loadout asymmetry)**  
- round_phase=BUY_TIME, t_remaining=62.0, bomb=False  
- alive_counts=(5, 5), hp_totals=(500,500), loadout_totals=(16650,1000)  
- rails low=0.5283 high=0.7308 → **map width = 0.2025**  
- p_hat_final=0.7307  

(Score state differs: 8-1 vs 9-1, so bounds differ. At same score 9-1, tick 30 fragile loadout vs tick 45 more balanced: tick 30 width 0.2025, tick 45 width 0.185 → fragile case has wider map corridor.)

**p_hat_final within map corridor:**  
- Anchor 15: 0.5097 ≤ 0.6110 ≤ 0.7179 ✓  
- Anchor 30: 0.5283 ≤ 0.7307 ≤ 0.7308 ✓  

---

## 6) Invariant check

- **series_low ≤ map_low ≤ map_high ≤ series_high:** Rails are computed by `compute_rails_cs2` and clamped to `(bound_lo, bound_hi)` (series corridor); the harness prints map corridor as “rails low/high”. By construction, map corridor is inside bounds, so this holds.
- **map_low ≤ p_hat_final ≤ map_high:**  
  - Anchor 15: 0.5097 ≤ 0.6110 ≤ 0.7179 ✓  
  - Anchor 30: 0.5283 ≤ 0.7307 ≤ 0.7308 ✓  

---

## 7) Known limitations / TODOs

- Replay anchor script does not print series_low/series_high (bounds); only map corridor (rails) and p_hat. Invariant is asserted from code path and from p_hat ∈ [map_low, map_high].
- Context_risk weights (0.4/0.4/0.2) and fragility thresholds are fixed; no config knobs yet.
- No tuning of CONTEXT_WIDEN_BETA (1.0) or uncertainty_multiplier range [1, 2] from data.

---

## 8) Commit

- **Proposed message:** `phase3d: contextual widening for map corridor (context_risk, debug)`  
- **Staged:** engine/compute/rails_cs2.py, engine/compute/rails.py, backend/services/runner.py, tests/unit/test_compute_slice1.py, tests/unit/test_rails_cs2_basic.py, tests/unit/test_rails_cs2_context_widening.py  
- **Not included in this commit:** scripts/replay_anchor_dump.py (untracked, pre-existing).
