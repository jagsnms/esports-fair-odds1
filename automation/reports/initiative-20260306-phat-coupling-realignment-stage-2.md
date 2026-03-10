# Stage report — PHAT Coupling and Movement Realignment (Stage 2)

## Stage scope (approved)

Stage 2 was constrained to:
- In `engine/compute/resolve.py`, IN_PROGRESS branch only: change PHAT from snap/clamp-to-target to movement step per Bible Ch 6 Step 8.
- Formula: `p_hat_final = p_hat_old + confidence * (target_p_hat - p_hat_old)` then clamp to rails (and preserve [0,1] behavior if present).
- No changes to BUY_TIME, FREEZETIME, non-IN_PROGRESS, q/rail/runner/diagnostics/calibration/config.
- No new dependencies or config flags.

## Deliverables completed

1. **Movement step in resolve.py (IN_PROGRESS only)**
   - Replaced `p_hat_final = max(rail_low, min(rail_high, p_mid_clamped))` with:
     - `target_p_hat = result["p_mid"]` (Bible target: rail_low + q*(rail_high - rail_low)).
     - `p_hat_final = p_hat_old + midround_weight * (target_p_hat - p_hat_old)`.
     - `p_hat_final = max(rail_low, min(rail_high, p_hat_final))`.
   - Confidence used: existing `midround_weight = 0.25`.

2. **Test updates (test_resolve_micro_adj.py)**
   - Renamed and rewrote `test_midround_v2_p_hat_equals_p_mid_clamped_when_rails_wide` → `test_midround_v2_p_hat_follows_movement_formula_when_rails_wide`: asserts `p_hat_final = p_hat_old + confidence*(target_p_hat - p_hat_old)` clamped to rails.
   - Renamed and rewrote `test_contract_testing_mode_flags_movement_gap` → `test_contract_testing_mode_runtime_follows_movement_contract`: asserts runtime p_hat_final matches `expected_p_hat_after_movement` (no movement_contract_gap when formula is implemented).
   - Updated `TestResolveMicroAdj` class to call the new test names.

## Files changed

| Path | Change |
|------|--------|
| `engine/compute/resolve.py` | IN_PROGRESS branch: use movement formula toward target_p_hat (result["p_mid"]), then clamp to rails. |
| `tests/unit/test_resolve_micro_adj.py` | Movement-formula assertion test; contract-follows-runtime test; unittest class method names. |

## Validation performed

- `python -m pytest tests/unit/test_resolve_micro_adj.py -v` → 28 passed
- `python -m pytest tests/unit/test_corridor_invariants.py tests/unit/test_invariants_contract_diagnostics.py -v` → 7 passed
- `python -m pytest tests/unit/test_compute_slice1.py tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_rails_cs2_basic.py -v` → 62 passed

## Behavior unchanged (Stage 2 constraints)

- BUY_TIME / FREEZETIME: still freeze at rail midpoint; no code paths changed.
- Non-IN_PROGRESS: still `p_hat_final = p_hat_old`; no code paths changed.
- q computation, rail computation, runner, diagnostics logic, calibration, config: untouched.

## Recommendation

- Stage 2 scope completed as approved.
- Ready for human review and promotion decision into agent-initiative-base.
