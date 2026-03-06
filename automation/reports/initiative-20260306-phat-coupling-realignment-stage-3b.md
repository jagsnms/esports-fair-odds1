# Stage report — PHAT Coupling and Movement Realignment (Stage 3B)

## Stage scope (approved)

Stage 3B — inter-map-break and replay-mode parity completion:
- Real inter-map-break parity in `backend/services/runner.py` via one narrow shared helper.
- Explicit replay raw vs point runtime policy tags in runner-produced debug.
- Replace handcrafted parity tests with tests that assert on real runner-produced output.

No compute semantics were changed (resolve, midround_v2, q_intra, rails, bounds untouched). Stage 3B is intended to complete Stage 3 together with Stage 3A.

## Deliverables completed

1. **Shared helper `_inter_map_break_phat_and_dbg`**
   - Added in `backend/services/runner.py`: builds canonical inter_map_break p_hat and debug payload (rails from bounds, continuity from last_p, explain.phase = "inter_map_break"). Callers add source-specific keys after.

2. **BO3 / GRID / REPLAY raw use helper; continuity rule**
   - BO3: when `is_break`, `last_p` from store if `write_to_store` else `getattr(session_runtime, "last_p_hat", None)`. Call helper; add bo3_monotonic_gate, match_context_diag. After both branches, `setattr(session_runtime, "last_p_hat", p_hat)`.
   - GRID: same continuity rule (store when write_to_store, else session-local last_p_hat); call helper; add source, grid_series_id, match_context_diag; set session_runtime.last_p_hat after.
   - REPLAY raw: continuity from store only (single-session); call helper. No session_runtime.

3. **Replay-mode tags in runner-produced debug**
   - Raw replay path: `dbg["replay_mode"] = "raw_contract"` after the is_break/else block.
   - Point passthrough: `debug={"explain": explain, "replay_mode": "point_passthrough"}` in Derived.

4. **Tests assert on real runner output**
   - `test_runner_inter_map_break_parity.py`: `test_real_runner_inter_map_break_produces_canonical_debug` — run replay with `detect_inter_map_break` patched True; assert stored derived.debug has canonical keys (inter_map_break, inter_map_break_reason, p_hat_old, p_hat_final, series_low, series_high, map_low, map_high, explain.phase).
   - `test_runner_replay_contract_mode.py`: `test_real_runner_raw_replay_tags_contract_mode` — one raw payload through _tick_replay; assert derived.debug["replay_mode"] == "raw_contract". `test_real_runner_point_replay_tags_passthrough_mode` — one point payload; assert derived.debug["replay_mode"] == "point_passthrough".
   - No handcrafted-only parity proof.

## Files changed

| Path | Change |
|------|--------|
| `backend/services/runner.py` | Added _inter_map_break_phat_and_dbg; BO3/GRID/REPLAY raw use it; continuity store vs session-local; replay_mode tags; session_runtime.last_p_hat. |
| `tests/unit/test_runner_inter_map_break_parity.py` | Replaced with real runner-output test (replay + patched detect_inter_map_break → assert derived.debug). |
| `tests/unit/test_runner_replay_contract_mode.py` | Real runner-output tests for raw_contract and point_passthrough tags; kept helper detection test. |

## Validation performed

- `python -m pytest tests/unit/test_resolve_micro_adj.py tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_corridor_invariants.py tests/unit/test_runner_inter_map_break_parity.py tests/unit/test_runner_replay_contract_mode.py tests/unit/test_runner_source_contract_parity.py tests/unit/test_grid_reducer_and_envelope.py -v` → 56 passed
- `python -m pytest tests/unit/test_runner_bo3_hold.py tests/unit/test_compute_slice1.py tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_rails_cs2_basic.py -v` → 86 passed

## Shared helper

Yes: `_inter_map_break_phat_and_dbg(bound_low, bound_high, break_reason, last_p, bounds_debug)` in runner.py (module-level). Returns (p_hat, dbg) with canonical keys; callers add source-specific keys.

## Compute semantics

No changes to engine/compute/resolve.py, midround_v2_cs2.py, q_intra_cs2.py, rails_cs2.py, or bounds.py. Grid phase normalization in grid_reducer.py unchanged.

## Recommendation

Stage 3B completed as scoped. Ready for human review and promotion. Together with Stage 3A, completes Stage 3 (runner/source harmonization).
