# Stage report — PHAT Coupling and Movement Realignment (Stage 3)

## Stage scope (approved)

Stage 3 was constrained to runner/source harmonization only:
- Canonical GRID phase normalization for resolve contract parity.
- Inter-map-break continuity parity (tests only; no shared helper extracted to avoid broad refactor).
- Explicit replay raw vs point contract policy (tests document and assert policy).
- Explicit invariant-mode wiring at runner call sites.

No compute semantics were changed (resolve, midround_v2, q_intra, rails, bounds untouched).

## Deliverables completed

1. **Config.invariant_diagnostics and runner wiring**
   - Added `invariant_diagnostics: bool = False` to `engine/models.py` Config.
   - In `backend/services/runner.py`, before each of the three `resolve_p_hat` call sites (BO3, GRID, replay raw), added `setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False))` so that when enabled, resolve emits contract_diagnostics.

2. **Canonical GRID phase normalization**
   - In `engine/ingest/grid_reducer.py`: added `_normalize_grid_phase(clock_type)` mapping GRID clock_type to BUY_TIME, FREEZETIME, IN_PROGRESS. Replaced `state.round_phase = state.clock_type or "gameClock"` with `state.round_phase = _normalize_grid_phase(state.clock_type)`. Exported constants GRID_PHASE_BUY_TIME, GRID_PHASE_FREEZETIME, GRID_PHASE_IN_PROGRESS. `bomb_planted` now derived from raw clock_type for bomb presence.

3. **New tests**
   - `tests/unit/test_runner_source_contract_parity.py`: asserts that when config.invariant_diagnostics is True and resolve is called with that config, debug contains contract_diagnostics.
   - `tests/unit/test_runner_inter_map_break_parity.py`: asserts the inter_map_break debug structure contract (required keys and explain.phase).
   - `tests/unit/test_runner_replay_contract_mode.py`: asserts raw vs point payload detection and that point-like payloads have required keys (replay contract policy).
   - `tests/unit/test_grid_reducer_and_envelope.py`: added `test_grid_phase_normalized_to_canonical_vocabulary` for _normalize_grid_phase and reduce_event round_phase.

## Files changed

| Path | Change |
|------|--------|
| `engine/models.py` | Added Config.invariant_diagnostics. |
| `backend/services/runner.py` | Set contract_testing_mode from invariant_diagnostics before each resolve_p_hat call (3 sites). |
| `engine/ingest/grid_reducer.py` | _normalize_grid_phase(), use in reduce_event; bomb_planted from clock_type. |
| `tests/unit/test_grid_reducer_and_envelope.py` | test_grid_phase_normalized_to_canonical_vocabulary. |
| `tests/unit/test_runner_source_contract_parity.py` | New: invariant_diagnostics → contract_diagnostics test. |
| `tests/unit/test_runner_inter_map_break_parity.py` | New: inter_map_break dbg structure contract test. |
| `tests/unit/test_runner_replay_contract_mode.py` | New: raw vs point contract policy tests. |

## Validation performed

- `python -m pytest -q tests/unit/test_resolve_micro_adj.py tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_corridor_invariants.py` → 35 passed
- `python -m pytest -q tests/unit/test_runner_bo3_hold.py tests/unit/test_grid_reducer_and_envelope.py tests/unit/test_runner_telemetry_status.py` → 45 passed
- `python -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_runner_inter_map_break_parity.py tests/unit/test_runner_replay_contract_mode.py` → 4 passed

## Narrow helper

No shared inter-map-break helper was introduced; parity is enforced by tests and existing in-path logic to avoid broad refactor of runner.py.

## Compute semantics

No changes to engine/compute/resolve.py, midround_v2_cs2.py, q_intra_cs2.py, rails_cs2.py, or bounds.py.

## Recommendation

Stage 3 scope completed as approved. Ready for human review and promotion decision.
