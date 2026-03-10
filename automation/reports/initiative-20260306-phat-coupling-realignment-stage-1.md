# Stage report — PHAT Coupling and Movement Realignment (Stage 1)

## Stage scope (approved)

Stage 1 was constrained to:
- add/rewrite contract tests for Bible-defined target/movement behavior,
- add testing-mode diagnostics for behavioral violations,
- clearly separate structural vs behavioral violations in diagnostics/tests,
- avoid production PHAT semantic migration, runner/source harmonization, or calibration retuning.

## Deliverables completed

1. **Contract diagnostics added without changing runtime PHAT outputs**
   - Added `compute_phat_contract_diagnostics(...)` in `engine/diagnostics/invariants.py`.
   - Wired diagnostics into `resolve_p_hat` debug output as `contract_diagnostics` in `engine/compute/resolve.py`.
   - Diagnostics include:
     - `target_p_hat`,
     - `expected_p_hat_after_movement`,
     - structural violations (`q_out_of_bounds`, `rail_order_invalid`),
     - testing-mode behavioral violation (`movement_contract_gap`).

2. **Structural vs behavioral invariant classification**
   - Added canonical `compute_corridor_invariants(...)` in `engine/diagnostics/invariants.py` with:
     - `invariant_structural_violations`,
     - `invariant_behavioral_violations`,
     - mode-aware primary list (`invariant_violations`).
   - Runner now imports this canonical diagnostics function (no PHAT logic migration):
     - `backend/services/runner.py`

3. **Contract and diagnostics tests**
   - Rewrote corridor invariant tests to use canonical diagnostics module and assert structural/behavioral separation:
     - `tests/unit/test_corridor_invariants.py`
   - Added new contract-diagnostics unit tests:
     - `tests/unit/test_invariants_contract_diagnostics.py`
   - Extended resolve tests for contract diagnostics behavior:
     - `tests/unit/test_resolve_micro_adj.py`

## Files changed

| Path | Change |
|------|--------|
| `engine/diagnostics/invariants.py` | Added corridor invariant classification and PHAT contract diagnostics helper functions. |
| `engine/diagnostics/__init__.py` | Exported new diagnostics helpers. |
| `engine/compute/resolve.py` | Added diagnostics-only contract reporting (`contract_diagnostics`) to debug output. |
| `backend/services/runner.py` | Switched to canonical diagnostics import for corridor invariants. |
| `tests/unit/test_corridor_invariants.py` | Reworked tests for structural vs behavioral invariant classification and mode behavior. |
| `tests/unit/test_resolve_micro_adj.py` | Added target formula and testing-mode movement-gap diagnostics tests. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Added new unit coverage for structural/behavioral contract diagnostics. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_resolve_micro_adj.py tests/unit/test_corridor_invariants.py tests/unit/test_invariants_contract_diagnostics.py`
  - Result: `35 passed`
- `python3 -m pytest -q tests/unit/test_compute_slice1.py tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_rails_cs2_basic.py`
  - Result: `62 passed`

## Runtime-semantics boundary check

- No production PHAT update formula migration performed.
- No broad runner/source path harmonization performed.
- No calibration or movement-parameter retuning performed.
- Changes are diagnostics/test-contract focused and bounded to Stage 1.

## Risks / follow-ups

- Behavioral violation diagnostics are currently opt-in for contract checks via `config.contract_testing_mode`.
- Stage 2 will need actual runtime contract migration decisions; this stage intentionally does not implement them.

## Recommendation

- Stage 1 scope completed as approved.
- Ready for human review and explicit approval/deferral decision for Stage 2.
