# Promotion report

branch: fast/run-20260306-0704-contract-diag-coverage
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing contract diagnostics fields required for replay diagnosis
- **Title:** `contract_diagnostics` lacked rail/p-hat/timer state fields needed by Bible diagnostics contract

## Why it outranked alternatives

Current evidence on `agent-base` showed no unresolved rank 1-4 issues:
- Structural invariant violations: replay assessment reported `structural_violations_total=0`.
- Failing canonical tests: `python3 -m pytest -q` passed.
- Confirmed replay mismatches: no mismatch signal in replay assessment.
- High-frequency diagnostic invariant failures: replay assessment reported zero behavioral/invariant totals.

The highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): replay-emitted `contract_diagnostics` existed, but key state fields were missing, limiting diagnostic usefulness for contract drift analysis.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `ca748b2b5364459ce597449bce4d7870b53fad6e`
- Baseline checks:
  - `python3 -m pytest -q` → `361 passed in 7.71s`
  - `python3 tools/replay_verification_assess.py` → `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract diagnostics key probe on raw replay sample:
    - `contract_diag_keys` lacked required state fields
    - `missing_keys=['rail_low','rail_high','p_hat_prev','p_hat_final','round_time_remaining_s','is_bomb_planted']`

## Files changed

| Path | Change |
|------|--------|
| `engine/diagnostics/invariants.py` | Extended `compute_phat_contract_diagnostics` payload with rail values, p-hat pre/final, and timer state (`round_time_remaining_s`, `is_bomb_planted`). |
| `engine/compute/resolve.py` | Wired frame-derived timer state into contract diagnostics and passed through existing rail/p-hat context. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Updated calls for new diagnostic fields and added assertions for emitted core state fields. |
| `tests/unit/test_runner_source_contract_parity.py` | Added integration assertions that resolve output includes the new contract diagnostics state fields. |
| `automation/reports/fast-run-20260306-0704-contract-diag-coverage.md` | Added this human-readable run report. |
| `automation/reports/fast-run-20260306-0704-contract-diag-coverage.json` | Added machine-readable run report artifact for this run. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_runner_source_contract_parity.py` → `5 passed`
- `python3 -m pytest -q` → `362 passed in 7.28s`
- `python3 tools/replay_verification_assess.py` → replay assessment clean (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`)
- Post-change contract diagnostics key probe on raw replay sample:
  - `missing_keys=[]`

## Before/after evidence

- **Before:** `contract_diagnostics` lacked explicit rail/p-hat/timer-state fields; probe showed missing keys `['rail_low','rail_high','p_hat_prev','p_hat_final','round_time_remaining_s','is_bomb_planted']`.
- **After:** `contract_diagnostics` includes those fields; probe shows `missing_keys=[]` and replay/canonical tests remain green.

## Unresolved risks

- Raw fixtures currently do not exercise planted-bomb countdown semantics, so `is_bomb_planted=True` timer-state combinations in replay diagnostics remain lightly covered.
- This run improves payload completeness but does not yet add replay-summary metrics for per-point field completeness.

## Stop reason

Stopped after the selected rank-5 instrumentation gap was fixed with minimal bounded changes and validated; further expansion (broader replay diagnostics analytics) would exceed single-issue scope.

## Recommendation

- `promote`
