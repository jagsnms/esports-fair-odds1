# Promotion report

branch: fast/run-20260306-2005-contract-cash-diag
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing economy totals in `contract_diagnostics` payload and replay completeness gate
- **Title:** `cash_totals` was absent from contract diagnostics, leaving team-economy context out of Bible-aligned diagnostic evidence

## Why it outranked alternatives

Issue ranking from current `agent-base` evidence:

1. Structural invariant violations: none (`structural_violations_total=0`).
2. Failing canonical tests: none (`379 passed`).
3. Confirmed replay mismatches: none (raw-contract replay assessment clean).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, rank 5 (missing instrumentation blocking diagnosis) was the highest unresolved bounded issue. Bible Chapter 6 Step 1 includes team economy in the round-state vector, but `contract_diagnostics` and replay completeness checks did not include `cash_totals`.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `3a871b09bf26f9afefec73fb2fdad16118138ea8`
- Baseline checks:
  - `python3 -m pytest -q` -> `379 passed in 7.88s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract/replay key probe:
    - `python3 - <<'PY' ... resolve_p_hat(...) + run_assessment(...) ... PY`
    - result: `cash_totals_in_contract_diagnostics=False`
    - result: `cash_totals_required_in_replay_assess=False`

## Files changed

| Path | Change |
|------|--------|
| `engine/diagnostics/invariants.py` | Added `cash_totals` field to `compute_phat_contract_diagnostics` payload contract. |
| `engine/compute/resolve.py` | Wired validated `Frame.cash_totals` into contract diagnostics emission. |
| `tools/replay_verification_assess.py` | Added `cash_totals` to required contract-diagnostics completeness keys. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Updated diagnostics unit inputs/assertions to include `cash_totals`. |
| `tests/unit/test_runner_source_contract_parity.py` | Added parity assertion that resolve-emitted diagnostics include `cash_totals`. |
| `automation/reports/fast-run-20260306-2005-contract-cash-diag.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-2005-contract-cash-diag.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py` -> `6 passed`
- `python3 tools/replay_verification_assess.py` -> `cash_totals` present in required keys with presence rate `1.0`, missing `0`
- `python3 -m pytest -q` -> `379 passed in 7.39s`
- Post-change probe:
  - `python3 - <<'PY' ... resolve_p_hat(...) + run_assessment(...) ... PY`
  - result: `cash_totals_in_contract_diagnostics=True`, `cash_totals_value=(3200.0, 1800.0)`
  - result: `cash_totals_required_in_replay_assess=True`, `cash_totals_presence_rate=1.0`

## Before/after evidence

- **Before:** `contract_diagnostics` omitted `cash_totals`, and replay assessment did not require that key.
- **After:** `contract_diagnostics` emits `cash_totals` and replay assessment enforces/report its completeness (`presence_rate=1.0`, `missing=0` on canonical raw fixture).

## Unresolved risks

- This run enforces field presence, not semantic correctness of economy inputs from all providers.
- Replay fixtures are still small; broader corpora may reveal upstream normalization issues despite complete key presence.

## Stop reason

Stopped after fixing the selected rank-5 instrumentation gap with minimal bounded edits and repeated issue-specific validation; further expansion into additional economy semantics would exceed single-issue scope.

## Recommendation

- `promote`
