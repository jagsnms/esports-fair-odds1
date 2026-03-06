# Promotion report

branch: fast/run-20260306-1107-contract-roundstate-diag
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing round-state vector fields in `contract_diagnostics` payload
- **Title:** Replay/contract diagnostics lacked alive/hp/loadout and round phase metadata required for full Bible-aligned diagnosis

## Why it outranked alternatives

Issue ranking was applied from current `agent-base` evidence:

1. Structural invariant violations: none (`structural_violations_total=0`).
2. Failing canonical tests: none (`369 passed`).
3. Confirmed replay mismatches: none (raw-contract replay assessment clean).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): `contract_diagnostics` omitted core round-state vector fields needed to localize failures to concrete round state.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `5f52998862ad26c31924a376b239a39905549519`
- Baseline checks:
  - `python3 -m pytest -q` -> `369 passed in 7.79s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract diagnostics field probe:
    - `python3 - <<'PY' ... resolve_p_hat(...) ... PY`
    - result: `{'alive_count_a': False, 'alive_count_b': False, 'hp_total_a': False, 'hp_total_b': False, 'loadout_total_a': False, 'loadout_total_b': False, 'round_phase': False, 'round_number': False}`
  - Replay required-key probe:
    - `python3 - <<'PY' ... run_assessment(...) ... PY`
    - result: `{'alive_count_a': False, 'alive_count_b': False, 'hp_total_a': False, 'hp_total_b': False, 'loadout_total_a': False, 'loadout_total_b': False, 'round_phase': False, 'round_number': False}`

## Files changed

| Path | Change |
|------|--------|
| `engine/diagnostics/invariants.py` | Extended `compute_phat_contract_diagnostics` payload with round-state vector fields (`alive_counts`, `hp_totals`, `loadout_totals`, `round_phase`, `round_number`). |
| `engine/compute/resolve.py` | Wired round-state values from `Frame`/bomb metadata into contract diagnostics with shape/type guards. |
| `tools/replay_verification_assess.py` | Added the new round-state fields to required contract-diagnostics completeness keys. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Updated contract diagnostics unit tests for new args and assertions. |
| `tests/unit/test_runner_source_contract_parity.py` | Added parity assertions that resolve-emitted diagnostics include new round-state fields. |
| `automation/reports/fast-run-20260306-1107-contract-roundstate-diag.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1107-contract-roundstate-diag.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py tests/unit/test_resolve_micro_adj.py` -> `34 passed`
- `python3 tools/replay_verification_assess.py` -> new keys present with full coverage (`alive_counts`, `hp_totals`, `loadout_totals`, `round_phase`, `round_number` all rate `1.0`, missing `0`)
- `python3 -m pytest -q` -> `369 passed in 7.23s`

## Before/after evidence

- **Before:** `contract_diagnostics` and replay completeness contract did not include round-state vector fields (alive/hp/loadout/phase metadata), reducing root-cause visibility when invariants fire.
- **After:** diagnostics now emit those fields every point, replay assessment enforces their completeness, and all targeted + canonical tests remain green.

## Unresolved risks

- Field presence is now enforced, but semantic validity of each value is still governed by upstream normalize/ingest correctness.
- `round_number` can legitimately be `null` when source payloads omit it; this run improves observability, not source backfilling.

## Stop reason

Stopped after fixing the selected rank-5 instrumentation gap with minimal bounded edits and repeated validation; further expansion into additional diagnostic semantics would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
