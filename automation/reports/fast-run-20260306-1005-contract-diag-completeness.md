# Promotion report

branch: fast/run-20260306-1005-contract-diag-completeness
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing per-key `contract_diagnostics` completeness metrics in replay assessment output
- **Title:** `tools/replay_verification_assess.py` did not report required contract-diagnostic key coverage/missing counts

## Why it outranked alternatives

Current evidence on `agent-base` showed no unresolved rank 1-4 issues:
- Structural invariant violations: `python3 tools/replay_verification_assess.py` reported `structural_violations_total=0`.
- Failing canonical tests: `python3 -m pytest -q` passed (`365 passed`).
- Confirmed replay mismatches: replay assessment showed canonical raw-contract processing (`raw_contract_points=3`, `unknown_replay_mode_points=0`) with no mismatch signal.
- High-frequency diagnostic invariant failures: replay assessment reported `behavioral_violations_total=0` and `invariant_violations_total=0`.

The highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): summary output only provided aggregate `points_with_contract_diagnostics` and lacked per-required-field completeness metrics, so field-level contract regressions could not be quickly localized.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `b9991c02c0e758ba860feb8541f25fdc3a06b474`
- Baseline checks:
  - `python3 -m pytest -q` -> `365 passed in 7.74s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Completeness-key probe:
    - `python3 - <<'PY' ... print({k: (k in summary) ...}) ... PY`
    - Result: `{'contract_diagnostics_required_keys': False, 'contract_diagnostics_key_presence_counts': False, 'contract_diagnostics_key_presence_rates': False, 'contract_diagnostics_missing_key_counts': False}`

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added required contract-diagnostic key set and emitted per-key presence counts, missing counts, and presence rates. |
| `tools/schemas/replay_validation_summary.schema.json` | Added new required schema properties for diagnostics completeness metrics. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Extended schema-type assertion support for arrays and added explicit completeness metric assertions. |
| `automation/reports/fast-run-20260306-1005-contract-diag-completeness.md` | Added human-readable report for this run. |
| `automation/reports/fast-run-20260306-1005-contract-diag-completeness.json` | Added machine-readable report artifact for this run. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 tools/replay_verification_assess.py` -> summary includes new completeness maps with full key coverage on raw fixture
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> full key coverage across 6 replay points
- `python3 -m pytest -q` -> `365 passed in 7.18s`

## Before/after evidence

- **Before:** replay summary had no field-level completeness keys; probe confirmed all expected completeness keys were absent.
- **After:** replay summary includes:
  - `contract_diagnostics_required_keys`
  - `contract_diagnostics_key_presence_counts`
  - `contract_diagnostics_missing_key_counts`
  - `contract_diagnostics_key_presence_rates`
  and current fixtures report complete presence (`rate=1.0`, missing `0`) for all required keys.

## Unresolved risks

- Completeness metrics verify field presence, not semantic correctness of each value.
- Current fixtures remain small; broader replay corpora could still reveal value-quality issues even when key presence is complete.

## Stop reason

Stopped after fixing the selected rank-5 diagnostics instrumentation gap with minimal bounded changes and repeated validation; further expansion into semantic-quality analytics would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
