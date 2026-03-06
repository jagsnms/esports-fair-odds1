# Promotion report

branch: fast/run-20260306-1608-contract-diag-type-coverage
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing per-key contract-diagnostics type coverage metrics in replay assessment
- **Title:** `tools/replay_verification_assess.py` enforced key presence but did not measure value-type validity per required diagnostics field

## Why it outranked alternatives

Applied issue ladder from current `agent-base` evidence:

1. Structural invariant violations: none (`structural_violations_total=0` from replay assessment).
2. Failing canonical tests: none (`375 passed`).
3. Confirmed replay mismatches: none (raw-contract replay points only; no mismatch signal).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): current replay summary tracked required key presence only, but had no per-key type-validity metrics, which can hide malformed diagnostics even when keys exist.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.72s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Type-coverage probe:
    - `python3 - <<'PY' ... run_assessment(...) ... PY`
    - result: `{'contract_diagnostics_expected_types': False, 'contract_diagnostics_key_type_valid_counts': False, 'contract_diagnostics_key_type_invalid_counts': False, 'contract_diagnostics_key_type_valid_rates': False}`

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added required type-contract descriptors and per-key type-valid/invalid counts plus valid-rate metrics for `contract_diagnostics`. |
| `tools/schemas/replay_validation_summary.schema.json` | Added schema-required properties for diagnostics type-coverage output. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Extended assertions to require and validate type-coverage metrics and expected type descriptors. |
| `automation/reports/fast-run-20260306-1608-contract-diag-type-coverage.md` | Added human-readable run report artifact. |
| `automation/reports/fast-run-20260306-1608-contract-diag-type-coverage.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> summary now includes:
  - `contract_diagnostics_expected_types`
  - `contract_diagnostics_key_type_valid_counts`
  - `contract_diagnostics_key_type_invalid_counts`
  - `contract_diagnostics_key_type_valid_rates`
  with all required keys at valid=`6`, invalid=`0`, rate=`1.0` on fixture.
- `python3 -m pytest -q` -> `375 passed in 7.28s`

## Before/after evidence

- **Before:** replay assessment output had no per-key diagnostics type-coverage fields; probe confirmed all four fields absent.
- **After:** replay assessment emits expected type descriptors and per-key type-validity metrics; fixture run reports full type validity across all required diagnostics keys while full canonical tests remain green.

## Unresolved risks

- Type coverage validates shape/type contracts, not semantic correctness of numeric values.
- Current replay fixtures are small; broader corpora may reveal value-quality issues not detectable by type-only instrumentation.

## Stop reason

Stopped after fixing the selected rank-5 instrumentation gap with minimal bounded edits and repeated validation; further expansion into semantic diagnostics scoring would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
