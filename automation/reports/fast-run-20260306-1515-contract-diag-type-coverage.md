# Promotion report

branch: fast/run-20260306-1515-contract-diag-type-coverage
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing per-key contract diagnostics type/shape coverage metrics in replay assessment
- **Title:** `tools/replay_verification_assess.py` only measured key presence, not value type/shape conformance for required diagnostics fields

## Why it outranked alternatives

Current evidence on `agent-base` showed no unresolved rank 1-4 issues:

1. Structural invariant violations: `python3 tools/replay_verification_assess.py` reported `structural_violations_total=0`.
2. Failing canonical tests: `python3 -m pytest -q` passed (`375 passed`).
3. Confirmed replay mismatches: replay assessment showed canonical raw-contract processing with no mismatch signal.
4. High-frequency diagnostic invariant failures: replay assessment reported `behavioral_violations_total=0` and `invariant_violations_total=0`.

With ranks 1-4 clear, the highest unresolved bounded issue remained rank 5 (missing instrumentation blocking diagnosis): replay assessment had per-key presence metrics but no per-key type/shape conformance metrics, leaving semantic diagnostic regressions hard to localize.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.79s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Type-coverage probe:
    - `python3 -c "... run_assessment ... keys=['contract_diagnostics_required_key_type_contract','contract_diagnostics_type_mismatch_counts','contract_diagnostics_type_mismatch_rates'] ..."`
    - result: `{'contract_diagnostics_required_key_type_contract': False, 'contract_diagnostics_type_mismatch_counts': False, 'contract_diagnostics_type_mismatch_rates': False}`

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added required per-key type contract + type/shape mismatch counts and rates to replay summary. |
| `tools/schemas/replay_validation_summary.schema.json` | Added required schema fields for type contract metadata and mismatch metrics. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertions covering type contract presence and zero mismatch metrics on canonical fixture. |
| `automation/reports/fast-run-20260306-1515-contract-diag-type-coverage.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1515-contract-diag-type-coverage.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> all required keys present, all `contract_diagnostics_type_mismatch_counts` values `0`
- `python3 tools/replay_verification_assess.py` -> default raw fixture reports all mismatch rates `0.0`
- `python3 -m pytest -q` -> `375 passed in 7.21s`
- Post-change type-coverage probe:
  - `python3 -c "... run_assessment ... keys=[...] ..."`
  - result: `{'contract_diagnostics_required_key_type_contract': True, 'contract_diagnostics_type_mismatch_counts': True, 'contract_diagnostics_type_mismatch_rates': True}`

## Before/after evidence

- **Before:** replay summary only exposed diagnostics key presence/missing metrics; probe confirmed type-contract keys were absent.
- **After:** replay summary now includes `contract_diagnostics_required_key_type_contract`, `contract_diagnostics_type_mismatch_counts`, and `contract_diagnostics_type_mismatch_rates`, with current canonical fixtures showing zero mismatches.

## Unresolved risks

- Type/shape checks improve diagnostics coverage but do not yet enforce deeper semantic value constraints (for example, numeric ranges or monotonic relationships).
- Type contract is currently maintained in-code; future contract expansions require synchronized updates to this mapping and schema.

## Stop reason

Stopped after fixing the selected rank-5 instrumentation gap with minimal bounded edits and repeated validation; further expansion into semantic-quality scoring would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
