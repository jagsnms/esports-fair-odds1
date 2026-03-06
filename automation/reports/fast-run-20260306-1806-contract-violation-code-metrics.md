# Promotion report

branch: fast/run-20260306-1806-contract-violation-code-metrics
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing per-code violation reason counters in replay verification summary
- **Title:** `tools/replay_verification_assess.py` emitted only aggregate violation totals, not reason-code distributions needed for diagnostics triage

## Why it outranked alternatives

Ranking on current `agent-base` evidence:

1. Structural invariant violations: none (`structural_violations_total=0` on replay assessment).
2. Failing canonical tests: none (`375 passed`).
3. Confirmed replay mismatches: none (`raw_contract_points` only; `unknown_replay_mode_points=0`).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): replay summary lacked per-code violation counters, so when violations do occur the output did not localize which contract/invariant code failed most often.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.93s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Missing-key probe:
    - `python3 -c "import asyncio; ... run_assessment('tools/fixtures/replay_multimatch_small_v1.jsonl') ..."`
    - result: `{'contract_diagnostics_structural_violation_code_counts': False, 'contract_diagnostics_behavioral_violation_code_counts': False, 'invariant_violation_code_counts': False}`

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added per-code counters for `contract_diagnostics` structural violations, `contract_diagnostics` behavioral violations, and `invariant_violations`; emitted them in summary output. |
| `tools/schemas/replay_validation_summary.schema.json` | Added the three violation-code counter fields as required schema properties. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertions for the new summary fields on the canonical fixture. |
| `automation/reports/fast-run-20260306-1806-contract-violation-code-metrics.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1806-contract-violation-code-metrics.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 tools/replay_verification_assess.py` -> summary now includes:
  - `contract_diagnostics_structural_violation_code_counts`
  - `contract_diagnostics_behavioral_violation_code_counts`
  - `invariant_violation_code_counts`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> same three fields present on multimatch fixture
- `python3 -m pytest -q` -> `375 passed in 7.32s`
- Post-change presence probe:
  - result: `{'contract_diagnostics_structural_violation_code_counts': True, 'contract_diagnostics_behavioral_violation_code_counts': True, 'invariant_violation_code_counts': True}`

## Before/after evidence

- **Before:** replay summary had only aggregate counts (`*_violations_total`) and did not expose per-code reason frequencies.
- **After:** replay summary includes explicit per-code violation counter maps for contract structural, contract behavioral, and invariant violations, enabling direct root-cause frequency triage.

## Unresolved risks

- Current fixtures produce zero violations, so non-empty counter distributions are not exercised in this run.
- Counters reflect emitted codes; semantic quality still depends on correctness of upstream violation code emission.

## Stop reason

Stopped after fixing the selected rank-5 diagnostics instrumentation gap with minimal bounded changes and repeated validation; expanding into synthetic failure generation would exceed single-issue maintenance scope.

## Recommendation

- `promote`
