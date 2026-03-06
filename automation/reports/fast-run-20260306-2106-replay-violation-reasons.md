# Promotion report

branch: fast/run-20260306-2106-replay-violation-reasons
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing replay-summary violation reason-code distributions
- **Title:** `tools/replay_verification_assess.py` reported only aggregate violation totals, not per-reason counts needed for fast diagnosis when invariants fail

## Why it outranked alternatives

Issue ranking from current evidence on `agent-base`:

1. Structural invariant violations: none (`structural_violations_total=0`).
2. Failing canonical tests: none (`python3 -m pytest -q` -> `379 passed`).
3. Confirmed replay mismatches: none (raw-contract replay assessment clean).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): replay summary lacked reason-code distributions for structural/behavioral/invariant violations, limiting root-cause localization if failures appear.

## Baseline evidence

- Baseline branch: `fast/run-20260306-2106-replay-violation-reasons` (created from `agent-base`)
- Baseline commit: `3a871b09bf26f9afefec73fb2fdad16118138ea8`
- Baseline checks:
  - `python3 -m pytest -q` -> `379 passed in 7.31s`
  - `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Reason-count key probe:
    - `python3 - <<'PY' ... run_assessment(...) ... PY`
    - result: `{'structural_violation_reason_counts': False, 'behavioral_violation_reason_counts': False, 'invariant_violation_reason_counts': False}`

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added per-reason aggregation maps for structural, behavioral, and invariant violations in replay summary output. |
| `tools/schemas/replay_validation_summary.schema.json` | Added new required schema properties for the three violation reason-count maps. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertions for new reason-count fields and total-to-sum consistency checks. |
| `automation/reports/fast-run-20260306-2106-replay-violation-reasons.md` | Added human-readable run report artifact. |
| `automation/reports/fast-run-20260306-2106-replay-violation-reasons.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> new fields present: `structural_violation_reason_counts`, `behavioral_violation_reason_counts`, `invariant_violation_reason_counts`
- `python3 -m pytest -q` -> `379 passed in 7.35s`
- `python3 tools/replay_verification_assess.py` -> canonical replay summary includes the new reason-count fields and remains clean

## Before/after evidence

- **Before:** replay summary exposed only aggregate violation totals; reason-count fields were absent.
- **After:** replay summary now includes three reason-code distribution maps (`structural_violation_reason_counts`, `behavioral_violation_reason_counts`, `invariant_violation_reason_counts`), enabling direct attribution of future violation spikes.

## Unresolved risks

- Current canonical fixtures have zero violations, so new reason maps are presently empty; non-empty-path behavior depends on future violation-bearing fixtures.
- Reason maps count string reason codes as emitted; upstream naming consistency of reason codes remains an external dependency.

## Stop reason

Stopped after fixing the selected rank-5 instrumentation gap with minimal bounded edits and repeated validation; further expansion into broader calibration analytics would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
