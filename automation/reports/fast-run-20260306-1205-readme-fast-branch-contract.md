# Promotion report

branch: fast/run-20260306-1205-readme-fast-branch-contract
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Automation branch-name contract mismatch in `automation/README.md`
- **Title:** README still prescribed `agent/run-*` naming while active fast-lane report/template contract uses `fast/run-YYYYMMDD-HHMM-short-slug`

## Why it outranked alternatives

Current evidence on `agent-base` showed no unresolved rank 1-6 issues:

1. Structural invariant violations: none (`python3 tools/replay_verification_assess.py` reported `structural_violations_total=0`).
2. Failing canonical tests: none (`python3 -m pytest -q` passed with `369 passed`).
3. Confirmed replay mismatches: none (`raw_contract_points` only, `unknown_replay_mode_points=0`, `non_canonical_point_points=0`).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).
5. Missing instrumentation blocking diagnosis: no active blocker found in current replay diagnostics summary (required keys present with 1.0 presence rates).
6. Calibration weaknesses: no newly evidenced bounded calibration defect in this run.

With ranks 1-6 clear, the highest unresolved bounded issue was rank 7 documentation/contract consistency: `automation/README.md` conflicted with active fast-lane branch/report contract.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `624c1bf087b6a5903b73bc90418366c0ad766df8`
- Baseline checks:
  - `python3 -m pytest -q` -> `369 passed in 7.67s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract mismatch probe:
    - `rg "agent/run-YYYYMMDD-issue-slug" automation/README.md` -> match present
    - `rg "fast/run-YYYYMMDD-HHMM-short-slug" automation/templates/promotion_report_template.md` -> match present

## Files changed

| Path | Change |
|------|--------|
| `automation/README.md` | Updated run-branch naming format and example from `agent/run-*` to `fast/run-YYYYMMDD-HHMM-short-slug` to match active fast maintenance contract. |
| `automation/reports/fast-run-20260306-1205-readme-fast-branch-contract.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1205-readme-fast-branch-contract.json` | Added machine-readable run report artifact. |

## Validation performed

- `rg "agent/run-YYYYMMDD-issue-slug" automation/README.md` -> no matches
- `rg "fast/run-YYYYMMDD-HHMM-short-slug" automation/README.md` -> 1 match
- `python3 -m pytest -q` -> `369 passed in 7.31s`
- `python3 tools/replay_verification_assess.py` -> replay diagnostics remained clean (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`)

## Before/after evidence

- **Before:** `automation/README.md` required `agent/run-YYYYMMDD-issue-slug`, conflicting with fast-lane branch/report template contract.
- **After:** `automation/README.md` now specifies `fast/run-YYYYMMDD-HHMM-short-slug` and aligned example naming.

## Unresolved risks

- Other non-canonical docs still reference `agent/run-*`; this run intentionally scoped to the selected README contract mismatch only.
- This is a documentation-contract fix and does not enforce branch naming through code-level validation.

## Stop reason

Stopped after fixing the selected rank-7 documentation contract mismatch with the smallest viable change set and confirming no regressions in canonical tests/replay checks.

## Recommendation

- `promote`
