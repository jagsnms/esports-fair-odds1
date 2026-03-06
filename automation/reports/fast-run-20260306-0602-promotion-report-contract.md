# Promotion report

branch: fast/run-20260306-0602-promotion-report-contract
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** missing required run metadata fields in promotion report templates
- **Title:** Promotion report template contract gap for branch/lane/status metadata

## Why it outranked alternatives

Current run evidence showed no unresolved rank 1-4 issues:
- Structural invariants: no violations in replay diagnostics (`structural_violations_total=0`).
- Failing canonical tests: `360 passed`.
- Confirmed replay mismatches: replay assessment completed with no mismatch signal.
- High-frequency diagnostic invariant failures: replay diagnostic totals were `0`.

With rank 1-4 clear, the highest unresolved bounded issue was rank 5 (**missing instrumentation blocking diagnosis**): report templates did not encode required run metadata fields (`branch`, `base_branch`, `lane`, `run_type`, `status`, `recommendation`) needed for consistent automation traceability.

## Baseline evidence

- Baseline branch before edits: `fast/run-20260306-0602-promotion-report-contract` (from `agent-base`)
- Baseline commit: `a73bdde960c9ea2d7625b9a163c7fe27f8a285f2`
- Baseline checks:
  - `python3 -m pytest -q` → `360 passed in 7.75s`
  - `python3 tools/replay_verification_assess.py` → `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract-gap proof:
    - metadata search in markdown template returned no matches for required top fields.
    - JSON template lacked top-level `base_branch`, `lane`, `run_type`, and `status`.

## Files changed

| Path | Change |
|------|--------|
| `automation/templates/promotion_report_template.md` | Added required top metadata block and explicit recommendation value contract. |
| `automation/templates/promotion_report_template.json` | Added top-level run metadata fields and allowed-value arrays for `status` and `recommendation`. |
| `automation/reports/fast-run-20260306-0602-promotion-report-contract.md` | Added this human-readable run report. |
| `automation/reports/fast-run-20260306-0602-promotion-report-contract.json` | Added machine-readable run report artifact for this run. |

## Validation performed

- `python3 -m json.tool automation/templates/promotion_report_template.json` → pass
- `python3 -m pytest -q` → `360 passed in 7.23s`
- `python3 tools/replay_verification_assess.py` → no structural/behavioral/invariant violations
- Post-change contract checks:
  - markdown template now includes required metadata keys at top.
  - JSON template now includes `branch`, `base_branch`, `lane`, `run_type`, `status`, `recommendation`.

## Before/after evidence

- **Before:** promotion report templates did not include the required run metadata header contract, creating inconsistent run artifacts.
- **After:** both markdown and JSON templates include the required run metadata contract and constrained values, enabling consistent reporting across fast maintenance runs.

## Unresolved risks

- Older historical reports remain on previous schema; this run only updates canonical templates and the current run artifacts.
- `_status_allowed_values` / `_recommendation_allowed_values` are convention fields, not enforced by a JSON Schema validator yet.

## Stop reason

Stopped after the selected single rank-5 issue was fixed with minimal changes and validated; additional work (historical backfill or schema validator tooling) would be out of bounded scope for this run.

## Recommendation

- `promote`
