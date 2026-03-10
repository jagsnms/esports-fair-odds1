# Promotion report

## Selected issue

- **Issue:** missing machine-readable promotion report artifact (`automation/reports/*.json`)
- **Title:** Add canonical machine-readable promotion report template and policy wiring

## Why it outranked alternatives

Ranking used the required ladder. Current evidence showed no structural invariant violations, no failing canonical tests, no confirmed replay mismatches, and no high-frequency diagnostic invariant failures. The highest unresolved item was rank #5 (**missing instrumentation blocking diagnosis**): run reports were markdown-only despite Bible Chapter 11 requiring both machine-readable and human-readable outputs.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `46b9a27`
- Canonical tests: `python3 -m pytest -q` → `341 passed in 7.19s`
- Machine-readable report artifact count before changes:
  - `automation/reports/*.json` → `0` files

## Files changed

| Path | Change |
|------|--------|
| `automation/AUTOMATION_RUN_POLICY.md` | Updated report requirement to require markdown + JSON artifacts and point to both templates. |
| `automation/README.md` | Documented markdown + JSON promotion report artifacts for each run. |
| `automation/templates/promotion_report_template.json` | Added canonical machine-readable promotion report template with required fields. |
| `automation/reports/report-20260306-machine-readable-promotion-report.md` | Added this human-readable run report. |
| `automation/reports/report-20260306-machine-readable-promotion-report.json` | Added machine-readable run report artifact for this run. |

## Validation performed

- `python3 -m json.tool automation/templates/promotion_report_template.json`
- `python3 -m json.tool automation/reports/report-20260306-machine-readable-promotion-report.json`
- `python3 -m pytest -q`
- Artifact count check for machine-readable reports:
  - `automation/reports/*.json` count moved from `0` to `1`

## Before/after evidence

- Before: no machine-readable promotion reports existed (`automation/reports/*.json` count = 0), and automation docs/policy referenced only markdown report output.
- After: canonical JSON template exists, docs/policy explicitly require both markdown and JSON artifacts, and this run emits both report forms (`automation/reports/*.json` count = 1).

## Unresolved risks

- Existing historical reports remain markdown-only; backfilling old runs is out of this run’s scope.
- This run adds contract/templates and emits JSON for the current run; external tooling that consumes reports may still need implementation in future runs.

## Stop reason

Stopped after completing the selected single issue with bounded changes and issue-specific validation. Additional work (e.g., historical backfill or external ingestion tooling) would expand scope beyond this run.

## Recommendation

- [x] **Promote** — ready for human review/merge
- [ ] **Hold** — keep branch for later review
- [ ] **Discard** — do not promote
