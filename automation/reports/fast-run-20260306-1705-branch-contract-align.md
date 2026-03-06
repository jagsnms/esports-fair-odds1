# Promotion report

branch: fast/run-20260306-1705-branch-contract-align
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** DOC-CONTRACT-FAST-LANE
- **Title:** Automation operator docs used outdated run-branch and report filename conventions that conflicted with the active fast-lane contract

## Why it outranked alternatives

Issue ranking from current repository evidence:

1. Structural invariant violations: none (`python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`).
2. Failing canonical tests: none (`python3 -m pytest -q` -> `375 passed`).
3. Confirmed replay mismatches: none (raw-contract replay assessment clean, no unknown mode points).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).
5. Missing instrumentation blocking diagnosis: no current blocking gap found in canonical replay assessment output.
6. Calibration weaknesses: no new bounded calibration failure signal from canonical checks.

With ranks 1-6 clear, the highest unresolved bounded issue was rank 7 (cleanup/documentation): explicit run-contract drift in operator docs.

## Baseline evidence

- Baseline branch: `agent-base` (before run branch creation)
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.76s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract-gap evidence (`rg "agent/run-|report-YYYYMMDD-issue-slug\\.md" /workspace/automation`):
    - `automation/README.md` contained `agent/run-YYYYMMDD-issue-slug`
    - `automation/FIRST_RUN_OPERATOR_GUIDE.md` contained `agent/run-YYYYMMDD-issue-slug`
    - `automation/FIRST_RUN_OPERATOR_GUIDE.md` contained `report-YYYYMMDD-issue-slug.md`

## Files changed

| Path | Change |
|------|--------|
| `automation/README.md` | Updated run-branch naming to `fast/run-YYYYMMDD-HHMM-short-slug` and refreshed example. |
| `automation/FIRST_RUN_OPERATOR_GUIDE.md` | Updated required run-branch naming and report filename convention to branch-stem-based `.md` + `.json` artifacts. |
| `automation/reports/fast-run-20260306-1705-branch-contract-align.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1705-branch-contract-align.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q` -> `375 passed in 7.31s`
- `python3 tools/replay_verification_assess.py` -> replay diagnostics remain clean (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`)
- `rg "agent/run-YYYYMMDD-issue-slug|report-YYYYMMDD-issue-slug\\.md" /workspace/automation/README.md /workspace/automation/FIRST_RUN_OPERATOR_GUIDE.md` -> no matches
- `rg "fast/run-YYYYMMDD-HHMM-short-slug|automation/reports/<branch-name-with-slashes-replaced-by-hyphens>\\.(md|json)" /workspace/automation` -> expected matches in updated docs and templates

## Before/after evidence

- **Before:** `automation/README.md` and `automation/FIRST_RUN_OPERATOR_GUIDE.md` instructed `agent/run-YYYYMMDD-issue-slug`; guide also instructed legacy `report-YYYYMMDD-issue-slug.md`.
- **After:** both docs now align with fast-lane run branch naming (`fast/run-YYYYMMDD-HHMM-short-slug`) and branch-stem report artifact naming for both markdown and JSON.

## Unresolved risks

- Other historical documents (outside the two edited operator docs) may still contain legacy branch examples and could require a separate, explicitly scoped documentation harmonization run.

## Stop reason

Stopped after fixing the selected single rank-7 contract-drift issue with minimal doc-only edits and validation; additional broad documentation harmonization would exceed this bounded run.

## Recommendation

- `promote`
