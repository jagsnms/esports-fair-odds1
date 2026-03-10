# Automation run policy

Operating instructions for autonomous runs. Authority and rules below.

## Authority order

When sources conflict, prefer in this order:

1. **Bible** — ENGINE_BIBLE and ENGINE_SPEC in this repo
2. **Canonical repo/test paths** — checked-in tests and canonical fixtures
3. **Current evidence** — logs, runs, and live observations from this run
4. **Prior reports** — promotion reports in `automation/reports/`
5. **Memory** — session or prior run context (lowest authority)

## Issue ranking ladder

Select exactly one primary issue per run. Rank unresolved issues by this ladder (1 = highest):

1. **Structural invariant violations** — core design or data-structure invariants broken
2. **Failing canonical tests** — tests in the repo that are failing
3. **Confirmed replay mismatches** — replay vs expected outcome mismatches with clear evidence
4. **High-frequency diagnostic invariant failures** — repeated failures of diagnostic checks
5. **Missing instrumentation blocking diagnosis** — cannot diagnose because instrumentation is absent
6. **Calibration weaknesses** — model or threshold calibration issues
7. **Cleanup/documentation** — code/docs cleanup, no functional fix

Pick the highest-ranked unresolved issue. Do not skip a tier for a lower one unless the higher tier has no unresolved items.

## Exactly one primary issue per run

Each run has one primary issue. Work on that issue only. Do not batch multiple issues into one run.

## Non-reselection rule

Do not reselect an issue that has already been addressed (e.g. fixed and validated in a prior run or report) unless current evidence shows **regression** or **non-resolution**. If in doubt, prefer the next issue on the ladder.

## Baseline-before-edit rule

Before changing any code or config, capture baseline evidence: branch, commit, test results, and any relevant logs. Record this in the promotion report.

## Minimal-change rule

Make the smallest set of changes that address the primary issue. Avoid refactors or scope creep. Prefer targeted edits.

## Validation rule

Validate aggressively in sandbox (or equivalent): run canonical tests, relevant replays, and any checks that apply to the changed paths. Record what was run and the outcome in the promotion report.

## Diminishing-returns stop rule

Stop when:
- The primary issue is resolved and validated, or
- Further edits show diminishing returns (e.g. no clear progress, or new failures), or
- A time or step limit is reached.

Do not continue past clear diminishing returns. Document the stop reason in the report.

## Report requirement

Every run must leave both report artifacts in `automation/reports/`:

- A human-readable markdown promotion report using `automation/templates/promotion_report_template.md`
- A machine-readable JSON promotion report using `automation/templates/promotion_report_template.json`

Both artifacts must include: baseline evidence, files changed, validation performed, before/after evidence, unresolved risks, stop reason, and recommendation (promote / hold / discard).
