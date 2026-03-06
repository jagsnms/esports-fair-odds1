# First-run operator guide

Concise setup and verification guide for Cursor Automation runs against this repo.

## Required starting branch

Every run must start from **agent-base**. Create the run branch from it; do not run from `main`, `master`, `dev`, `release`, or any human-owned feature branch.

## Required run branch naming

Use this format for each run:

```
agent/run-YYYYMMDD-issue-slug
```

Example: `agent/run-20250305-fix-replay-mismatch`.

## Policy files to read before each run

Read these in order before starting work:

1. **docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md** — canonical engine and domain spec
2. **automation/AUTOMATION_RUN_POLICY.md** — issue ranking, authority order, run rules
3. **automation/README.md** — branch purpose, push/merge rules, report requirement
4. **automation/templates/promotion_report_template.md** — report structure to fill

## Required per-run behavior

- **Rank** candidate issues using the fixed ladder in AUTOMATION_RUN_POLICY.md (structural invariants → failing tests → replay mismatches → … → cleanup/docs).
- **Choose** exactly one primary issue per run.
- **Gather** baseline evidence (branch, commit, tests, logs) before making edits.
- **Make** the smallest viable change that addresses the primary issue.
- **Validate** repeatedly in sandbox (canonical tests, replays, checks as applicable).
- **Stop** at diminishing returns (issue resolved, no clear progress, or limit reached).
- **Write** a promotion report to `automation/reports/` using the template.

## Report filename convention

Save each run’s report as:

```
report-YYYYMMDD-issue-slug.md
```

Example: `report-20250305-fix-replay-mismatch.md`. Place it under `automation/reports/`.

## Memory vs repo evidence

Memory (session or prior-run context) is advisory only. Current repo evidence—tests, logs, Bible, policy files—wins when it conflicts with memory.
