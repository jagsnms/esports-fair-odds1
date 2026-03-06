# Automation

## Purpose of agent-base

`agent-base` is the approved starting branch for all autonomous automation work. Every automation run must branch from `agent-base`—never from `main`, `master`, `dev`, `release`, or any human-owned feature branch.

## Run-branch naming

Create a fresh branch for each run using this format:

```
agent/run-YYYYMMDD-issue-slug
```

Example: `agent/run-20250305-add-validation`.

## Push and merge rules

- **Push:** Automations may only push to their own run branch. They may never write directly to `main`, `master`, `dev`, `release`, or human-owned feature branches.
- **Merge:** Automations may never merge their own work. Promotion to other branches is a human decision.

## Promotion report

Each run must produce a promotion report and leave it in the run branch (e.g. under `automation/reports/`). The report provides baseline evidence, files changed, validation performed, before/after results, unresolved risks, and stop reason. Use `automation/templates/promotion_report_template.md` as the starting point.
