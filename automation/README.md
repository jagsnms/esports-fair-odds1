# Automation

Run policy: [AUTOMATION_RUN_POLICY.md](AUTOMATION_RUN_POLICY.md).

## Branch-discipline summary

- **Maintenance/bounded-fix runs** must start from `agent-base`.
- **Initiative proposal/planning runs** must start from `agent-initiative-base`.
- **Initiative implementation/review runs for approved initiative stages** must start from `agent-initiative-base`, unless an explicit human promotion instruction says otherwise.

For operational branch-start rules, `automation/FIRST_RUN_OPERATOR_GUIDE.md` is authoritative.
For initiative-lane governance rules, `automation/INITIATIVE_RUN_POLICY.md` is lane-specific authority.

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

Each run must produce promotion report artifacts and leave them in the run branch (e.g. under `automation/reports/`):

- human-readable markdown via `automation/templates/promotion_report_template.md`
- machine-readable JSON via `automation/templates/promotion_report_template.json`

Reports provide baseline evidence, files changed, validation performed, before/after results, unresolved risks, and stop reason.
