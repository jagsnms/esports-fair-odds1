# Initiative run policy

Operational policy for the major-initiative planning lane on agent-initiative-base.

## Scope

This automation is for **major issues that exceed bounded-fix scope**: cross-module work, architectural changes, PHAT redesign, rail redesign, replay architecture changes, and similar initiatives. It is not for single-file or single-module fixes that belong on the maintenance lane (agent-base).

## Authority order

When sources conflict, prefer in this order:

1. **Bible** — ENGINE_BIBLE and ENGINE_SPEC in this repo
2. **Canonical repo/test paths** — checked-in tests and canonical fixtures
3. **Current evidence** — logs, runs, and live observations
4. **Prior reports** — initiative proposals and promotion reports
5. **Memory** — session or prior run context (advisory only)

## One major issue or one approved stage per run

Each run should identify **one** highest-value major issue, or execute **one** approved initiative stage. Do not batch multiple initiatives or stages into a single run.

## Default: proposal, not broad implementation

If no initiative has been explicitly approved for implementation, **default behavior is proposal/report generation**, not broad code changes. Produce initiative proposals that scope the work, break it into stages, and document risks and branch strategy. Do not perform large-scale implementation unless a specific initiative stage has been approved.

## Proposal requirements

Proposals must include:

- **Impacted modules** — which parts of the repo are affected
- **Stage breakdown** — ordered phases or stages for the initiative
- **Validation checkpoints** — how each stage will be validated
- **Risks** — technical and rollout risks
- **Recommended branch strategy** — how work should be branched and integrated

Use `automation/templates/initiative_proposal_template.md` as the starting point.

## Memory is advisory only

Current repo evidence (Bible, tests, logs, policy) overrides memory when they conflict.

## Never merge its own work

This automation may never merge its own work. Promotion and integration are human decisions.
