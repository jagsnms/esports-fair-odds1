# Initiative run policy

Operational policy for the major-initiative planning lane on agent-initiative-base.
This document is lane-specific authority for initiative rules.
Initiative proposal/planning runs and approved-stage implementation/review runs must start from `agent-initiative-base`, unless an explicit human promotion instruction says otherwise.

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

## Mandatory stale-proposal prevention checks (before proposing/planning)

Before drafting any new proposal, runs must explicitly verify all three:

1. **Promoted initiatives registry**: `automation/PROMOTED_INITIATIVES.md`
2. **Banked initiatives registry**: `automation/BANKED_INITIATIVES.md`
3. **Shared/origin branch truth**: current promoted state on shared branches (including `origin/agent-base` and `origin/agent-initiative-base`)

If the candidate initiative/stage is already promoted/shared truth, it must **not** be re-proposed.
If the candidate initiative/stage is banked/deferred, it must not be re-proposed unless new evidence justifies reopening.

## Proposal requirements

Proposals must include:

- **Impacted modules** — which parts of the repo are affected
- **Stage breakdown** — ordered phases or stages for the initiative
- **Validation checkpoints** — how each stage will be validated
- **Risks** — technical and rollout risks
- **Recommended branch strategy** — how work should be branched and integrated
- **Stale-proposal check record** — explicit confirmation of promoted-registry, banked-registry, and shared/origin checks

## Review-ready marker + thin validation gate

The initiative proposal validator applies only to artifacts marked with the exact marker:

- `Review-ready: yes`

Unmarked or differently marked artifacts are treated as draft/not-applicable and are non-blocking.
Validator scope is syntactic contract completeness only; it does not perform semantic novelty checks or automated git/origin truth verification.

## Preferred initiative workflow (simplified)

Use this sequence for initiative work:

1. proposal
2. decision memo / contract freeze
3. implementation
4. code-first promotion review
5. revise if needed
6. promote
7. reconverge

Use `automation/templates/initiative_proposal_template.md` as the starting point.

## Memory is advisory only

Current repo evidence (Bible, tests, logs, policy) overrides memory when they conflict.

## Never merge its own work

This automation may never merge its own work. Promotion and integration are human decisions.
