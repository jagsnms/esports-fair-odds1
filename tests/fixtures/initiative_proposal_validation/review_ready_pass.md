# Initiative proposal

Review-ready: yes

## Required stale-proposal prevention checklist

- [x] Checked `automation/PROMOTED_INITIATIVES.md`
- [x] Checked `automation/BANKED_INITIATIVES.md`
- [x] Checked shared/origin truth (`origin/agent-base` and `origin/agent-initiative-base`)
- [x] Confirmed this proposal is **not already promoted/shared truth**
- [x] Confirmed this proposal is not banked/deferred without new evidence

### Check results (required)

- Promoted registry findings: Stage 3A governance line already promoted; excluded from proposal.
- Banked registry findings: Replay + Simulation Validation Architecture remains deferred.
- Shared/origin truth findings: Checked origin initiative lanes for latest promoted hashes.
- Non-duplication confirmation: This proposal does not repeat promoted stages.

## Initiative title

Governance gate hardening

## Why it outranks other major issues

Prevents stale proposal churn and preserves review bandwidth.

## Why it exceeds bounded-fix scope

Touches initiative governance contract across policy + template + validation utility.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| automation/* | Governance-only enforcement |

## Proposed stages

1. Stage 0 contract freeze
2. Stage 1 minimal validator
3. Code-first review

## Validation checkpoints

Validator pass/fail fixtures and deterministic output checks.

## Risks

Overly strict syntax checks can reject valid human wording variants.

## Recommended branch plan

Use `agent-initiative-base` only until explicit promotion request.

## Recommendation / disposition

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
