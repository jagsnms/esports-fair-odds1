# Initiative proposal

Review-ready: yes

## Required stale-proposal prevention checklist

- [x] Checked `automation/PROMOTED_INITIATIVES.md`
- [x] Checked `automation/BANKED_INITIATIVES.md`
- [x] Checked shared/origin truth (`origin/agent-base` and `origin/agent-initiative-base`)
- [x] Confirmed this proposal is **not already promoted/shared truth**
- [x] Confirmed this proposal is not banked/deferred without new evidence

Review-ready marker contract:
- Set exactly `Review-ready: yes` only when this artifact is ready for formal review.
- Validator enforcement applies only when that exact marker is present.

### Check results (required)

- Promoted registry findings: No promoted initiative covers an ENGINE_SPEC diagnostics payload parity validator (no entries for `diagnostics_payload_required_fields`, `round_state_vector`, `movement_confidence_params`, or `violation_reason_codes`).
- Banked registry findings: Only `Replay + Simulation Validation Architecture` is banked; this proposal does not reopen that architecture and remains a thin validator/reporting patch.
- Shared/origin truth findings: `origin/agent-base`=`097132aa1e5bd621556fc78099bdcf45234d176c`; `origin/agent-initiative-base`=`d3b22a9f1f5378a9fde2aa0433ab2333867b93e2`; current tip has promotion-packet assembler/validator and evidence-gate runner, but no SPEC diagnostics parity gate.
- Non-duplication confirmation: Current replay assessment required keys differ from ENGINE_SPEC required diagnostics fields; missing exact SPEC fields are `round_state_vector`, `q`, `p_hat`, `movement_confidence_params`, `timer_state`, `violation_reason_codes`, so this is unresolved and not duplicate work.

## Initiative title

ENGINE_SPEC diagnostics payload contract alignment (thin validator-first)

## Why it outranks other major issues

Using the Bible ranking ladder:

1. Structural invariant violations: not present in current replay assessment (`structural_violations_total=0`).
2. Failing canonical tests: not present (`466 passed`).
3. Confirmed replay mismatches: not currently confirmed by latest replay verification evidence (no replay mismatch confirmation; zero behavioral/invariant totals in current assessment artifacts).
4. High-frequency diagnostic invariant failures: not present (`behavioral_violations_total=0`, `invariant_violations_total=0`).
5. Missing instrumentation blocking diagnosis: present. Current diagnostics completeness checks are keyed to local assessment fields, but there is no explicit validator that enforces the ENGINE_SPEC diagnostics payload contract (`invariants.diagnostics_payload_required_fields`).

Therefore this is the highest-ranked unresolved issue class and the next most urgent bounded patch.

## Why it exceeds bounded-fix scope

This is initiative-lane work because it spans policy contract alignment across tooling and governance outputs (assessment summary schema, validator tooling, and stage evidence expectations), not a one-line isolated bug. It needs staged delivery to keep implementation deterministic and auditable without broad engine rewrites.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `docs/ENGINE_SPEC.json` | Canonical source for required diagnostics payload fields. |
| `tools/replay_verification_assess.py` | Current diagnostics completeness key set and summary output producer. |
| `tools/schemas/replay_validation_summary.schema.json` | Machine-readable replay summary contract to extend/check. |
| `tools/` (new thin validator script) | New read-only SPEC parity validator for diagnostics payload fields. |
| `tests/unit/` (new/updated validator tests) | Deterministic contract tests for pass/fail parity behavior. |
| `automation/reports/` | Stage report and machine-readable validator output artifact. |

## Proposed stages

1. **Stage 1 (next patch, narrow and deterministic):**
   - Add a thin read-only validator that compares ENGINE_SPEC `diagnostics_payload_required_fields` against replay assessment diagnostics key coverage and emits machine-readable pass/fail + missing field list.
   - Add unit tests for: full pass, partial miss, empty diagnostics case.
   - No compute, rail, calibration, replay-policy, or runtime engine behavior changes.
2. **Stage 2 (only if Stage 1 approved and merged):**
   - Add optional canonical alias mapping report (current key names -> SPEC field names) to make drift localization explicit in output artifacts.
3. **Stage 3 (only if Stage 2 shows sustained parity gaps):**
   - Wire the parity result into promotion packet completeness checks as a non-bypassable evidence field for initiative-lane packets.

## Validation checkpoints

- Stage 1:
  - `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` (regression safety)
  - `python3 -m pytest -q tests/unit/test_validate_engine_spec_diagnostics_contract.py` (new thin-validator tests)
  - `python3 tools/replay_verification_assess.py` + validator run on produced summary (expected deterministic output and explicit missing/present accounting)
- Stage 2:
  - Golden-output test for alias mapping report stability.
- Stage 3:
  - Unit tests for promotion packet validator integration and explicit failure messaging when parity evidence is absent.

## Risks

- **False precision risk:** Exact-name parity may fail when semantically-equivalent fields exist under different names; mitigate with explicit alias mapping report before any hard fail gate expansion.
- **Scope creep risk:** Pressure to modify engine diagnostics payload directly; mitigate by stage contract that Stage 1 is validator/reporting only.
- **Governance coupling risk:** Over-tight gating could block promotion packets prematurely; mitigate by introducing parity as explicit evidence first, then tightening only after review.

## Recommended branch plan

- Keep initiative lane discipline: proposal/decision first, then one approved stage at a time on `agent-initiative-base`.
- For implementation stage runs, keep one commit per stage with explicit report artifacts and no promotion/reconvergence unless explicitly requested.
- Preserve strict no-side-work policy and avoid touching banked Replay+Simulation architecture unless reopen criteria are met.

## Recommendation / disposition

- [ ] **Approve planning only** — accept proposal; do not start implementation
- [x] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
