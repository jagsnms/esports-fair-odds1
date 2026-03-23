branch: deep/proposal-20260323-1710-replay-v2-validation-gate
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Replay/simulation validation trust gate for rail-v2 semantics

## Why it outranks alternatives

Current evidence shows a validation trust gap in the Bible-primary replay/simulation path, not a compute-path failure:

- Canonical tests are green (`528 passed`), with no current structural test failures.
- Replay assessments show zero structural/behavioral/invariant violations on fixtures, but rail-v2 activation is absent on most canonical replay fixtures:
  - `raw_replay_sample.jsonl`: v2 `0/3`, fallback `3/3` (`V2_REQUIRED_FIELDS_MISSING`)
  - `replay_multimatch_small_v1.jsonl`: v2 `0/6`, fallback `6/6`
  - `replay_non_tiny_canonical_v1.jsonl`: v2 `0/24`, fallback `24/24`
  - `replay_non_tiny_negative_control_v1.jsonl`: v2 `0/24`, fallback `24/24`
  - only `replay_carryover_complete_v1.jsonl` activates v2 (`3/3`)
- The replay-vs-simulation pilot can return `decision=pass` on a non-tiny canonical replay slice even when rail-v2 activation is `0%`.

This outranks fresh PHAT/rail/calibration redesign work right now because those efforts would rely on validation outcomes that can still produce false confidence while not exercising promoted v2 carryover semantics on representative replay slices.

## Bible progression justification

### Direct mismatch or prerequisite blocker

This initiative is a **true prerequisite blocker removal** (not a direct compute mismatch rewrite):

- Chapter 6 requires dynamic rail computation each tick with carryover semantics.
- Chapter 9 defines replay validation as primary and simulation as required.
- Current pass/fail artifacts can pass without proving that replay evidence exercised rail-v2 semantics on non-trivial slices.

### Specific next Bible-facing step unlocked

If completed, this unlocks trustworthy next-step Bible-facing work:

1. PHAT/rail behavior tuning against replay evidence that is demonstrably rail-v2-active where required.
2. Calibration/reliability campaigns (Brier/logloss/reliability) grounded in semantically-correct replay cohorts.
3. Re-open of broader replay/simulation architecture work with trustworthy gating signals.

### Why this outranks PHAT / rails / calibration / replay alternatives now

- **PHAT behavior:** no current structural or behavioral failure evidence demanding immediate PHAT formula/movement redesign.
- **Rails compute redesign:** major rail-v2 semantic switch is already promoted; the current blocker is trust in validation coverage of that semantics.
- **Calibration:** calibration quality work is lower-value while validation can pass without v2 activation on representative replay inputs.
- **Replay+simulation architecture (banked):** this was banked, but the reopen condition is now met by new evidence that simulation/replay pass criteria can be satisfied while rail-v2 semantics are not exercised on key fixtures.

### Why this is not subsystem drift or momentum bias

Recent deep work touched replay/rails, so adjacency risk is real. This selection is still justified because it addresses a newly observed **cross-module trust defect** (assessment + pilot decision contract + fixture coverage + promotion gating), not local cleanup in one module.

### What remains blocked if skipped

If skipped, future Bible-facing improvements remain unreliable:

- replay/simulation pass signals may continue to mask missing rail-v2 semantic exercise,
- calibration campaigns may optimize against fallback-dominated evidence,
- promotion decisions can remain under-constrained relative to Chapter 6/9 intent.

## Why it exceeds bounded-fix scope

This is not a single-file maintenance fix. It requires coordinated redesign across replay assessment contracts, pilot decision logic, fixture/corpus policy, and promotion gating. The required changes span multiple modules and validation layers and need staged rollout with explicit evidence checkpoints.

## Stale-proposal prevention check record

- Promoted registry (`automation/PROMOTED_INITIATIVES.md`): rail-v2 activation backbone and replay carryover provenance/completeness Stage 1 are promoted, but no promoted stage enforces a trust gate that blocks replay/simulation `pass` when representative replay cohorts are fallback-only for rail-v2 semantics.
- Banked registry (`automation/BANKED_INITIATIVES.md`): `Replay + Simulation Validation Architecture` is banked; reopen conditions include “new evidence makes simulation the immediate blocker.” This condition is now met by current pilot/pass evidence under 0% v2 activation on non-tiny canonical replay.
- Shared/origin truth:
  - `origin/agent-base`: `097132aa1e5bd621556fc78099bdcf45234d176c`
  - `origin/agent-initiative-base`: `d6f08286fd76325be0fcbc55d7e3262b0d7ddd3c`
  - Current shared truth does not contain a hard validation gate coupling replay/simulation pass eligibility to rail-v2 activation coverage thresholds on canonical non-tiny slices.
- Non-duplication confirmation: this proposal does not re-propose already promoted rail-v2 compute activation; it proposes validation trust-gating that is currently missing.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `tools/replay_verification_assess.py` | Emit cohort-level v2 activation coverage summaries suitable for hard gating. |
| `tools/run_replay_simulation_validation_pilot.py` | Add pass-eligibility contract requiring minimum rail-v2 activation evidence for designated replay cohorts. |
| `tools/schemas/replay_validation_summary.schema.json` | Extend machine-readable schema for new trust-gate fields. |
| `tools/schemas/` (pilot decision schema area) | Freeze/validate new trust-gate decision fields. |
| `tools/validate_promotion_packet.py` and promotion packet assemblers | Require trust-gate evidence in initiative-lane promotion packets. |
| `tools/fixtures/replay_non_tiny_canonical_v1.jsonl` and companion fixtures | Define canonical cohort policy and expected gate outcomes. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Add/adjust tests for v2 activation cohort evidence and gating fields. |
| `tests/unit/test_run_replay_simulation_validation_pilot.py` | Add pass/mismatch tests for trust-gate behavior. |
| `tests/unit/test_validate_promotion_packet.py` | Enforce promotion evidence presence and fail-closed behavior. |

## Proposed stages

1. **Stage 1 — Validation trust contract freeze**
   - Define canonical replay cohorts and required minimum rail-v2 activation evidence for `pass` eligibility.
   - Add explicit decision fields: trust-gate status, failure reason codes, and cohort coverage metrics.
   - No engine compute changes.
2. **Stage 2 — Replay assessment + pilot gating implementation**
   - Implement gated pass logic in assessment/pilot paths.
   - Ensure deterministic mismatch/inconclusive behavior when trust gate is not met.
   - Keep existing structural/behavioral checks intact.
3. **Stage 3 — Promotion integration**
   - Require trust-gate evidence in initiative-lane promotion packet validation.
   - Fail closed when evidence is missing/invalid.
4. **Stage 4 — Canonical fixture/corpus uplift plan**
   - Define migration plan for non-tiny canonical replay fixture coverage toward rail-v2 activation-ready cohorts.
   - Keep this stage as data/validation architecture work, not compute-model redesign.

## Validation checkpoints

- Checkpoint A (Stage 1 contract freeze):
  - Unit tests assert schema-required trust-gate fields and deterministic reason codes.
  - Backward-compatibility checks for existing summary readers.
- Checkpoint B (Stage 2 gating logic):
  - Pilot on `replay_non_tiny_canonical_v1.jsonl` must not return `pass` when trust-gate threshold is unmet.
  - Negative-control fixture must remain non-pass with coherent mismatch class.
  - Existing structural/invariant tests remain green.
- Checkpoint C (Stage 3 promotion integration):
  - Promotion packet validation fails when trust-gate evidence is absent.
  - Promotion packet validation passes with compliant trust-gate evidence.
- Checkpoint D (Stage 4 evidence quality):
  - Canonical cohort reports include activation-rate distribution and explicit fallback reasons.
  - Reproducibility checks for seeded runs remain deterministic.

## Risks

- **Over-gating risk:** gates that are too strict can stall useful diagnostics; mitigate with staged rollout and explicit `inconclusive` pathways.
- **Schema churn risk:** downstream tooling may break on new required fields; mitigate with contract versioning and compatibility tests.
- **Data-availability risk:** non-tiny fixtures may need uplift before strict thresholds can be met; mitigate with staged threshold policy and explicit corpus migration plan.
- **Scope creep risk:** pressure to patch compute behavior during gate rollout; mitigate by hard boundary: no compute semantics changes in this initiative.

## Recommended branch plan

- Keep this proposal branch as planning only.
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-replay-v2-validation-gate-s1`
  - `deep/stage-YYYYMMDD-HHMM-replay-v2-validation-gate-s2`
  - `deep/stage-YYYYMMDD-HHMM-replay-v2-validation-gate-s3`
  - `deep/stage-YYYYMMDD-HHMM-replay-v2-validation-gate-s4`
- Require report + evidence artifact per stage before any promotion decision.
- Do not merge directly; human promotion only.

## Recommendation / disposition

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later

