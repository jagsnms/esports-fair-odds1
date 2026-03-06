branch: deep/proposal-20260306-2245-timer-pressure-contract
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Timer-pressure contract realignment (pre-plant/post-plant directionality + post-plant hard boundary)

## Why it outranks other major issues

This is the highest-value unresolved Bible-facing issue because current runtime q/PHAT behavior still lacks explicit timer directionality semantics and the hard post-plant boundary required by Chapters 5, 6, and 7:

- `engine/compute/midround_v2_cs2.py` applies time as a scalar urgency multiplier (`urgency = floor + scale * time_progress`) but does not encode a pre-plant CT-favoring timer force versus post-plant T-favoring timer force.
- No engine compute path currently implements an explicit "if post-plant time remaining < defuse time then CT win probability = 0" rule.
- `engine/diagnostics/invariants.py` currently checks q bounds, rail order, and movement-gap diagnostics, but does not include timer-directionality or defuse-boundary diagnostics.
- Current unit coverage validates bomb-side term usage and generic monotonicity, but does not assert Bible timer-directionality behavior or the post-plant hard boundary contract.

By comparison, recent deep work has already delivered bounded Stage-1 progress on replay/rail contract surfaces with clean structural outcomes (implemented stage reports show deterministic counters and no structural regressions). Those areas remain important, but they are currently less urgent than this direct Bible mismatch in core model behavior.

Deferred alternative (banked): replay + simulation validation architecture remains banked in `automation/BANKED_INITIATIVES.md` and does not currently meet reopen conditions.

## Bible progression justification

### Direct Bible mismatch addressed vs prerequisite blocker

This initiative directly reduces an active Bible mismatch:

- Chapter 5/6 requires timer pressure to change direction by state (pre-plant vs post-plant), not only increase generic confidence.
- Chapter 5 specifies a hard post-plant boundary (`time_remaining < defuse_time => CT probability = 0`).
- Chapter 7 defines timer directionality as a behavioral invariant requiring diagnostic visibility.

The current implementation is missing these semantics as first-class modeled behavior and as contract diagnostics/tests.

### Specific next Bible-facing step unlocked

Completing this initiative unlocks trustworthy Bible-facing replay validation of timer behavior:

1. Run canonical replay + scenario validation where timer-directionality and defuse-boundary checks are measurable and enforced.
2. Then perform calibration work on timer curve/weights against replay outcomes without calibrating the wrong timer semantics.

Without this, replay/calibration can only optimize a semantically incomplete timer model.

### Why this outranks PHAT / rails / calibration / replay right now

- **PHAT:** this is PHAT-core behavior (q and movement target driver), not peripheral cleanup.
- **Rails:** rail input contract activation/fallback stages have recent implemented evidence and bounded observability already in place; timer semantics remain a stronger direct Bible mismatch.
- **Calibration:** calibrating before timer semantics are correct risks fitting incorrect directional behavior.
- **Replay architecture:** major replay architecture work is banked and explicitly deferred; timer semantics are an active engine-behavior blocker now.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

This proposal intentionally avoids recent rail/replay momentum and selects a different subsystem (core q/timer semantics + invariants/tests) because it has higher Bible misalignment. It is not a cosmetic extension of recent deep-stage rail work.

### What remains blocked/unreliable if skipped

If skipped:

- q/PHAT behavior will continue lacking required timer-direction semantics and hard post-plant boundary guarantees.
- Replay and scenario validation cannot credibly claim Bible timer alignment.
- Calibration and model-weight tuning remain structurally underconstrained and may overfit incorrect timer behavior.

## Why it exceeds bounded-fix scope

This exceeds bounded maintenance scope because it requires coordinated cross-module contract changes, not a localized patch:

1. q/timer semantic redesign in compute modules.
2. Diagnostic/invariant contract expansion for timer-direction and boundary failures.
3. Replay/scenario validation harness and schema updates for new timer diagnostics.
4. Canonical test suite expansion across compute, diagnostics, and replay assessment paths.

A single-file fix would not provide the necessary architecture-level consistency or validation chain.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/compute/midround_v2_cs2.py` | Introduce explicit timer-direction contribution and post-plant defuse-boundary logic in q pathway. |
| `engine/compute/q_intra_cs2.py` | Align legacy/debug q computation contract or deprecate/divert to avoid contradictory timer semantics. |
| `engine/compute/resolve.py` | Ensure target/movement diagnostics report timer-direction and boundary-driven behavior consistently. |
| `engine/diagnostics/invariants.py` | Add timer-directionality and defuse-boundary behavioral diagnostics (testing-visible, non-silent). |
| `tools/replay_verification_assess.py` | Aggregate/report timer-direction and boundary violation counters in replay evidence artifacts. |
| `tools/schemas/replay_validation_summary.schema.json` | Extend schema for timer-direction/boundary diagnostics fields. |
| `tests/unit/test_midround_v2_cs2.py` | Add pre-plant/post-plant directionality and defuse-boundary tests. |
| `tests/unit/test_resolve_micro_adj.py` | Validate end-to-end resolve behavior under timer-direction scenarios. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Add invariant-level timer diagnostics assertions. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Assert deterministic summary support for new timer diagnostics counters. |

## Proposed stages

1. **Stage 0 (contract design + evidence lock)**
   - Define explicit timer-direction contract and defuse-boundary semantics.
   - Lock deterministic reason codes and diagnostics payload keys.
   - Produce fixture/scenario matrix for pre-plant and post-plant timing edges.

2. **Stage 1 (compute semantics)**
   - Implement timer-direction force in midround/q pathway.
   - Implement hard post-plant defuse boundary in q contract (CT zero-probability boundary path).
   - Preserve existing structural invariants and movement formula contract.

3. **Stage 2 (diagnostics + replay artifact contract)**
   - Add invariant diagnostics for timer-direction and boundary violations.
   - Add replay summary counters/reason-code rollups and schema updates.
   - Keep diagnostics testing-visible (no silent masking/clamping).

4. **Stage 3 (validation + regression gate)**
   - Add scenario tests and replay checks for directionality and defuse boundary.
   - Establish before/after metrics: violation counts, trajectory deltas, and stability checks.
   - Gate promotion on zero structural regressions and deterministic artifact outputs.

## Validation checkpoints

- **Contract checkpoint:** deterministic diagnostic keys + reason codes present for every evaluated point.
- **Directionality checkpoint:** pre-plant timer decrease shifts q toward CT; post-plant timer decrease shifts q toward T in controlled scenarios.
- **Hard-boundary checkpoint:** post-plant `time_remaining < defuse_time` yields CT round-win probability effectively 0 in q pathway.
- **Invariant checkpoint:** no new structural violations (`q` bounds, rail ordering); timer behavioral violations tracked in testing mode.
- **Replay checkpoint:** replay summary contains timer-direction and boundary counters with stable schema conformance and deterministic output.
- **Regression checkpoint:** no unintended degradation in existing rail contract tests and replay contract-gate tests.

## Risks

- **Over-coupling risk:** introducing timer-direction terms can unintentionally overpower alive/hp/loadout signals if not bounded.
- **Semantic split risk:** if `q_intra_cs2` and midround-v2 semantics diverge, diagnostics can become internally inconsistent.
- **Boundary edge risk:** defuse-time handling may vary by telemetry precision/source; requires explicit normalization policy.
- **Replay comparability risk:** new diagnostics fields may require careful schema migration to preserve deterministic artifact consumers.

## Recommended branch plan

- Proposal branch (this run): `deep/proposal-20260306-2245-timer-pressure-contract`
- Approved implementation branches (one stage per run):
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s0-contract`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s1-compute`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s2-diag-replay`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s3-validation`

Rules:
- Start each stage from `agent-initiative-base` (or explicitly approved promoted predecessor).
- Keep one primary issue class per stage.
- Do not merge from automation; human promotion only.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start broad implementation in this run.
- [ ] **Approve stage 1** — approve first implementation stage.
- [ ] **Defer** — do not approve now.
