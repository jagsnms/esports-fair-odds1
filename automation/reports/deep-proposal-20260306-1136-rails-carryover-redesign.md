branch: deep/proposal-20260306-1136-rails-carryover-redesign
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Carryover-faithful dynamic rail endpoint redesign (contract rails)

## Why it outranks alternatives

Selected major issue: **contract rail semantics are structurally underpowered versus Bible Chapters 2/3/6**.

Current evidence from canonical code:
- `engine/compute/rails_cs2.py` returns contract rails from score-only counterfactual endpoints (`cf_low/cf_high`) with minimal post-processing, while richer round-quality signals are kept only in heuristic debug rails and are not used as contract outputs.
- `engine/compute/map_fair_cs2.py` is intentionally score-driven and explicitly excludes economy/persistent carryover modeling.
- Result: rail endpoints do not encode the Bible-required next-round carryover quality differences (e.g., stronger endpoint after high-survivor/economy-preserving wins).

Why this outranks other major candidates right now:
1. **PHAT movement confidence shaping** is still important, but movement sits on top of target rails; improving movement before rail semantics risks polishing motion toward structurally incorrect endpoints.
2. **Replay/simulation architecture** remains valuable but is currently banked in `automation/BANKED_INITIATIVES.md` unless reopen conditions are met. It is noted as a deferred alternative, not selected.
3. **Calibration work** cannot be trusted while rails omit core carryover semantics; calibration would absorb structural error rather than fix model identity mismatch.

Deferred banked alternative (explicit): Replay + Simulation Validation Architecture remains relevant but deferred per banked policy until its reopen conditions are met.

## Bible progression justification

### Direct Bible mismatch addressed

This initiative directly targets Bible mismatch:
- Chapter 2: rails must depend on **carryover-persistent** signals across round boundary.
- Chapter 3: rail endpoints should reflect round-win quality differences (survivors/economy carryover).
- Chapter 6 Step 6: rails must be dynamic round-terminal endpoints recomputed from current context.

Current contract rails are effectively score-only and therefore insufficiently expressive for these requirements.

### Specific next Bible-facing step this unlocks

It unlocks a trustworthy next step: **Bible-aligned PHAT movement/timer behavior validation against meaningful rail endpoints** (Chapter 4 + Chapter 9 replay validation), instead of validating motion against rails that underrepresent carryover state.

### Why this outranks PHAT / rails / calibration / replay alternatives now

- Outranks PHAT movement retuning: target endpoint semantics (rails) are upstream of movement confidence shaping.
- Outranks calibration: parameter fitting on structurally incomplete rails creates false calibration gains.
- Outranks replay/simulation architecture now: replay infrastructure stages are active and simulation architecture is banked; meanwhile this is an immediate direct Bible mismatch in core engine semantics.

### Why this is not subsystem drift or momentum bias

Recent approved deep implementation work is replay-contract infrastructure (`deep/stage-20260306-0714...`, `0818...`, `0950...`). This selection intentionally shifts to core model semantics, not replay-lane continuation. It is chosen for direct Bible mismatch severity, not local convenience.

### What remains blocked or unreliable if skipped

If skipped:
- PHAT trajectory validation remains partially untrustworthy because endpoint semantics underrepresent carryover context.
- Timer/movement improvements can still converge to endpoints with incorrect structural meaning.
- Calibration metrics risk optimizing to a mis-specified rail target, reducing interpretability and Bible fidelity.

## Why it exceeds bounded-fix scope

This is not a single-file fix. It requires coordinated, staged architecture work across:
- rail endpoint construction semantics,
- carryover feature contract (what is allowed/forbidden),
- source normalization parity (BO3/GRID/replay raw),
- diagnostic and replay validation artifacts,
- acceptance testing for rail integrity and monotonic behavior.

Any narrow patch would either violate Chapter 6/9 validation requirements or create source-specific drift.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `engine/compute/rails_cs2.py` | Primary rail endpoint contract redesign and carryover-only endpoint computation. |
| `engine/compute/map_fair_cs2.py` | Candidate endpoint primitive updates or separation of score baseline vs carryover adjustment. |
| `engine/models.py` | Formal carryover feature fields/flags contract (if new persistent signals are required). |
| `engine/ingest/grid_reducer.py` | GRID-side normalization parity for carryover fields used by rails. |
| `backend/services/runner.py` | Cross-source wiring and diagnostics surfacing for rail contract fields. |
| `engine/diagnostics/invariants.py` | Rail integrity diagnostics (carryover-only contract checks, endpoint sanity diagnostics). |
| `tools/replay_verification_assess.py` | Replay evidence reporting for rail contract metrics and drift counters. |
| `tests/unit/test_rails_cs2_basic.py` | Extend/replace for carryover-sensitive contract rails behavior. |
| `tests/unit/test_runner_source_contract_parity.py` | Cross-source parity assertions for rail contract inputs/outputs. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Schema/summary extension for rail-contract evidence fields. |

## Proposed stages

1. **Stage 1 — Rail input contract and diagnostics baseline**
   - Define explicit contract for allowed carryover rail inputs (and forbidden transient inputs).
   - Add diagnostics proving which fields influenced contract rail endpoints each tick.
   - No large behavior rewrite yet; establish observability and parity checks.

2. **Stage 2 — Contract rail endpoint computation redesign**
   - Implement carryover-aware endpoint computation for `rail_low/rail_high` while preserving structural invariants.
   - Keep strict separation between contract rails and optional heuristic/debug outputs.

3. **Stage 3 — Cross-source parity and replay validation hardening**
   - Ensure BO3, GRID, and replay raw produce equivalent carryover inputs and rail semantics.
   - Extend replay assessment summary with rail-contract metrics and mismatch counters.

4. **Stage 4 — Calibration gate (post-structure)**
   - Only after stages 1-3 are stable, evaluate whether movement/calibration retuning is required.
   - Do not blend structural rail redesign with parameter optimization in the same stage.

## Validation checkpoints

- **Checkpoint A (Stage 1):** unit tests verify diagnostics expose rail input provenance and carryover-only contract classification.
- **Checkpoint B (Stage 2):** structural invariant tests pass (`rail_low <= rail_high`, bounds, map-point boundary behavior) plus new carryover-sensitivity tests.
- **Checkpoint C (Stage 3):** parity tests across BO3/GRID/replay raw; replay assessment emits deterministic rail-contract summary fields.
- **Checkpoint D (Stage 4 gate):** compare before/after replay reliability metrics and invariant violation rates before allowing any calibration retune.

## Risks

- **Semantic overreach risk:** accidental use of transient signals (HP/timer/position) in contract rails.
- **Source parity risk:** BO3 vs GRID carryover field availability mismatch can produce inconsistent rails.
- **Regression risk:** endpoint changes can shift PHAT trajectories materially; requires strict staged rollout and replay diff checks.
- **Calibration confounding risk:** premature tuning could mask structural rail defects.

## Recommended branch plan

- Keep this proposal branch as planning artifact only:
  - `deep/proposal-20260306-1136-rails-carryover-redesign`
- For approved implementation, use one bounded branch per stage:
  - `deep/stage-YYYYMMDD-HHMM-rails-carryover-s1-contract`
  - `deep/stage-YYYYMMDD-HHMM-rails-carryover-s2-endpoints`
  - `deep/stage-YYYYMMDD-HHMM-rails-carryover-s3-parity-replay`
  - `deep/stage-YYYYMMDD-HHMM-rails-carryover-s4-calibration-gate`
- Require explicit human approval between stages; no multi-stage batching.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
