branch: deep/proposal-20260306-2055-rail-carryover-semantic-switch
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Rail Carryover Semantic Switch (v2): migrate contract rail endpoints from score-only v1 to Bible-aligned carryover-conditioned semantics.

## Why it outranks alternatives

This is the highest-value unresolved major issue because there is direct, explicit evidence that rail endpoint semantics are intentionally locked to a non-final mode:

- `engine/compute/rails_cs2.py` sets:
  - `RAIL_INPUT_V2_POLICY = "observe_v2_use_v1_endpoints"`
  - `rail_input_v1_fallback_used = True` on every evaluation
  - `rail_input_active_endpoint_semantics = "v1"`
  - fallback reason code includes `STAGE1_LOCKED_NO_SEMANTIC_SWITCH`
- `tests/unit/test_rails_input_contract.py` asserts this lock as expected Stage 1 behavior (`test_v2_fallback_always_used`, `test_v2_fallback_reason_stage1_locked_when_required_complete`).

So the repo currently confirms a planned-but-unfinished architecture migration in a Bible-critical subsystem (rails), not a speculative cleanup.

Why not other major alternatives right now:

- **PHAT behavior:** recently advanced through staged deep work (`initiative-20260306-phat-coupling-realignment-stage-2/3/3b/4c.md`) with no confirmed current mismatch in available replay assessment evidence.
- **Replay/simulation validation architecture:** explicitly banked in `automation/BANKED_INITIATIVES.md` with reopen conditions; no new evidence in this run shows those reopen conditions have clearly activated above this rail mismatch.
- **Calibration-first initiative:** calibration quality work depends on endpoint semantics being meaningfully Bible-aligned; calibrating against known Stage-1-locked rail semantics risks optimizing the wrong target.

Deferred alternative (banked): Replay + Simulation Validation Architecture remains relevant, but is deferred per banked policy unless reopen conditions are met.

## Bible progression justification

### Direct Bible mismatch addressed

This initiative directly reduces a Bible mismatch (not just tooling debt):

- Bible Ch.2/Ch.3/Ch.6 Step 6 require rails to represent post-round carryover state and be recomputed from carryover-relevant signals.
- Current contract rail semantics are still v1 score/prematch driven and explicitly not switched to carryover-conditioned v2 endpoints.
- The code itself marks this as transitional observability-only behavior, confirming the mismatch is known and active.

### Specific next Bible-facing step this unlocks

Completing this initiative unlocks trustworthy Bible-facing replay validation of rail behavior:

1. verify rails move with carryover context (economy/loadout/armor persistence),
2. then run replay and scenario checks to measure invariant behavior under the correct endpoint semantics,
3. then perform any needed calibration with confidence that Step 6 semantics are structurally correct.

Without this, downstream replay/calibration results remain structurally ambiguous because the endpoint model is knowingly transitional.

### Why this outranks PHAT / rails alternatives / calibration / replay now

- It is a **direct structural blocker** for Bible Step 6 fidelity.
- PHAT movement work has progressed with bounded stage evidence already; rails endpoint semantics remain intentionally stage-locked.
- Replay/simulation architecture expansion is banked and should not be re-proposed without reopen evidence.
- Calibration before endpoint semantic completion risks fitting to pre-migration artifacts.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

Although adjacent to recent rail Stage 1 observability work, this is not "more of the same" instrumentation:

- Stage 1 explicitly stopped before semantic migration; this proposal is the first architecture step that actually removes that locked semantic gap.
- The target is Bible-critical engine behavior (rail endpoint definition), not local refactor polish.
- The proposal is chosen because the code/test evidence shows an intentional, unresolved architecture boundary, not because the subsystem is convenient.

### What remains blocked or unreliable if skipped

If skipped:

- Bible-consistent rail endpoint semantics remain unresolved.
- Replay validation can still pass while testing a transitional endpoint contract.
- Calibration and invariants trend analysis on rails will remain partially non-actionable due to semantic mismatch risk.

## Why it exceeds bounded-fix scope

This exceeds bounded maintenance because it requires cross-module semantic migration with staged safeguards:

- endpoint semantics redesign in `engine/compute/rails_cs2.py`,
- potential support updates in map-level probability shaping (`engine/compute/map_fair_cs2.py`) and carryover feature interpretation,
- source-ingest and runner parity checks to ensure BO3/GRID/replay provide required carryover fields consistently,
- replay assessment/schema/test contract updates to validate the new semantics and avoid silent regressions.

This is multi-stage architecture work, not a one-file bug fix.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/compute/rails_cs2.py` | Primary semantic migration from v1 endpoints to v2 carryover-conditioned contract endpoints. |
| `engine/compute/map_fair_cs2.py` | Candidate map probability adjustments to ensure endpoint derivation can consume carryover context while preserving invariants. |
| `engine/models.py` | Verify/lock carryover field contracts used by rails v2 (types/default behavior). |
| `engine/ingest/grid_reducer.py` | Ensure GRID path emits carryover fields required by v2 consistently. |
| `backend/services/runner.py` | Ensure source/replay metadata and diagnostics expose active rail contract semantics and fallback reasons correctly. |
| `tools/replay_verification_assess.py` | Add/extend rail-contract migration diagnostics and parity counters for replay validation. |
| `tools/schemas/replay_validation_summary.schema.json` | Schema evolution for new rail migration evidence keys. |
| `tests/unit/test_rails_input_contract.py` | Convert Stage-1 lock assertions into stage-gated semantic-switch assertions. |
| `tests/unit/test_rails_cs2_basic.py` | Preserve structural rails behavior and add carryover-responsiveness expectations. |
| `tests/unit/test_runner_source_contract_parity.py` | Confirm cross-source parity for rail contract diagnostics after semantic switch. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Validate summary artifact includes required rail semantic migration evidence. |

## Proposed stages

1. **Stage 0 (planning + evidence gate)**
   - Freeze explicit acceptance criteria for semantic switch.
   - Define required carryover inputs and exact fallback policy for missing/invalid fields.
   - Produce baseline replay/fixture evidence pack before code migration.

2. **Stage 1 (contract semantic switch in rails)**
   - Implement v2 endpoint semantics in `compute_rails_cs2` behind explicit stage policy.
   - Keep deterministic fallback and diagnostics for partial data cases.
   - Preserve structural invariants and map-point boundary behavior.

3. **Stage 2 (source parity + ingestion hardening)**
   - Ensure BO3/GRID/replay raw inputs produce consistent required carryover fields.
   - Eliminate source-specific silent semantic drift in rail endpoint computation.

4. **Stage 3 (replay validation + diagnostics contract uplift)**
   - Extend replay assessment outputs and schema for rail semantic conformance.
   - Add replay/scenario checks targeting carryover sensitivity and forbidden-transient invariance.

5. **Stage 4 (calibration readiness checkpoint, not full calibration campaign)**
   - Confirm rails are stable and Bible-aligned enough to authorize subsequent calibration-focused initiatives.

## Validation checkpoints

- **Structural hard checks:** `rail_low <= rail_high`, rails within bounds/[0,1], map-point alignment deltas near zero.
- **Carryover sensitivity checks:** changing allowed carryover signals (cash/loadout/armor/persistent equipment proxies) changes endpoints as expected.
- **Transient invariance checks:** HP/alive/timer/position perturbations do not directly alter contract endpoint semantics.
- **Cross-source parity checks:** BO3 vs GRID vs replay raw produce deterministic contract classification and comparable endpoint behavior for matched carryover state.
- **Replay artifact checks:** replay assessment reports stage policy, fallback counts, and semantic-activation evidence with schema conformance.

## Risks

- **Semantic shock risk:** switching endpoint semantics may shift PHAT trajectories and expose hidden assumptions in downstream expectations.
- **Data quality risk:** carryover fields can be missing/noisy by source, requiring explicit fallback policy and observability.
- **Parity risk:** BO3/GRID/replay may encode carryover context differently, causing false regression signals without normalization discipline.
- **Calibration coupling risk:** premature calibration edits during semantic migration could confound causal attribution.

## Recommended branch plan

- Planning branch (this run):  
  `deep/proposal-20260306-2055-rail-carryover-semantic-switch`
- If approved, implement one stage per branch:
  - `deep/stage-<timestamp>-rail-carryover-semantic-switch-s1`
  - `deep/stage-<timestamp>-rail-carryover-semantic-switch-s2`
  - `deep/stage-<timestamp>-rail-carryover-semantic-switch-s3`
- Do not batch multiple stages in one branch.
- Promote only through human review; no self-merge.

## Recommendation

- [x] **Approve planning only** — lock stage acceptance criteria and evidence gates before semantic migration.
- [ ] **Approve stage 1**
- [ ] **Defer**
