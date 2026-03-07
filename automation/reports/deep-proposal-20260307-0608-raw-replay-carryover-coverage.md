branch: deep/proposal-20260307-0608-raw-replay-carryover-coverage
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

**Raw replay carryover completeness and contract-activation coverage architecture**

## Why it outranks other major issues

This is the highest-value unresolved issue because current evidence shows the engine is structurally healthy but still under-validates the Bible-required rail semantics on representative replay inputs:

- `python3 -m pytest -q` passes (`411 passed`), so there is no active structural test failure outranking it.
- Replay assessment on canonical raw fixtures still falls back to v1 rails due missing required carryover fields:
  - `tools/fixtures/raw_replay_sample.jsonl`: `rail_input_v2_activated_points=0`, `rail_input_v1_fallback_points=3`, reason `V2_REQUIRED_FIELDS_MISSING`.
  - `tools/fixtures/replay_multimatch_small_v1.jsonl`: `rail_input_v2_activated_points=0`, `rail_input_v1_fallback_points=6`, reason `V2_REQUIRED_FIELDS_MISSING`.
  - Only the bounded carryover-complete fixture activates (`replay_carryover_complete_v1.jsonl --prematch-map 0.55`: `rail_input_v2_activated_points=3`).
- Legacy point replay remains non-canonical under current policy (`logs/history_points.jsonl`: `point_like_inputs_seen=61`, `point_like_inputs_rejected=61`, `total_points_captured=0`), so it cannot validate v2 carryover rail behavior.

So the current blocker is not PHAT math correctness; it is **cross-source data-path completeness for Bible-aligned rail semantics in replay validation**.

Deferred alternative (banked): **Replay + Simulation Validation Architecture** remains banked and is not selected here; current replay-validation stages are not exhausted for canonical raw carryover coverage yet, so simulation-first reopening would be premature.

## Bible progression justification

### Direct Bible mismatch addressed, or exact prerequisite blocker removed

This initiative removes a concrete prerequisite blocker for Bible Chapter 2/3/6 Step 6 and Chapter 9 replay validation:

- Bible requires rails to be recomputed from carryover signals (economy/loadout/persistent equipment) and validated via replay.
- Current replay corpus frequently lacks required carryover fields, so many replay points validate fallback v1 semantics instead of intended v2 carryover semantics.

This is therefore a **true prerequisite removal**, not a semantic rewrite.

### Specific next Bible-facing step this initiative unlocks

It unlocks a trustworthy next step: **Stageable replay-driven calibration/validation on real raw-contract data where v2 carryover rails are actually active at meaningful coverage**, enabling reliable Chapter 8 calibration and Chapter 9 regression signals.

### Why this outranks PHAT / rails / calibration / replay alternatives right now

- **PHAT/timer redesign**: recent deep stages already aligned movement and timer contract behavior; tests and current replay diagnostics show no active structural/invariant crisis to outrank this.
- **Rail formula redesign**: Stage 1 semantic switch exists; the immediate bottleneck is activation coverage in real replay paths, not new endpoint formula work.
- **Calibration campaign now**: calibration on fallback-heavy replay data would optimize against incomplete rail semantics and produce low-trust results.
- **Replay+simulation architecture now**: banked; simulation does not remove the present raw-carryover evidence gap in canonical replay sources.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

Although adjacent to recent replay/rail work, selection is evidence-driven (activation/fallback coverage metrics), not momentum-driven:

- Majority of assessed canonical raw fixture points still run fallback semantics (`9/12` fallback across two standard fixtures).
- Non-canonical point corpus is fully rejected under default policy (`61/61`), leaving a real validation blind spot.

This is a cross-module architecture/data-contract issue (ingest + runner + fixtures + replay assessment + calibration handoff), not local cleanup.

### What remains blocked or unreliable if this is skipped

If skipped, the project remains unable to claim that replay validation meaningfully exercises Bible-aligned carryover rail semantics at scale. Any subsequent calibration, replay regression interpretation, or PHAT behavior conclusions remain partially untrusted due to fallback-dominated evidence.

## Why it exceeds bounded-fix scope

This cannot be solved by a single bounded bug fix. It requires coordinated changes across source ingestion contracts, replay corpus format/collection, runner wiring, validation metrics, and calibration input policy. The work is architectural because it defines and enforces a cross-source data contract with staged migration and acceptance gates.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/normalize/bo3_normalize.py` | Source of raw BO3 carryover field extraction (`cash_totals`, `loadout_totals`, `armor_totals`) and completeness signaling. |
| `engine/ingest/grid_reducer.py` | GRID-to-Frame carryover normalization parity and reliability semantics. |
| `backend/services/runner.py` | Replay mode handling, contract gate behavior, source/replay-kind tagging, and diagnostics propagation. |
| `engine/compute/rails.py` | Wrapper contract to pass source metadata to rail compute path (currently omits `source`/`replay_kind` plumbing). |
| `engine/compute/rails_cs2.py` | v2 activation/fallback policy and reason-code semantics; target consumer of improved input completeness. |
| `engine/replay/bo3_jsonl.py` | Replay payload format loading and classification baseline for migration tooling. |
| `tools/replay_verification_assess.py` | Canonical replay evidence and activation-rate measurement gate. |
| `tools/fixtures/*.jsonl` | Fixture classes for sparse, carryover-complete, and migration-path coverage. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Replay assessment contract assertions and future activation-threshold assertions. |
| `tests/unit/test_runner_replay_contract_mode.py` | Point/raw policy behavior and migration safety tests. |
| `tests/unit/test_rails_input_contract.py` | Rail v2 activation/fallback contract invariants across sources. |
| `tools/round_level_calibration.py` | Downstream calibration consumer currently tied to legacy point-history shape; handoff target once canonical raw coverage is available. |

## Proposed stages

1. **Stage 1 — Source provenance and completeness contract**
   - Plumb `source`/`replay_kind` end-to-end into rails debug from runner/compute wrapper.
   - Emit deterministic completeness diagnostics for required carryover inputs per source.
   - No rail formula or PHAT semantic changes.

2. **Stage 2 — Replay raw corpus uplift and migration tooling**
   - Add bounded tooling/format support to generate canonical raw-contract replay artifacts from available BO3 capture paths.
   - Expand fixture classes to include representative carryover-complete and partial-coverage cases.
   - Keep point-like default rejection policy unchanged unless explicitly approved.

3. **Stage 3 — Coverage gates and CI-quality replay evidence**
   - Add replay assessment acceptance gates (activation ratio, fallback reason distribution, invariant totals).
   - Require deterministic reporting of v2 activation coverage by fixture/source class.

4. **Stage 4 — Calibration handoff readiness**
   - Define and test bounded handoff from legacy point-history calibration inputs to canonical replay-derived inputs where labels are trustworthy.
   - Keep calibration weight tuning out of scope; this stage is data-contract readiness only.

## Validation checkpoints

- Full canonical unit suite remains green (`python3 -m pytest -q`).
- Replay assessment matrix required per stage:
  - sparse raw fixture(s): fallback remains deterministic with expected reason codes.
  - carryover-complete fixture(s): non-zero v2 activation required.
  - point-like history fixture(s): policy behavior explicit (rejected by default unless transition mode explicitly enabled).
- New stage gates should include:
  - minimum v2 activation coverage threshold on designated canonical raw classes,
  - zero structural invariant violations in assessment summaries,
  - deterministic reason-code distributions for fallback/activation.
- Cross-source parity checks (BO3/GRID/REPLAY raw) for equivalent required inputs must pass.

## Risks

- **Data availability risk:** real replay sources may not consistently provide required carryover fields; migration may need staged partial-coverage handling.
- **Policy coupling risk:** changing replay ingestion/corpus shape can unintentionally affect existing tooling expecting point-history format.
- **False-confidence risk:** raising activation on synthetic fixtures only could mask real-source completeness gaps.
- **Scope creep risk:** pressure to mix calibration retuning or rail formula redesign before data-contract readiness is achieved.

## Recommended branch plan

- Current run (proposal): `deep/proposal-20260307-0608-raw-replay-carryover-coverage`.
- If approved, execute one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-replay-carryover-provenance-s1`
  - `deep/stage-YYYYMMDD-HHMM-replay-raw-corpus-uplift-s2`
  - `deep/stage-YYYYMMDD-HHMM-replay-coverage-gates-s3`
  - `deep/stage-YYYYMMDD-HHMM-calibration-handoff-readiness-s4`
- Promote each stage only after explicit review and evidence pack sign-off; do not bundle multiple stages.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
