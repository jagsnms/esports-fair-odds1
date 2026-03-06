branch: deep/proposal-20260306-2203-rail-v2-activation-contract
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

**Rail V2 Activation Contract Completion (Replay/Source Carryover Backbone)**

## Why it outranks other major issues

Current canonical evidence shows rail semantic switch logic exists but is effectively inactive on replay validation inputs:

- `python3 tools/replay_verification_assess.py` (default raw fixture): `rail_input_v2_activated_points=0`, reason `V2_REQUIRED_FIELDS_MISSING`.
- `run_assessment("tools/fixtures/replay_multimatch_small_v1.jsonl")`: `6/6` points fallback with `V2_REQUIRED_FIELDS_MISSING`.
- `tools/fixtures/*.jsonl` currently contain no `player_states` fields, so `bo3_snapshot_to_frame(...)` emits `cash_totals/loadout_totals/armor_totals=None`.

This is a higher-value blocker than adjacent alternatives because it prevents validation of Chapter 6 Step 6 dynamic rail semantics in the canonical replay path. The rail semantics change exists in code, but trusted evidence cannot exercise it.

## Bible progression justification

### Direct Bible mismatch addressed vs prerequisite blocker

This initiative is a **true prerequisite blocker removal** for Bible alignment, not a cosmetic cleanup.

- Bible Chapter 6 Step 6 requires **dynamic rails** driven by allowed carryover signals.
- Bible Chapter 9 defines replay validation as primary evidence.
- Current replay evidence path cannot activate v2 carryover semantics due missing required carryover inputs, so Chapter 6 Step 6 behavior is not verifiable in canonical replay validation.

### Specific next Bible-facing step unlocked

Unlocks the next concrete step: **Stage validation of dynamic carryover rail behavior on canonical replay fixtures**, including measurable v2 activation coverage and carryover-directionality checks under replay.

### Why this outranks PHAT / rails / calibration / replay alternatives right now

- **PHAT movement redesign (Chapter 4/6 Step 8):** important, but calibrating movement before rail endpoint semantics are activatable in replay risks tuning toward fallback-v1 endpoint behavior rather than intended carryover-aware endpoints.
- **Further rail formula redesign:** not selected because the immediate blocker is not formula absence; it is missing carryover input availability across replay/source contracts.
- **Calibration campaign:** premature while a core structural rail signal path cannot be exercised in replay evidence.
- **Replay+simulation architecture refresh (banked):** remains relevant but deferred; this narrower blocker should be resolved first so current replay framework can validate intended rail semantics.

### Why this is not subsystem drift, momentum bias, or safe local continuation

Although adjacent to recent rail-stage work, this is not local refinement:

- scope crosses replay fixtures, ingest normalization contracts, runner/source wiring, and replay validation schema/tests;
- evidence shows global activation failure (`v2_activated_points=0`) in canonical replay assessment;
- without this, recently introduced rail-v2 semantics remain mostly theoretical in replay validation.

### What remains blocked or unreliable if skipped

If skipped, the project remains unable to claim replay-backed validation of carryover-driven rail dynamics, leaving Bible Step 6 alignment unproven and making subsequent movement/calibration changes causally ambiguous.

## Why it exceeds bounded-fix scope

This is cross-module architecture work, not a single bounded patch:

1. source/fixture contract definition (what carryover fields are required and when),
2. normalize/ingest behavior for partial vs complete carryover payloads,
3. runner replay contract/observability alignment,
4. replay assessment schema and gating updates,
5. parity validation across BO3/GRID/REPLAY raw sources.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/normalize/bo3_normalize.py` | Emits `cash_totals/loadout_totals/armor_totals`; currently `None` when `player_states` absent. |
| `backend/services/runner.py` | Replay raw path orchestration and debug/provenance emission. |
| `engine/compute/rails_cs2.py` | v2 required-field contract and activation/fallback reasoning already implemented; target consumer of improved inputs. |
| `tools/replay_verification_assess.py` | Canonical replay evidence entrypoint; should enforce activation/coverage checkpoints once data is available. |
| `tools/schemas/replay_validation_summary.schema.json` | Needs stage-gated keys/threshold fields for activation quality evidence. |
| `tools/fixtures/raw_replay_sample.jsonl` | Current raw fixture class lacks carryover-complete fields. |
| `tools/fixtures/replay_multimatch_small_v1.jsonl` | Current multi-match fixture lacks carryover-complete fields. |
| `tests/unit/test_bo3_normalize_microstate_fields.py` | Contract tests for carryover field emission and missingness semantics. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Replay summary contract tests; extend for rail-v2 activation evidence gates. |
| `tests/unit/test_runner_source_contract_parity.py` | Source-parity assertions for carryover contract completeness and policy determinism. |

## Proposed stages

1. **Stage 0 — Contract freeze + baseline evidence**
   - Freeze required carryover field contract per source/replay mode.
   - Record baseline activation/fallback rates over canonical fixtures and available replay logs.

2. **Stage 1 — Replay fixture/data-contract completion**
   - Add bounded canonical replay fixture class with carryover-complete payloads (including `player_states` or equivalent mapped inputs).
   - Keep existing sparse fixtures for fallback-path coverage.

3. **Stage 2 — Ingest/runner parity hardening**
   - Ensure BO3/GRID/REPLAY raw produce deterministic required-field validity classification.
   - Wire source metadata into rails provenance uniformly where missing.

4. **Stage 3 — Validation gate upgrade**
   - Extend replay assessment/schema/tests with explicit activation evidence (e.g., non-zero v2 activations on carryover-complete fixture class, deterministic fallback reasons on sparse class).
   - Preserve structural invariant zero-violation requirement.

5. **Stage 4 — Promotion checkpoint**
   - Produce before/after replay evidence pack proving rail-v2 path is executable and diagnosable in canonical replay validation.

## Validation checkpoints

- Canonical tests remain green: `python3 -m pytest -q`.
- Replay assessment determinism preserved for each fixture class.
- Carryover-complete fixture class demonstrates `rail_input_v2_activated_points > 0`.
- Sparse fixture class continues deterministic fallback (`V2_REQUIRED_FIELDS_MISSING`) without structural regressions.
- No increase in `structural_violations_total` or invariant violation totals.
- BO3/GRID/REPLAY parity tests confirm consistent required-field classification and reason-code determinism.

## Risks

- **Data realism risk:** synthetic carryover-complete fixtures may not match live provider sparsity distribution.
- **Over-inference risk:** estimating carryover inputs from weak proxies could create false precision.
- **Contract drift risk:** source-specific handling could diverge again without strict parity tests.
- **Scope risk:** could expand into broad replay architecture redesign if stage boundaries are not enforced.

## Recommended branch plan

- Keep this run as planning-only on `deep/proposal-20260306-2203-rail-v2-activation-contract`.
- If approved, execute one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-rail-v2-activation-s0`
  - `deep/stage-YYYYMMDD-HHMM-rail-v2-activation-s1`
  - `deep/stage-YYYYMMDD-HHMM-rail-v2-activation-s2`
  - etc.
- Require stage reports with explicit before/after activation and invariant evidence.
- No self-merge; human promotion only.

## Recommendation

- **approve planning only**

## Deferred alternatives (banked/rejected this run)

- **Replay + Simulation Validation Architecture** remains banked in `automation/BANKED_INITIATIVES.md`; still a valuable future track, but this run proposes the narrower immediate blocker removal needed before broader replay/simulation expansion.
