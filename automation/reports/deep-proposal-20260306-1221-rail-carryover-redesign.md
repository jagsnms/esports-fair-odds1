branch: deep/proposal-20260306-1221-rail-carryover-redesign
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Rail endpoint carryover redesign: replace score-only contract rails with round-terminal carryover-state rails aligned to Bible Chapters 2, 3, and 6.

## Why it outranks other major issues

Current repository evidence shows a direct architecture mismatch in rail semantics:

- Bible requires rails to represent post-round series outcomes based on carryover state (economy/loss-bonus/saved equipment/persistent gear).
- Current code explicitly defines contract rails using only `bounds`, `frame.scores`, `frame.series_score`, `frame.series_fmt`, and `config.prematch_map` (`engine/compute/rails_cs2.py`), while explicitly classifying micro/carryover-like fields as forbidden for contract rails.
- `engine/compute/map_fair_cs2.py` is score-driven and explicitly does not model econ/pistol/side/carryover context.
- `tests/unit/test_rails_input_contract.py` enforces invariance when loadout/cash/armor/timer/alive/hp are perturbed, confirming contract rails intentionally ignore those fields.

Competing major areas were ranked lower for this run:

- **PHAT behavior:** Recent deep stages already moved runtime to movement-form coupling and current replay diagnostics show no active structural/behavioral violation signal on canonical raw fixtures.
- **Replay architecture:** Replay contract gate/signaling stages are in place and currently enforce deterministic policy behavior; replay/simulation architecture is also banked in `automation/BANKED_INITIATIVES.md` unless reopen conditions are met.
- **Calibration:** Further calibration before rail semantic correction would tune against endpoints that are not Bible-aligned, reducing reliability of any gains.

Deferred alternative (banked): **Replay + Simulation Validation Architecture** remains relevant but is still deferred unless its reopen conditions are triggered.

## Bible progression justification

### Direct Bible mismatch or prerequisite

This initiative directly targets a Bible mismatch:

- Chapter 2/3/6 rail philosophy requires round-terminal carryover semantics.
- Present contract rails are score/prematch based and explicitly non-carryover.

Therefore this is a **direct mismatch reduction**, not merely a convenience refactor.

### Specific next Bible-facing step unlocked

Completing this initiative unlocks trustworthy Bible-facing validation/calibration steps:

1. Replay validation of rail behavior under realistic carryover transitions (saved guns, econ swings, loss-bonus progression).
2. Calibration of PHAT movement and q terms against endpoints that reflect intended post-round state semantics.
3. Meaningful comparison of rail diagnostics against Chapter 9 replay/invariant goals.

### Why this outranks PHAT / rails / calibration / replay alternatives now

- **Outranks PHAT confidence refinements:** movement tuning without endpoint semantic correctness optimizes toward the wrong target rails.
- **Outranks replay-system expansion:** replay architecture now has contract gating/visibility; the bigger blocker is that canonical rails being replayed are semantically under-specified vs Bible.
- **Outranks calibration pushes:** calibration on score-only rails risks encoding compensatory error into q/movement.
- **Outranks banked replay+simulation initiative:** banked status still applies; rail semantics are the more immediate Bible mismatch in active compute path.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

Although adjacent to recent rail contract observability work, this is not local continuation for convenience:

- Recent rail stage was explicitly observability-only and preserved existing endpoint behavior.
- New proposal changes the core endpoint semantics needed by Bible architecture, which is cross-module and not a cosmetic continuation.
- It is selected because it is the highest current Bible mismatch in canonical compute, not because rails were touched recently.

### What remains blocked if skipped

If skipped, the project remains blocked on:

- Bible-consistent interpretation of rail endpoints.
- Trustworthy replay-based rail validation for carryover transitions.
- Calibration confidence that separates q/movement errors from endpoint-definition errors.

## Why it exceeds bounded-fix scope

This cannot be solved as a bounded maintenance fix because it requires coordinated redesign across data contracts, compute semantics, diagnostics, replay fixtures, and tests:

- Introduces a new canonical carryover-state representation at round boundary.
- Changes rail endpoint computation contract and associated diagnostics.
- Requires cross-source ingestion alignment (BO3/GRID/REPLAY raw) for carryover inputs.
- Requires new replay/scenario validation assets and stage-gated rollout to prevent model drift.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/compute/rails_cs2.py` | Replace score-only rail contract with carryover-state endpoint model; versioned contract v2. |
| `engine/compute/map_fair_cs2.py` | Refactor/replace score-only map-fair endpoint helper to consume carryover-state features. |
| `engine/models.py` | Add canonical carryover-state fields required at round boundary. |
| `engine/normalize/bo3_normalize.py` | Populate carryover-relevant fields from BO3 snapshots where available. |
| `engine/ingest/grid_reducer.py` | Align GRID reduction output with the same carryover contract fields. |
| `backend/services/runner.py` | Ensure source parity and consistent rail-input provenance emission for BO3/GRID/REPLAY raw. |
| `engine/diagnostics/invariants.py` | Add rail contract diagnostics for carryover field presence/usage and mismatch reasons. |
| `tools/replay_verification_assess.py` | Extend replay assessment metrics for carryover rail coverage/quality. |
| `tools/fixtures/*.jsonl` | Add/upgrade fixtures containing round-terminal carryover transitions for validation. |
| `tests/unit/test_rails_input_contract.py` and related rail/replay parity tests | Replace v1 invariance assumptions with v2 carryover semantics and parity assertions. |

## Proposed stages

1. **Stage 1 — Carryover contract design + observability gate**
   - Define rail input contract v2 (allowed/required/optional carryover fields).
   - Add diagnostics showing v2 coverage and v1 fallback usage without semantic switch.

2. **Stage 2 — Data plumbing parity**
   - Implement BO3/GRID/REPLAY raw extraction/plumbing for v2 carryover fields.
   - Add source parity tests for field presence and normalization behavior.

3. **Stage 3 — Rail endpoint semantic migration**
   - Implement v2 carryover-based endpoint computation in rails path behind explicit stage gate.
   - Keep score-only logic as temporary fallback with side-by-side diagnostics.

4. **Stage 4 — Validation hardening**
   - Expand replay/scenario suites for carryover transitions.
   - Add acceptance thresholds for rail behavior consistency and invariant stability.

5. **Stage 5 — Default promotion + fallback retirement plan**
   - Promote v2 semantics as default once validation gates pass.
   - Restrict/deprecate score-only fallback with explicit policy window.

## Validation checkpoints

- **Checkpoint A (end Stage 1):** contract v2 schema and diagnostics present; no runtime rail semantic change.
- **Checkpoint B (end Stage 2):** BO3/GRID/REPLAY raw parity tests pass for carryover field population.
- **Checkpoint C (end Stage 3):** rail endpoint deltas vs v1 are measurable and explainable on controlled fixtures; no structural invariant regressions.
- **Checkpoint D (end Stage 4):** replay/scenario packs include carryover transition coverage with stable invariant metrics.
- **Checkpoint E (end Stage 5):** canonical unit suite green; replay assessment reports v2 coverage near-complete and fallback usage bounded.

Current baseline evidence used for ranking:

- `python3 -m pytest -q tests/unit` -> `375 passed`
- `python3 tools/replay_verification_assess.py` -> raw contract fixture clean (`0` structural/behavioral/invariant violations)
- `python3 tools/replay_verification_assess.py logs/history_points.jsonl` -> deterministic default rejection of point-like payloads (`POINT_REPLAY_REJECTED_DEFAULT_POLICY`)

## Risks

- **Data availability risk:** carryover fields may be sparse/inconsistent across sources; requires robust missingness policy.
- **Semantic regression risk:** endpoint migration can shift PHAT trajectories materially; needs staged gates and side-by-side telemetry.
- **Parity risk:** BO3/GRID/REPLAY raw may diverge in carryover extraction unless contracts are enforced at runner boundaries.
- **Calibration interaction risk:** existing fitted coefficients may partially compensate for old endpoint semantics.

## Recommended branch plan

- Keep this run as proposal only on `deep/proposal-20260306-1221-rail-carryover-redesign`.
- If approved, execute one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-rail-carryover-s1-contract-v2`
  - `deep/stage-YYYYMMDD-HHMM-rail-carryover-s2-data-parity`
  - `deep/stage-YYYYMMDD-HHMM-rail-carryover-s3-endpoint-migration`
  - `deep/stage-YYYYMMDD-HHMM-rail-carryover-s4-validation`
  - `deep/stage-YYYYMMDD-HHMM-rail-carryover-s5-promotion-gate`
- No self-merge; human promotion required after each stage report.

## Recommendation

- [x] **Approve planning only**
- [ ] **Approve stage 1**
- [ ] **Defer**
