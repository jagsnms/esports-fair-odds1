branch: deep/proposal-20260307-0827-replay-carryover-contract-coverage
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative proposal

## Initiative title

**Replay carryover contract coverage initiative (unlock rail-v2 validation and Bible-aligned replay evidence).**

## Why it outranks other major issues

Current evidence shows the engine is structurally healthy, but Bible-facing rail behavior is still under-validated on realistic replay classes because required carryover inputs are missing:

- `python3 -m pytest -q` -> **411 passed** (no broad structural/test failure forcing emergency PHAT/rail formula rewrites).
- `tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl` -> `rail_input_v2_activated_points=0/3`, reason `V2_REQUIRED_FIELDS_MISSING`.
- `tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> `rail_input_v2_activated_points=0/6`, reason `V2_REQUIRED_FIELDS_MISSING`.
- `tools/replay_verification_assess.py tools/fixtures/replay_carryover_complete_v1.jsonl --prematch-map 0.55` -> `rail_input_v2_activated_points=3/3`, reason `V2_STRICT_ACTIVATED`.

Interpretation: rail-v2 semantics work when carryover fields exist, but canonical sparse replay classes still remain fallback-only. This makes replay validation of Bible Chapter 6 Step 6 (dynamic rails from carryover state) unreliable on realistic ingestion classes.

Deferred alternative per banked registry:
- **Replay + Simulation Validation Architecture** remains banked. It is still relevant, but this carryover-coverage gap is the immediate blocker that must be removed before a replay+simulation architecture campaign can produce trustworthy rail-v2 validation data.

## Bible progression justification

### Direct Bible mismatch addressed, or exact prerequisite blocker being removed

This initiative removes a **true prerequisite blocker**: missing carryover inputs in replay ingestion/fixtures prevent consistent activation of carryover-aware rail-v2 semantics.  
Bible Chapter 2/3/6 requires rails to reflect carryover-only post-round state; fallback-only replay classes prevent that rule from being exercised broadly enough for reliable validation.

### Specific next Bible-facing step this initiative unlocks

It unlocks a bounded implementation stage where replay validation can enforce measurable rail-v2 activation coverage (not just synthetic spot checks), enabling trustworthy Bible-aligned rail-behavior verification before any calibration push.

### Why this outranks PHAT / rails / calibration / replay alternatives right now

- **PHAT behavior:** recent timer and movement stages are implemented; current tests are green and no new PHAT invariant failure signal is present.
- **Rail formula redesign:** not justified yet; current evidence indicates **data availability/coverage**, not immediate endpoint-formula failure.
- **Calibration campaign:** premature while replay classes are mostly fallback-only; calibration would optimize against incomplete rail-v2 exposure.
- **Replay+simulation architecture (banked):** still valuable, but larger architecture work should not proceed before this immediate replay carryover contract blocker is cleared.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

Although adjacent to recent rail/replay work, this is not cosmetic continuation:
- It targets the first currently measurable blocker preventing Bible-facing rail validation on realistic replay classes.
- It is evidence-driven (activation/fallback metrics), not based on convenience.
- It is cross-module (ingest normalization, replay runner contract, fixtures/assessment schema, validation gates), exceeding local cleanup scope.

### What would remain blocked or unreliable if this initiative were skipped

- Replay evidence would continue to over-represent v1 fallback semantics.
- Rail-v2 validation confidence would remain weak outside synthetic carryover-complete fixtures.
- Any next calibration or deeper replay/simulation initiative would start from an unreliable rail-input coverage baseline.

## Why it exceeds bounded-fix scope

This is not a single bug fix. It requires staged cross-module changes:

1. Contracting replay carryover data requirements across ingestion and fixtures.
2. Expanding replay corpus/collection to include required persistent fields.
3. Adding coverage metrics/gates so rail-v2 activation can be measured and enforced in validation.
4. Coordinating rollout without breaking existing replay sources or policy gates.

That is architecture/validation-lane work, not maintenance-lane bounded repair.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `engine/normalize/bo3_normalize.py` | Produces `cash_totals/loadout_totals/armor_totals`; currently `None` when replay snapshots lack `player_states`. |
| `engine/compute/rails_cs2.py` | Defines strict required rail-v2 fields and fallback/activation reason codes. |
| `backend/services/runner.py` | REPLAY raw-contract path and replay policy gate behavior; source for replay mode and contract diagnostics emission. |
| `tools/replay_verification_assess.py` | Canonical replay evidence aggregator; already emits activation/fallback counters to turn into hard coverage gates. |
| `tools/schemas/replay_validation_summary.schema.json` | Validation artifact schema; may need additions for stage-level coverage targets/reasons. |
| `tools/fixtures/raw_replay_sample.jsonl` | Sparse baseline fixture that currently demonstrates fallback-only behavior. |
| `tools/fixtures/replay_multimatch_small_v1.jsonl` | Sparse multi-match fallback baseline fixture. |
| `tools/fixtures/replay_carryover_complete_v1.jsonl` | Carryover-complete activation reference class; anchor for positive-control checks. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Existing deterministic fallback/activation contract tests; base for stronger stage gates. |
| `scripts/bo3_full_pull.py` (and replay collection scripts) | Likely ingestion source to extend for carryover-complete replay data collection. |

## Proposed stages

1. **Stage 0 — Coverage baseline and contract freeze**
   - Freeze required replay carryover field contract for rail-v2 validation classes.
   - Produce baseline activation/fallback distribution across available replay corpora (fixture classes + sampled real replay data).

2. **Stage 1 — Replay carryover ingestion/backfill path**
   - Add bounded ingestion/collection support so replay snapshots include enough persistent carryover signals (`player_states`-derived fields and prematch map context).
   - Preserve existing fallback behavior where completeness is impossible; no rail formula rewrite.

3. **Stage 2 — Validation gate and class segmentation**
   - Add explicit replay class segmentation (carryover-complete vs sparse/legacy) with deterministic reason taxonomy.
   - Enforce minimum activation-rate gates for carryover-complete classes in CI/nightly replay checks.

4. **Stage 3 — Bible-facing rail-v2 replay validation**
   - Run expanded replay validation pack with coverage-aware reporting.
   - Confirm invariants remain stable while rail-v2 activation reaches agreed thresholds.

5. **Stage 4 — Promotion decision point**
   - Decide whether conditions are now sufficient to reopen larger replay+simulation architecture work (currently banked) or proceed to calibration-focused stages.

## Validation checkpoints

- **Checkpoint A (post Stage 0):** reproducible baseline report with per-class `rail_input_v2_activated_points`, `rail_input_v1_fallback_points`, and reason-code distributions.
- **Checkpoint B (post Stage 1):** new/updated replay inputs demonstrate increased carryover completeness without structural regressions.
- **Checkpoint C (post Stage 2):** deterministic tests enforce class segmentation and activation-rate policy for carryover-complete classes.
- **Checkpoint D (post Stage 3):** replay validation evidence pack shows:
  - non-zero, policy-compliant v2 activation in intended classes,
  - stable `structural_violations_total=0`,
  - no replay contract-mode regressions.
- **Checkpoint E (stage stop gate):** explicit go/no-go recommendation for either calibration readiness or reopening banked replay+simulation architecture.

## Risks

- **Data availability risk:** some replay sources may not expose `player_states` reliably, requiring class-specific fallback policy rather than universal activation.
- **Contract drift risk:** ad-hoc ingestion fixes could create inconsistent field semantics across sources if not frozen early.
- **False confidence risk:** activation gains on synthetic fixtures may overstate real-world coverage unless real replay classes are included in baseline/targets.
- **Scope creep risk:** pressure to redesign replay architecture broadly during this initiative; must remain focused on carryover coverage contract and validation gates.

## Recommended branch plan

- Planning branch (this run):  
  `deep/proposal-20260307-0827-replay-carryover-contract-coverage`
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-replay-carryover-coverage-s0`
  - `deep/stage-YYYYMMDD-HHMM-replay-carryover-coverage-s1`
  - `deep/stage-YYYYMMDD-HHMM-replay-carryover-coverage-s2`
  - `deep/stage-YYYYMMDD-HHMM-replay-carryover-coverage-s3`
- Merge/promotion remains human-controlled; no self-merge.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
