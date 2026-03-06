branch: deep/proposal-20260306-0640-simulation-harness-phase1
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve stage 1

# Initiative proposal

## Initiative title

Bible-required simulation validation harness (Phase 1 synthetic generator + replay/sim comparator architecture)

## Why it outranks other major issues

This is the highest-value unresolved major issue because current evidence shows a hard requirement gap in the validation architecture, while recent major work has already advanced replay basics:

- Bible requirement exists and is explicit: Chapter 9.9 requires a simulation sandbox harness and Phase 1 synthetic state generator.
- Spec requirement is explicit and machine-readable:
  - `docs/ENGINE_SPEC.json` includes `testing_framework.required_suites` containing `simulation_tests`.
  - `docs/ENGINE_SPEC.json` sets `testing_framework.simulation.required = true`.
- Repository implementation gap is concrete:
  - `tools/*sim*` -> no simulation tooling files found.
  - `tests/**/*sim*` -> no simulation test files found.
- Current replay validation is still bounded and insufficient for this requirement:
  - `tools/replay_verification_assess.py` emits deterministic Stage 1 aggregate counts/violations only.
  - `automation/reports/deep-stage-20260306-0616-replay-validation-stage1.md` confirms Stage 1 scope and explicitly notes "No simulation work added."
  - Replay fixtures currently used for this lane are tiny (3-point raw sample, 6-point multi-match sample), which is not enough to satisfy the simulation requirement or stress coverage goals in Chapter 9.

Alternative major candidates (for example, replay raw/point mode semantics refinement, calibration report expansion) are important, but they do not outrank the current missing required-suite architecture mandated by Bible + Spec.

## Why it exceeds bounded-fix scope

This cannot be solved by a single-file or narrow bounded maintenance patch. It requires coordinated cross-module architecture:

1. A new synthetic trajectory generation layer (seeded and reproducible),
2. Integration with runner/replay compute paths for comparable outputs,
3. Shared artifact schemas and comparison contracts for replay vs simulation,
4. New canonical tests and fixtures for stress/regression evidence,
5. Staged CI/test-entrypoint strategy so deep validation is enforceable without collapsing fast test loops.

That is multi-stage, cross-module architectural work and belongs in the deep initiative lane.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md` | Canonical requirement source (Chapter 9.9 simulation harness and synthetic generator). |
| `docs/ENGINE_SPEC.json` | Machine-readable contract requiring simulation test suite and seeded synthetic generation. |
| `tools/replay_verification_assess.py` | Existing Stage 1 replay assessment foundation to be reused/split for replay-vs-simulation comparators. |
| `tools/schemas/replay_validation_summary.schema.json` | Current replay summary schema; likely baseline for a broader validation artifact family. |
| `backend/services/runner.py` | Canonical execution path whose outputs must be consumed consistently by replay and simulation validation flows. |
| `engine/replay/bo3_jsonl.py` | Replay loading contract; used as replay-side input baseline for comparator architecture. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Existing Stage 1 deterministic/schema test to keep green while expanding architecture. |
| `tests/unit/test_runner_replay_contract_mode.py` | Existing replay mode guardrails; should remain stable as validation architecture broadens. |
| `tools/` (new simulation modules, stage-gated) | New seeded synthetic state generator + replay/simulation comparison tooling. |
| `tests/` (new simulation validation suites) | New canonical simulation and replay/sim parity validation tests. |

## Proposed stages

1. **Stage 1 - Simulation contract and artifact schema freeze**
   - Define canonical simulation input/output schema (state-sequence contract, seed metadata, invariants payload fields).
   - Define replay/simulation comparator schema (trajectory deltas, invariant-rate deltas, rail envelope drift).
   - Lock acceptance gates before generator implementation.

2. **Stage 2 - Phase 1 synthetic state generator (Bible Ch 9.9)**
   - Implement seeded synthetic generator for required trajectory families:
     - players_alive paths,
     - loadout regime shifts,
     - bomb-plant/timer transitions,
     - carryover/economy swings.
   - Ensure outputs are deterministic under fixed seed and consumable by validation pipeline.

3. **Stage 3 - Replay/simulation unified validation runner**
   - Build orchestrator that runs replay assessment and simulation assessment through aligned summary contracts.
   - Emit comparable metrics (violation totals/rates, p_hat/rail range summaries, trajectory drift buckets).

4. **Stage 4 - Canonical test and regression gating**
   - Add unit/integration tests for deterministic generation, schema conformance, and comparator correctness.
   - Define fast subset vs deep/nightly subset with stable commands and artifact outputs.

5. **Stage 5 - Policy-driven simulation prep (optional, post-stability)**
   - Add extension points for policy-driven trajectories once Stage 1-4 are stable and trusted.

## Validation checkpoints

- **Checkpoint A (after Stage 1):** schema contracts checked in and validated by tests; acceptance thresholds documented.
- **Checkpoint B (after Stage 2):** identical-seed runs produce byte-identical artifacts; structural invariants show zero hard violations.
- **Checkpoint C (after Stage 3):** replay and simulation runs emit unified comparable summaries from one orchestrated entrypoint.
- **Checkpoint D (after Stage 4):** canonical tests cover generator determinism, schema conformance, comparator metrics, and failure signaling.
- **Checkpoint E (after Stage 5 if approved):** policy-driven sequences increase coverage without violating structural invariants.

## Risks

- **Scope creep:** simulation initiative can drift into compute/calibration redesign unless stage boundaries are strictly enforced.
- **False confidence:** synthetic data can overfit expected behaviors if scenario distributions are too narrow.
- **Performance overhead:** deep validation can slow CI unless split into fast and deep tracks.
- **Contract drift:** replay and simulation summaries can diverge if schema versioning is not enforced centrally.
- **Environment reproducibility risk:** non-seeded randomness or hidden runtime dependencies can invalidate regression comparisons.

## Recommended branch plan

- Keep this branch as proposal only:
  - `deep/proposal-20260306-0640-simulation-harness-phase1`
- If approved, run one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-sim-harness-s1-contract`
  - `deep/stage-YYYYMMDD-HHMM-sim-harness-s2-generator`
  - `deep/stage-YYYYMMDD-HHMM-sim-harness-s3-unified-validation`
  - `deep/stage-YYYYMMDD-HHMM-sim-harness-s4-regression-gates`
  - `deep/stage-YYYYMMDD-HHMM-sim-harness-s5-policy-prep` (optional)
- Do not merge from deep branches directly; promotion remains human-gated.

## Recommendation

- [ ] **Approve planning only** — accept proposal; do not start implementation
- [x] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
