branch: deep/proposal-20260306-0604-replay-sim-validation-arch
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative proposal

## Initiative title

Replay + Simulation Validation Architecture (canonical, stage-gated, cross-source)

## Why it outranks other major issues

Current evidence indicates the engine compute lane is stable at the unit/regression level, while validation architecture remains structurally underpowered for major-behavior assurance:

- Canonical suite baseline is green (`python3 -m pytest -q`): **360 passed**.
- Existing bounded replay assessment (`python3 tools/replay_verification_assess.py`) exercised only **3** raw-contract points and reported **0** structural/behavioral/invariant violations.
- Replay test footprint is narrow: only one replay-focused unit test file currently exists (`tests/unit/test_runner_replay_contract_mode.py`), focused on mode tagging rather than large-sample behavioral validation.
- Bible/Spec requirements explicitly require simulation testing architecture:
  - `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md` Chapter 9.9 requires a simulation sandbox harness.
  - `docs/ENGINE_SPEC.json` marks `testing_framework.simulation.required = true` and defines synthetic-state and policy-driven phases.
- Repository evidence shows no implemented simulation harness/test surface:
  - `tools/*simulation*.py`: **0 files**
  - `tests/**/*simulation*`: **0 files**

Given no current structural invariant failures or failing canonical tests, this is the highest-value major unresolved issue because it is the primary blocker to statistically meaningful replay/simulation validation and future calibration confidence.

## Why it exceeds bounded-fix scope

This is not a single-module patch. It requires coordinated architecture across:

1. Replay ingestion/normalization contracts,
2. Runner execution pathways and output artifacting,
3. Scenario/simulation state generation,
4. Validation metric aggregation and reporting,
5. Canonical tests and fixtures for regression gating.

The effort spans engine, backend runner, tools, fixtures, and tests with staged rollout and explicit governance boundaries, which exceeds maintenance-lane bounded-fix scope.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Canonical execution surfaces for REPLAY raw/point behavior and diagnostic payload parity. |
| `engine/replay/bo3_jsonl.py` | Replay loading contract, format handling, multi-match payload sequencing. |
| `tools/replay_verification_assess.py` | Existing bounded assessment baseline; likely promoted into broader validation orchestrator or split into reusable modules. |
| `tools/fixtures/raw_replay_sample.jsonl` | Seed fixture; insufficient coverage today, will be one of many fixture classes. |
| `engine/models.py` | Configuration surface for validation mode controls (if needed for stage-gated diagnostics). |
| `tests/unit/test_runner_replay_contract_mode.py` | Current replay contract tests; foundation for deeper replay validation assertions. |
| `tests/unit/test_runner_inter_map_break_parity.py` | Existing source parity guardrails relevant to replay correctness under map boundaries. |
| `tests/unit/test_grid_reducer_and_envelope.py` | Related phase/envelope correctness guards that must remain stable under replay/sim validation expansion. |
| `tests/` (new replay/simulation validation suites) | Canonical statistical/behavioral validation gates required by Bible Ch 9 and spec. |
| `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md` | Source-of-truth requirement for replay + simulation validation architecture (Chapter 9.9). |
| `docs/ENGINE_SPEC.json` | Explicit required-suite contract (`simulation_tests`, required metrics). |

## Proposed stages

1. **Stage 0 - Validation contract hardening (planning + schema)**
   - Define canonical replay-validation artifact schema (inputs, per-point diagnostics, aggregate metrics, reason codes).
   - Define simulation scenario schema aligned to Bible signal hierarchy and invariants.
   - Freeze acceptance gates for each downstream stage.

2. **Stage 1 - Replay validation architecture expansion**
   - Generalize replay assessment from bounded script to reusable validation pipeline (multi-file, multi-match, deterministic summaries).
   - Add canonical replay validation tests that assert metrics/emission contracts, not just mode tags.
   - Add stratified fixtures (raw BO3 snapshots, mixed-quality feeds, edge-case transitions).

3. **Stage 2 - Synthetic simulation harness (Chapter 9.9 Phase 1)**
   - Implement seeded synthetic state generator for coverage/stress (players_alive trajectories, loadout regimes, bomb/timer transitions, carryover transitions).
   - Ensure generated traces pass structural invariants and produce replay-comparable artifacts.
   - Add scenario stress suites wired to canonical CI entrypoints.

4. **Stage 3 - Unified replay + simulation regression framework**
   - Create shared aggregation/reporting layer for replay and simulation outputs.
   - Add baseline-vs-candidate comparison gates for p_hat trajectories, rail behavior, invariant rates, and calibration proxies.
   - Establish nightly/deep-run profile while preserving fast per-commit subset.

5. **Stage 4 - Policy-driven simulation foundation (Chapter 9.9 Phase 2 prep)**
   - Introduce minimal policy hooks for execute/retake/clutch/eco behavior templates.
   - Keep this stage optional until Stage 1-3 are stable and producing trusted diagnostics.

## Validation checkpoints

- **Checkpoint A (post Stage 0):** approved validation schemas + acceptance gates documented and test-enforced.
- **Checkpoint B (post Stage 1):**
  - replay architecture processes real multi-match inputs;
  - deterministic summary metrics emitted;
  - replay regression tests added and passing.
- **Checkpoint C (post Stage 2):**
  - seeded synthetic generator produces reproducible trajectories;
  - structural invariants never violated in generated traces;
  - scenario stress suite runs in canonical test paths.
- **Checkpoint D (post Stage 3):**
  - unified before/after comparator reports trajectory drift, rail drift, and violation-rate deltas;
  - regression thresholds enforce hold/fail behavior for major degradations.
- **Checkpoint E (post Stage 4):**
  - policy-driven simulation hooks produce richer but bounded distributions without violating structural contracts.

## Risks

- **Scope creep risk:** replay architecture work can drift into compute/calibration redesign; must enforce stage boundaries.
- **False confidence risk:** small fixtures may pass while real-world replay diversity breaks contracts; requires larger fixture classes early.
- **Performance risk:** deep replay/simulation suites can slow CI; mitigated by split fast-vs-nightly gate design.
- **Determinism risk:** non-seeded generation undermines diffability and regression signal quality.
- **Contract drift risk:** runner replay raw/point behavior may diverge again without shared validation schema and explicit contract tests.

## Recommended branch plan

- Keep this proposal branch as planning artifact only:
  - `deep/proposal-20260306-0604-replay-sim-validation-arch` (current).
- If approved, execute **one stage per branch**:
  - `deep/stage-YYYYMMDD-HHMM-replay-sim-validation-s0-contract`
  - `deep/stage-YYYYMMDD-HHMM-replay-sim-validation-s1-replay-pipeline`
  - `deep/stage-YYYYMMDD-HHMM-replay-sim-validation-s2-synth-harness`
  - `deep/stage-YYYYMMDD-HHMM-replay-sim-validation-s3-regression-gates`
  - `deep/stage-YYYYMMDD-HHMM-replay-sim-validation-s4-policy-foundation`
- Promotion remains human-gated; no self-merge.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
