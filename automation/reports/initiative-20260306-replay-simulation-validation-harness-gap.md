# Initiative proposal

## Initiative title

Replay + simulation validation harness architecture gap closure (Bible Chapter 9.2.3 and 9.9)

## Why it outranks other major issues

Current repo evidence places this as the highest-value unresolved **major** issue:

- Canonical test baseline is green (`python3 -m pytest -q` -> `360 passed in 7.58s`), so there is no currently evidenced rank-1/2 blocker (no structural-break or failing-test signal to outrank this).
- PHAT coupling/movement major work has already advanced through staged reports (`automation/reports/initiative-20260306-phat-coupling-realignment-stage-1.md` through `stage-3b.md`) and is present on `agent-initiative-base`.
- Bible Chapter 9.2.3 declares replay validation the primary source of truth, and Chapter 9.9 requires simulation testing.
- Present architecture evidence still shows a gap:
  - no dedicated replay test lane (`tests/replay/**/*.py` -> none),
  - no dedicated simulation test lane (`tests/simulation/**/*.py` -> none),
  - no simulation module footprint in engine (`engine/**/*simulation*.py` -> none),
  - replay stepping remains embedded in runtime runner paths (`backend/services/runner.py` methods `_tick_replay` and `_tick_replay_point_passthrough`), limiting deterministic validation architecture.

Given green unit tests and completed PHAT realignment stages, the replay/simulation validation architecture gap is now the highest-leverage unresolved cross-module initiative.

## Why it exceeds bounded-fix scope

This cannot be solved by a localized bug patch. Closing the gap requires coordinated architectural work across replay contracts, deterministic harness entrypoints, new canonical test lanes, artifact schemas, and validation workflow integration. It spans engine modules, runner interfaces, test topology, and reporting integration, and therefore exceeds maintenance-lane bounded-fix scope.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Extract deterministic replay-execution hooks from live tick path, define stable harness entrypoints. |
| `engine/replay/bo3_jsonl.py` | Formalize replay fixture contract and normalization guarantees for deterministic harness runs. |
| `backend/api/routes_replay.py` | Expose run metadata/state required for reproducible replay-validation runs and artifact retrieval. |
| `tests/unit/test_runner_bo3_hold.py` | Preserve low-level runner behavior assertions while reducing role as the de facto replay architecture suite. |
| `tests/unit/test_runner_replay_contract_mode.py` | Keep mode-policy unit checks as lower-level guardrails under new replay suite. |
| `tests/replay/` (new) | Add canonical replay validation suites with fixed fixtures and trajectory/invariant assertions. |
| `engine/simulation/` (new) | Implement seeded synthetic state-sequence generator aligned to Bible 9.9 Phase 1. |
| `tests/simulation/` (new) | Validate simulation reproducibility and structural invariants over generated trajectories. |
| `automation/reports/` | Stage-level evidence reports with baseline/after metrics and stop reasons. |

## Proposed stages

1. Stage 1 - Replay validation contract + deterministic harness skeleton
   - Define canonical replay input/output schema (fixtures, run metadata, summary metrics JSON).
   - Introduce deterministic replay harness entrypoint separated from live run loop.
   - No PHAT/rail semantic retuning in this stage.
2. Stage 2 - Canonical replay validation suite
   - Add `tests/replay/` with fixture-driven trajectory/invariant checks.
   - Add one repeatable replay-validation command for sandbox/CI.
   - Add before/after drift comparison artifacts.
3. Stage 3 - Simulation phase-1 architecture
   - Add `engine/simulation/` seeded synthetic trajectory generator (alive/loadout/bomb/timer/carryover transitions).
   - Add `tests/simulation/` reproducibility + structural invariant checks.
4. Stage 4 - Integration and operational gates
   - Integrate replay + simulation validation into canonical run workflow.
   - Define thresholds, failure reason codes, and artifact conventions.
   - Document runbook for triage and stage promotion decisions.

## Validation checkpoints

- Checkpoint A (end Stage 1): deterministic replay harness executes fixed fixture repeatedly and emits byte-stable summary JSON.
- Checkpoint B (end Stage 2): replay suite under `tests/replay/` is runnable in one command and emits trajectory/invariant drift deltas.
- Checkpoint C (end Stage 3): simulation generator is seed-reproducible and preserves structural invariants across sampled runs.
- Checkpoint D (end Stage 4): integrated replay+simulation validation command works in sandbox/CI and emits machine-readable artifacts consumed by reports.

## Risks

- Replay schema drift can break determinism if contracts are underspecified.
- Validation runtime may increase and require staged budgets/tiers.
- Fixture-selection bias can create false confidence if replay corpus is narrow.
- Scope creep into model retuning can derail architecture-first staging.
- Extracting harness logic from runner paths can introduce behavior drift if parity tests are incomplete.

## Recommended branch plan

- Keep `agent-initiative-base` as immutable planning baseline.
- Execute one implementation stage per branch:
  - `agent/initiative/replay-simulation-validation-harness-stage-1`
  - `agent/initiative/replay-simulation-validation-harness-stage-2`
  - `agent/initiative/replay-simulation-validation-harness-stage-3`
  - `agent/initiative/replay-simulation-validation-harness-stage-4`
- For each stage branch:
  - include stage-scoped baseline evidence,
  - run stage-specific validation checkpoints,
  - publish one stage report under `automation/reports/`,
  - do not self-merge (human promotion only).

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
