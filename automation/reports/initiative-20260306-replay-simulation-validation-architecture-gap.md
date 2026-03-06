# Initiative proposal

## Initiative title

Replay + simulation validation architecture gap closure (Bible Chapter 9.2.3 and 9.9)

## Why it outranks other major issues

This is the highest-value unresolved major issue in current repository evidence:

- Canonical tests are currently green (`python3 -m pytest -q` -> `351 passed in 7.74s`), so there is no higher-ranked failing-test or structural-break signal to address first.
- The Bible defines replay validation as primary (Chapter 9.2.3) and simulation testing as a required target (Chapter 9.9), but the repo currently lacks a dedicated, canonical replay/simulation validation architecture.
- Evidence of the gap:
  - no `tests/replay/` or `tests/simulation/` suites,
  - no `engine/simulation/` module footprint,
  - replay execution is embedded in `backend/services/runner.py` (`_tick_replay*`) rather than exposed as a deterministic validation harness.

Alternative major candidates were considered (for example, additional PHAT movement/coupling refinement), but they are lower leverage right now because missing replay/simulation architecture limits confidence and measurement for any future model-behavior changes.

## Why it exceeds bounded-fix scope

This is not a localized bug fix. Closing this gap requires coordinated, cross-module architecture work:

- standard replay fixture contracts and deterministic execution interfaces,
- machine-readable replay metrics and drift comparison artifacts,
- simulation sequence generation with seeded reproducibility,
- canonical test-lane additions and validation command integration.

That scope spans engine modules, runner integration points, API/reporting surfaces, and test architecture. A bounded single-file patch would be incomplete and not auditable against Bible requirements.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Decouple replay execution logic from live tick loop into deterministic harness entrypoints. |
| `engine/replay/bo3_jsonl.py` | Harden/standardize replay fixture loading and normalization contracts. |
| `backend/api/routes_replay.py` | Surface harness controls and stable replay run metadata/status. |
| `tools/round_level_calibration.py` | Consume standardized replay outputs for calibration/reliability analysis. |
| `tests/unit/test_runner_bo3_hold.py` | Retain low-level runner behavior coverage while reducing role as de facto replay architecture test. |
| `tests/replay/` (new) | Canonical replay validation suite (fixtures, deterministic outputs, drift checks). |
| `engine/simulation/` (new) | Seeded synthetic trajectory generator aligned to Bible Chapter 9.9 phase-1 scope. |
| `tests/simulation/` (new) | Reproducibility + structural invariant checks over simulation outputs. |

## Proposed stages

1. Stage 1 - Replay contract + deterministic harness skeleton
   - Define replay fixture schema and output artifact schema.
   - Introduce deterministic replay harness entrypoint separated from live runtime path.
   - Emit baseline machine-readable summary metrics (invariants, timer directionality, trajectory summaries).
2. Stage 2 - Canonical replay validation suite
   - Add `tests/replay/` fixture-driven tests and a single replay validation command.
   - Add trajectory/invariant drift reporting for before-vs-after comparisons.
3. Stage 3 - Simulation phase-1 architecture
   - Add `engine/simulation/` seeded synthetic state generator (alive/loadout/bomb/timer/carryover transitions).
   - Add `tests/simulation/` reproducibility and structural invariant validations.
4. Stage 4 - Integration gates and rollout
   - Integrate replay + simulation checks into canonical validation workflow.
   - Define stage-level thresholds, failure triage outputs, and operator runbook.

## Validation checkpoints

- Checkpoint A (end Stage 1): deterministic replay harness runs fixed fixture and emits stable summary JSON (repeatable across runs).
- Checkpoint B (end Stage 2): replay suite exists under `tests/replay/` and produces drift deltas for trajectory + invariant counters.
- Checkpoint C (end Stage 3): seeded simulation runs are reproducible and preserve structural invariants across generated trajectories.
- Checkpoint D (end Stage 4): integrated replay+simulation command is runnable in CI/sandbox and emits machine-readable artifacts for review.

## Risks

- Replay payload schema drift may reduce determinism if contracts are underspecified.
- Validation runtime cost may increase and require phased execution budgets.
- Fixture coverage bias could create false confidence if data diversity is too narrow.
- Scope creep into PHAT retuning can dilute architecture-stage objectives if not explicitly deferred.
- Migration risk exists when separating runner-coupled replay logic from live compute paths.

## Recommended branch plan

- Keep `agent-initiative-base` as immutable starting point.
- Use one stage branch per approved implementation stage:
  - `agent/initiative/replay-simulation-validation-architecture-stage-1`
  - `agent/initiative/replay-simulation-validation-architecture-stage-2`
  - `agent/initiative/replay-simulation-validation-architecture-stage-3`
  - `agent/initiative/replay-simulation-validation-architecture-stage-4`
- Keep each stage branch single-issue and include:
  - baseline evidence,
  - stage-scoped validation results,
  - one stage report in `automation/reports/`.
- Do not self-merge; human review controls promotion.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
