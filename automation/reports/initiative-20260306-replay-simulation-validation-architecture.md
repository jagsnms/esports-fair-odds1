# Initiative proposal

## Initiative title

Replay + simulation validation architecture (Bible Chapter 9.2.3 and 9.9 compliance)

## Why it outranks other major issues

- **Current canonical tests are green**: `python3 -m pytest -q` -> **341 passed** (no unresolved structural-invariant or failing-test signal to rank above this).
- **Replay/simulation are defined as primary/required in the Bible**:
  - Chapter 9.2.3 states replay validation is the primary testing source of truth.
  - Chapter 9.9 marks simulation testing as a required target.
- **Current repo evidence shows a gap**:
  - Replay pipeline test logic is concentrated in one file: `tests/unit/test_runner_bo3_hold.py` (contains replay tick assertions but not a canonical replay-validation suite).
  - No dedicated replay test lane by path/name (`tests/**/replay/**/*.py` and `tests/unit/test_*replay*.py` return none).
  - No simulation module footprint in engine (`engine/**/*simulation*` returns none).
- This makes replay/simulation architecture the highest-value unresolved **major** issue because it blocks systematic detection of real-trajectory drift despite passing unit tests.

## Why it exceeds bounded-fix scope

This is not a single bug fix. It requires coordinated design and implementation across data contracts, runner/replay ingestion, validation tooling, diagnostics artifacts, and CI entrypoints. The work spans multiple modules and must be staged to avoid destabilizing the canonical runtime path.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `engine/replay/bo3_jsonl.py` | Extend/standardize replay ingestion contracts and filtering for validation harness use. |
| `backend/services/runner.py` | Provide deterministic replay stepping hooks and reusable instrumentation payloads. |
| `backend/api/routes_replay.py` | Expose replay run metadata/status needed for repeatable validation runs. |
| `scripts/replay_anchor_dump.py` | Migrate ad-hoc replay inspection toward standardized harness interfaces. |
| `tools/round_level_calibration.py` | Consume replay outputs for round-level calibration and reliability checks. |
| `tests/unit/test_runner_bo3_hold.py` | Preserve/extend low-level replay behavior checks while adding higher-level replay suites. |
| `tests/` (new replay/simulation suites) | Add canonical replay-validation and simulation-stress test paths with seeded reproducibility. |
| `automation/reports/` | Store stage reports with baseline/after metrics and stop reasons. |

## Proposed stages

1. **Stage 1 - Replay validation contract + harness skeleton**
   - Define canonical replay dataset contract (inputs, expected outputs, diagnostics schema).
   - Add a deterministic replay harness entrypoint (no engine semantic changes).
   - Produce baseline replay metrics artifact format (trajectory drift, invariant counters, timer-direction checks).
2. **Stage 2 - Canonical replay validation suite**
   - Add replay-driven tests that assert trajectory/invariant behavior on fixed fixtures.
   - Wire a repeatable command for local and CI replay validation.
   - Add regression comparison mode (before/after metric deltas).
3. **Stage 3 - Simulation Phase 1 synthetic generator**
   - Implement seeded synthetic state trajectory generator aligned to Bible Ch. 9.9 Phase 1.
   - Cover required transitions (alive counts, loadout regimes, bomb/timer transitions, carryover context shifts).
   - Emit diagnostics artifacts comparable to replay metrics.
4. **Stage 4 - Policy integration and gates**
   - Integrate replay + simulation checkpoints into canonical validation workflow.
   - Add thresholds for invariant health and trajectory stability with explicit reporting.
   - Document operator workflow and failure triage.

## Validation checkpoints

- **Checkpoint A (end Stage 1):**
  - Harness executes deterministically on fixed replay fixture (same seed/config -> byte-stable summary JSON).
  - Baseline report includes at least: invariant counts, timer-directionality counters, trajectory summary stats.
- **Checkpoint B (end Stage 2):**
  - Replay validation suite exists under canonical test paths and is runnable in one command.
  - Regression mode compares current run vs baseline and flags measurable drift.
- **Checkpoint C (end Stage 3):**
  - Simulation generator produces reproducible synthetic trajectories with seeded RNG.
  - Structural invariants hold in generated sequences; diagnostics artifacts are emitted.
- **Checkpoint D (end Stage 4):**
  - CI/local workflow runs replay + simulation validations and emits machine-readable summaries.
  - Documentation updated with run commands, artifact locations, and interpretation guide.

## Risks

- **Data contract drift risk:** replay payload schema variation may cause harness fragility without strict normalization.
- **Runtime/perf risk:** replay/simulation suites may increase validation time; requires staged performance budgeting.
- **False confidence risk:** poorly chosen fixtures can pass while missing real-world failure modes.
- **Scope creep risk:** initiative can expand into model-tuning work; must stay focused on architecture and validation plumbing first.
- **Adoption risk:** without clear artifacts and thresholds, new harnesses may be underused by maintainers.

## Recommended branch plan

- Keep `agent-initiative-base` as immutable starting point.
- Execute one stage per branch:
  - `agent/initiative/replay-simulation-validation-architecture-stage-1`
  - `agent/initiative/replay-simulation-validation-architecture-stage-2`
  - `agent/initiative/replay-simulation-validation-architecture-stage-3`
  - `agent/initiative/replay-simulation-validation-architecture-stage-4`
- Each stage branch should include:
  - one stage-scoped change set,
  - baseline and after metrics artifacts,
  - a stage report in `automation/reports/`,
  - no self-merge (human promotion only).

## Recommendation

- [x] **Approve planning only** - accept proposal; do not start implementation
- [ ] **Approve stage 1** - approve first stage for implementation
- [ ] **Defer** - do not approve; revisit later
