# Initiative proposal

## Initiative title

Replay + simulation validation architecture (Bible Chapter 9.2.3 and 9.9 compliance)

## Why it outranks other major issues

This run ranks replay/simulation validation architecture as the highest-value unresolved major issue because current evidence shows:

- **No higher-ranked failing signal in canonical tests**: `python3 -m pytest -q` -> `341 passed in 7.10s`.
- **Bible contract makes replay primary and simulation required**:
  - `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md` Chapter 9.2.3: replay validation is the primary source of truth.
  - Chapter 9.9: simulation testing is a required target.
- **Current repo lacks dedicated architecture for that contract**:
  - no simulation module footprint under `engine/` (`engine/**/*simulation*` -> none),
  - no dedicated replay/simulation test lane under `tests/` (`tests/**/*replay*.py`, `tests/**/*simulation*.py` -> none),
  - replay logic is mainly embedded in runner flow (`backend/services/runner.py` `_tick_replay*`) and mixed into BO3-focused unit coverage (`tests/unit/test_runner_bo3_hold.py`), not a canonical replay-validation harness.

Why this outranks alternatives (including PHAT coupling stage advancement): PHAT semantic realignment remains important, but without canonical replay/simulation validation architecture, stage-level PHAT migration cannot be verified against the Bible’s required replay-first evidence model.

## Why it exceeds bounded-fix scope

This cannot be solved as a localized fix. It requires cross-module architecture work:

- normalized replay data contracts,
- deterministic replay execution harness,
- replay metric artifacts and drift comparison,
- simulation state-sequence generator with seeded reproducibility,
- canonical tests and validation entrypoints.

That scope spans runtime services, engine replay utilities, tools, and test layout; a one-file patch would be incomplete and non-auditable.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Separate deterministic replay harness path from live/runtime tick path; expose stable replay instrumentation outputs. |
| `engine/replay/bo3_jsonl.py` | Harden replay input contract normalization and fixture loading for harness use. |
| `backend/api/routes_replay.py` | Surface replay run metadata/status and deterministic run controls. |
| `tools/round_level_calibration.py` | Consume standardized replay artifacts for calibration/reliability evaluation. |
| `tests/unit/test_runner_bo3_hold.py` | Keep low-level replay behavior checks while reducing overloading as de facto replay architecture test. |
| `tests/replay/` (new) | Canonical replay validation suite with fixed fixtures and drift checks. |
| `tests/simulation/` (new) | Seeded synthetic trajectory validation suite aligned to Bible Chapter 9.9. |
| `engine/simulation/` (new) | Phase-1 synthetic state generator and reproducible scenario orchestration. |

## Proposed stages

1. **Stage 1 - Replay validation contract + harness skeleton**
   - Define canonical replay fixture contract and output schema.
   - Add deterministic replay harness entrypoint (no PHAT semantic changes).
   - Emit baseline machine-readable replay metrics (invariant counts, timer directionality counts, trajectory summaries).

2. **Stage 2 - Canonical replay validation suite**
   - Add replay-driven tests under `tests/replay/`.
   - Add one standard command for local/CI replay validation.
   - Add before/after drift comparison mode for trajectories and invariant counters.

3. **Stage 3 - Simulation Phase 1 (Bible 9.9)**
   - Implement seeded synthetic state generator in `engine/simulation/`.
   - Cover required transitions (alive counts, loadout regimes, bomb/timer transitions, carryover changes).
   - Emit artifacts comparable to replay metrics.

4. **Stage 4 - Integration gates and rollout**
   - Integrate replay + simulation checkpoints into canonical validation workflow.
   - Add stage-level thresholds and failure triage outputs.
   - Document operator flow for baseline/after comparisons and stop criteria.

## Validation checkpoints

- **Checkpoint A (end Stage 1)**: deterministic replay harness runs fixed fixture with byte-stable summary JSON.
- **Checkpoint B (end Stage 2)**: replay suite exists and runs from a single command; drift report includes measurable deltas.
- **Checkpoint C (end Stage 3)**: seeded simulation run is reproducible; structural invariants remain satisfied throughout generated sequences.
- **Checkpoint D (end Stage 4)**: local/CI workflow executes replay + simulation checks and publishes machine-readable summaries.

## Risks

- Replay payload schema drift can break harness determinism if normalization is underspecified.
- Validation runtime cost may grow and require staged execution budgets.
- Fixture bias may create false confidence if datasets are narrow.
- Scope creep risk into PHAT tuning must be controlled until architecture/checkpoint work is complete.
- Teams may underuse the harness unless outputs are concise and actionable.

## Recommended branch plan

- Keep `agent-initiative-base` as the immutable starting point.
- Use one branch per approved stage:
  - `agent/initiative/replay-simulation-validation-architecture-stage-1`
  - `agent/initiative/replay-simulation-validation-architecture-stage-2`
  - `agent/initiative/replay-simulation-validation-architecture-stage-3`
  - `agent/initiative/replay-simulation-validation-architecture-stage-4`
- Keep each stage branch single-issue and include:
  - baseline evidence,
  - stage-scoped validation output,
  - one stage report under `automation/reports/`.
- No self-merge; promotion remains human-only.

## Recommendation

- [ ] **Approve planning only** — accept proposal; do not start implementation
- [x] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later
