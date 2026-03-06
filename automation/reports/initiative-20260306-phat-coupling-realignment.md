# Initiative proposal

## Initiative title

**PHAT Coupling and Movement Realignment (Bible Chapter 1/6/7 contract)**

## Selected issue

- **Issue class:** Major core model-behavior misalignment (PHAT semantics drift)
- **Primary mismatch:** Canonical code path does not consistently implement the Bible-defined coupling contract `target_p_hat = rail_low + q * (rail_high - rail_low)` followed by movement/inertia toward target.

## Why this outranks alternatives

This outranks other major candidates because it is a direct model-identity conflict in the core compute path, and it is currently encoded in both runtime behavior and tests:

1. **Bible contract requires coupling + movement pipeline**:
   - `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md`:
     - Core identity (`p_hat = rail_low + q * (rail_high - rail_low)`) at L44.
     - Canonical pipeline order (`... target_p_hat -> movement -> p_hat`) at L289-L329.
     - Testing policy warns against silent clamping masking behavioral faults at L412-L425.
2. **Current engine behavior is phase-gated/clamp-dominant**:
   - `engine/compute/resolve.py`:
     - Base+micro adjustment clamped immediately into rails (L115-L117).
     - BUY/FREEZETIME midpoint freeze (L127-L134).
     - Non-IN_PROGRESS path bypasses movement/coupling target progression (L166-L173).
     - IN_PROGRESS path uses `p_mid_clamped` and hard clamp into rails (L176-L184).
3. **Tests currently codify this drift as expected behavior**:
   - `tests/unit/test_resolve_micro_adj.py`: clamp and midpoint freeze assertions (L75-L85, L98-L137).
   - `tests/unit/test_compute_slice1.py`: asserts `p_hat` inside rails as hard behavior (L64-L75, L102-L113).
4. **Fresh evidence from this run**:
   - Targeted canonical engine tests pass and reinforce current behavior:
     - `python3 -m pytest -q tests/unit/test_resolve_micro_adj.py tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_rails_cs2_basic.py`
     - Result: `72 passed`.

Given the project’s major-initiative preference for core model/PHAT/rail behavior, this is higher value than infrastructure-first initiatives.

## Baseline evidence (current run)

- Baseline branch start: `agent-initiative-base`
- Baseline commit: `84f5a86`
- Required references read:
  - `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md`
  - `automation/AUTOMATION_RUN_POLICY.md` (repo-canonical policy path)
  - `automation/README.md`
  - `automation/templates/promotion_report_template.md` (only template currently present in repo)
- Test evidence gathered in this run:
  - `python3 -m pytest -q` -> collection blocked by missing local environment deps (`fastapi`, `pandas`, `bs4`) in this runner image.
  - `python3 -m pytest -q tests/unit/test_resolve_micro_adj.py tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_rails_cs2_basic.py` -> `72 passed`.

## Why infrastructure/validation alternatives were not selected first

Validation/replay architecture has real risks, but current evidence indicates it is **not the primary blocker** to initiating PHAT contract realignment:

- Replay-truth risk exists (`backend/services/runner.py` winner inference note at L1500 and replay passthrough mode around L2624), but diagnostics/tooling for replay alignment already exist and the primary issue here is still core model semantics.
- The stronger contradiction is not missing tooling; it is that the current model path itself is not aligned with Bible PHAT coupling/movement intent.
- By selection rule, we should not choose easier self-infrastructure work when a core behavior mismatch is materially present.

Therefore replay/validation improvements should be treated as **parallel guardrails for later stages**, not as the first selected initiative.

## Why this exceeds bounded-fix scope

This is not a localized bug fix. It requires coordinated architectural changes across:

- Core compute semantics (`resolve`, `midround_v2`, possibly `q` integration contract).
- Runner source paths (BO3/GRID/REPLAY) and inter-map break handling to keep a single PHAT contract.
- Diagnostics and invariant policy (distinguishing structural vs behavioral violations without masking).
- Canonical tests that currently encode clamp-first semantics.
- Replay/calibration tools that consume explain/debug fields.

Any one-file patch would create partial behavior and unstable semantics across modules.

## Affected modules/files

| Path | Expected role in initiative |
|------|-----------------------------|
| `engine/compute/resolve.py` | Replace clamp-dominant resolution with explicit target + movement contract and phase policy rules. |
| `engine/compute/midround_v2_cs2.py` | Define `q` contribution boundaries vs movement, and preserve explainability while decoupling direct final PHAT forcing. |
| `engine/compute/q_intra_cs2.py` | Clarify q semantics contract used by target computation. |
| `backend/services/runner.py` | Unify BO3/GRID/REPLAY handling with consistent PHAT contract at normal and inter-map-break phases. |
| `engine/diagnostics/invariants.py` | Extend invariant checks for target/movement diagnostics and non-silent behavioral visibility during testing mode. |
| `tests/unit/test_resolve_micro_adj.py` | Replace clamp-as-identity tests with coupling/movement contract tests. |
| `tests/unit/test_compute_slice1.py` | Update to contract-aware invariants (structural vs behavioral modes). |
| `tests/unit/test_midround_v2_cs2.py` | Ensure q behavior is validated independently from final PHAT movement. |
| `tests/unit/test_q_intra_cs2.py` | Preserve q bounds/monotonic tests while aligning with updated contract boundaries. |

## Proposed stages

### Stage 0 (this run): proposal and approval gate
- Deliver initiative proposal, scope boundaries, and stage checkpoints.
- No broad implementation.

### Stage 1: Contract formalization + test harness pivot
- Add explicit target/movement contract tests (unit-level) without full algorithm migration.
- Introduce clear testing-mode policy for behavioral violations (diagnostic visibility, no silent masking).
- Keep runtime behavior stable behind flags while proving contract expectations.

### Stage 2: Core resolve-path migration
- Implement canonical target computation and movement/inertia update path in `resolve`.
- Remove/contain phase shortcuts that bypass contract semantics.
- Preserve strict structural safety rules while surfacing behavioral diagnostics.

### Stage 3: Runner/source harmonization
- Align BO3, GRID, and REPLAY runner paths (including inter-map break logic) with the same PHAT contract.
- Ensure explain/debug payloads remain calibration-compatible and source-consistent.

### Stage 4: Replay + calibration verification pass
- Run replay/correlation/calibration tools for before/after drift assessment.
- Validate no regression in structural invariants and monitor behavioral violation trends.
- Tune movement parameters only after contract compliance is achieved.

## Validation checkpoints

1. **Contract unit tests (new):**
   - Verify `target_p_hat` is computed from `rail_low`, `rail_high`, and `q`.
   - Verify movement evolves toward target with bounded confidence/inertia.
2. **Structural invariants (must pass):**
   - `0 <= q <= 1`, `rail_low <= rail_high`, series resolution boundaries.
3. **Behavioral diagnostics (must be emitted, not hidden in testing mode):**
   - Rail containment violations counted and logged when present.
   - Convergence behavior tracked near certainty states.
4. **Cross-source parity checks:**
   - BO3/GRID/REPLAY produce consistent PHAT semantics for equivalent states.
5. **Regression and replay checks:**
   - No structural test regressions.
   - Replay calibration tools show no inversion/catastrophic drift relative to baseline.

## Risks

- **Semantic regression risk:** existing consumers and dashboards may assume clamp-first PHAT behavior.
- **Test migration blast radius:** many current tests encode the old behavior and will need intentional rewrites.
- **Replay comparability risk:** historical baselines may shift when movement semantics become contract-correct.
- **Tuning risk:** movement parameters may require multiple iterations to avoid instability after contract correction.

## Recommended branch plan

Start from `agent-initiative-base`, then use one branch per approved stage:

- `agent/initiative/phat-coupling-realignment-stage-1`
- `agent/initiative/phat-coupling-realignment-stage-2`
- `agent/initiative/phat-coupling-realignment-stage-3`
- `agent/initiative/phat-coupling-realignment-stage-4`

Rules:
- one primary issue per stage branch,
- no self-merge,
- promotion by human review only.

## Recommendation

- [x] **Approve planning only**
- [ ] **Approve stage 1**
- [ ] **Defer**

## Stop reason

Proposal mode completed. No broad implementation performed in this run.
