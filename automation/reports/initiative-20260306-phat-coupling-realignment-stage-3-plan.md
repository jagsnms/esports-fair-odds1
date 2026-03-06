# Stage 3 refined execution plan

## Initiative

**PHAT Coupling and Movement Realignment — Stage 3 (Runner/Source Harmonization)**

## Planning mode confirmation

- Planning only.
- No runtime implementation changes in this run.
- No branch creation in this run.

## Baseline context used for this plan

- Branch/state inspected: `agent-initiative-base`
- HEAD at planning time: `6ec1337`
- Stage 2 contract movement behavior is present in `engine/compute/resolve.py` (movement step in IN_PROGRESS and diagnostics).

## Stage 3 objective (refined)

Harmonize BO3, GRID, and REPLAY runner/source behavior so source-specific ingestion/phase/inter-map handling no longer causes semantic drift from the intended PHAT contract pipeline already established in Stage 2.

This stage is **runner/source semantics alignment**, not compute-model redesign.

## Exact harmonization needs after Stage 2

### 1) Normalize source phase semantics before resolve path

**Why:** `resolve_p_hat` applies Stage 2 movement only when `round_phase == IN_PROGRESS`.

**Current divergence evidence:**
- `engine/compute/resolve.py` checks `IN_PROGRESS` literal.
- `engine/ingest/grid_reducer.py` currently sets:
  - `state.round_phase = state.clock_type or "gameClock"`
  - frame `bomb_phase_time_remaining={"round_phase": state.round_phase}`
- GRID `clock_type` values are not guaranteed to match BO3/resolve canonical phase vocabulary.

**Stage 3 need:**
- Add a canonical phase normalization map for GRID so frame phase values align with resolve contract expectations.
- Ensure BO3/GRID/REPLAY raw all produce phase values from the same canonical enum set used by resolve.

---

### 2) Unify inter-map-break PHAT continuity semantics across sources

**Current divergence evidence (runner):**
- BO3 (`_tick_bo3`) inter-map-break branch uses store current `p_hat` fallback and manual explain block.
- GRID (`_tick_grid`) has similar branch but different debug payload fields (`source`, `grid_series_id`) and no parity checks vs BO3/REPLAY debug contract.
- REPLAY raw (`_tick_replay`) also has its own inter-map-break branch.

**Risk:**
- In multi-session non-primary paths (`write_to_store=False`), inter-map-break fallback still reads `self._store.get_current()` (shared), which can leak primary-session PHAT continuity into non-primary source sessions.

**Stage 3 need:**
- Centralize inter-map-break fallback in one shared helper with identical semantics and debug schema.
- Use session-local continuity source when `write_to_store=False`; use store continuity only when `write_to_store=True`.
- Keep inter-map explain/diagnostic payload shape source-consistent.

---

### 3) Reconcile REPLAY raw vs REPLAY point passthrough contract behavior

**Current divergence evidence:**
- REPLAY raw mode runs full pipeline (normalize -> reduce -> bounds -> rails -> resolve).
- REPLAY point mode (`_tick_replay_point_passthrough`) appends point as-is with synthetic explain; no resolve contract enforcement.

**Stage 3 need:**
- Define and enforce explicit policy:
  - either replay point mode is *non-contract mode* (explicitly tagged and excluded from contract parity checks), or
  - replay point mode must be transformed to canonical frame + resolve path before append.
- Remove implicit semantic ambiguity where some replay paths are contract-bound and others are passthrough.

---

### 4) Align source-specific gating behavior around append/broadcast decisions

**Current divergence evidence:**
- BO3 has driver-validity emission gate (`drv_valid_microstate`, `drv_valid_roundstate`) that can skip append/broadcast.
- GRID and REPLAY raw do not mirror BO3 gate behavior.

**Stage 3 need:**
- Define one cross-source gating contract:
  - either all sources support equivalent driver-validity gate semantics, or
  - BO3-only gating is explicitly scoped and excluded from parity expectations with reason codes.
- Ensure resulting behavior is deliberate and test-encoded, not accidental.

---

### 5) Standardize invariant-mode wiring at runner call sites

**Current divergence evidence:**
- `compute_corridor_invariants(...)` supports `testing_mode`, but runner calls currently rely on default (`True`) without explicit mode control.

**Stage 3 need:**
- Pass explicit mode from config/runtime context at all source call sites.
- Ensure invariant reporting parity and intent across BO3/GRID/REPLAY.

## Where BO3, GRID, and REPLAY still diverge semantically

1. **Phase interpretation divergence (high impact):**
   - GRID phase currently derived from clock type string, not canonical resolve phase enum.
2. **Inter-map-break continuity source divergence (high impact):**
   - Shared-store fallback used in branches that should be session-local in non-primary ticks.
3. **Replay modality divergence (high impact):**
   - Raw replay is contract-compute; point replay is passthrough.
4. **Driver gate divergence (medium/high impact):**
   - BO3 applies emission gate; GRID/REPLAY do not.
5. **Invariant mode explicitness gap (medium impact):**
   - Mode parameter exists but not explicitly wired at runner boundaries.

## Inter-map-break / source-specific phase conflicts with initiative

Conflicts that Stage 3 must resolve:

- Inter-map-break currently bypasses normal resolve path with source-local custom blocks and no single helper contract.
- Source branches may carry different explain/debug shape and continuity fallback semantics.
- GRID phase mapping can prevent Stage 2 movement path from being engaged when it should be.
- Replay point passthrough introduces non-contract points into the same history stream without strict contract-mode boundaries.

## What must remain unchanged during Stage 3

The following are **out of scope** and must not be altered in Stage 3:

1. PHAT compute semantics in:
   - `engine/compute/resolve.py` (movement formula, confidence value)
   - `engine/compute/midround_v2_cs2.py`
   - `engine/compute/q_intra_cs2.py`
   - `engine/compute/rails_cs2.py`
2. Calibration/tuning assets and fitting behavior:
   - `tools/*` calibration scripts
   - learned weight profiles and coefficients
3. Major architecture migrations beyond runner/source harmonization.
4. UI/front-end behavior and market logic unrelated to source contract parity.

## Exact files to touch in Stage 3 implementation

### Primary code files (expected)

- `backend/services/runner.py`
  - consolidate per-source pipeline parity helpers,
  - centralize inter-map-break handling,
  - explicit invariant mode wiring,
  - explicit replay mode contract policy handling.
- `engine/ingest/grid_reducer.py`
  - canonical phase normalization for GRID -> resolve-compatible phase values.
- `engine/models.py`
  - only if required to add narrowly scoped source-contract mode flags (e.g., explicit replay contract mode).

### Primary tests to add/update (expected)

- `tests/unit/test_runner_bo3_hold.py` (extend for source parity assertions where relevant).
- `tests/unit/test_grid_reducer_and_envelope.py` (phase normalization assertions).
- `tests/unit/test_runner_telemetry_status.py` (if mode/status implications are touched).
- **New tests expected:**
  - `tests/unit/test_runner_source_contract_parity.py`
  - `tests/unit/test_runner_inter_map_break_parity.py`
  - `tests/unit/test_runner_replay_contract_mode.py`

## Exact files that must not be touched yet

- `engine/compute/resolve.py`
- `engine/compute/midround_v2_cs2.py`
- `engine/compute/q_intra_cs2.py`
- `engine/compute/rails_cs2.py`
- `engine/compute/bounds.py`
- `tools/*` calibration scripts
- `ML/*`
- `frontend/*`

(Unless a minimal import or flag plumbing change is strictly required; any such exception must be explicitly justified in Stage 3 implementation report.)

## Required validation evidence before Stage 3 implementation approval

Approval to implement Stage 3 should require a pre-defined evidence pack:

### A) Baseline parity snapshot (pre-change)

1. Source-path matrix documenting, for BO3 vs GRID vs REPLAY(raw/point):
   - phase value entering resolve,
   - whether resolve IN_PROGRESS movement path is engaged,
   - inter-map-break fallback source for `p_hat_prev`,
   - append/broadcast gate behavior.
2. Concrete evidence logs/examples from each source with explain+debug fields.

### B) Test contract to be added in Stage 3 PR

Must include passing tests proving:

1. **Phase parity:** canonical phases map consistently across BO3/GRID/REPLAY raw.
2. **Inter-map-break parity:** same continuity rules and debug schema across sources; non-primary sessions do not use shared store fallback.
3. **Replay mode explicitness:** point mode contract policy is explicit and enforced.
4. **Invariant mode explicitness:** runner passes explicit testing/production mode to corridor invariant checks.
5. **No compute semantic drift:** existing Stage 2 resolve/contract tests still pass unchanged.

### C) Required command evidence (minimum)

- `python3 -m pytest -q tests/unit/test_resolve_micro_adj.py tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_corridor_invariants.py`
- `python3 -m pytest -q tests/unit/test_runner_bo3_hold.py tests/unit/test_grid_reducer_and_envelope.py tests/unit/test_runner_telemetry_status.py`
- `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_runner_inter_map_break_parity.py tests/unit/test_runner_replay_contract_mode.py` (new Stage 3 tests)

### D) Acceptance gates

Stage 3 implementation should not be approved unless:

- source parity tests pass,
- Stage 2 compute contract tests pass unchanged,
- no newly introduced structural invariant violations,
- no unintended PHAT formula/parameter changes are present in compute modules.

## Recommended Stage 3 stop conditions

Stop Stage 3 implementation if any required harmonization step would force:
- compute semantic migration (Stage 4+ concern),
- calibration retuning,
- broad architecture movement outside runner/source layer.

## Recommendation

- **Approve Stage 3 implementation only with the above file boundaries and validation gates locked.**
- Keep Stage 3 strictly as runner/source harmonization over already-promoted Stage 2 contract behavior.
