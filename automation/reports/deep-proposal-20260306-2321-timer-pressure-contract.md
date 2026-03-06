branch: deep/proposal-20260306-2321-timer-pressure-contract
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve stage 1

# Initiative title

Timer-pressure contract alignment (q-path directionality + hard post-plant CT boundary)

## Why it outranks other major issues

Current base evidence shows a direct Bible mismatch in core round-win probability behavior:

- `engine/compute/q_intra_cs2.py` explicitly states "no bomb direction bias".
- Timer is read for debug gating, but not used to change `q` directionally.
- Live probe on current base:
  - neutral microstate (`alive=3v3`, `hp=300/300`) produced `q=0.5` at 90s/60s/30s/10s for both planted and not-planted states.
  - advantaged microstate (`alive=3v2`, `hp=300/250`) produced constant `q` across time; only a fixed bomb narrowing was applied.
- Full canonical suite passes (`394 passed`), which means this mismatch is currently under-detected by canonical tests and can silently persist.

This outranks:

- **Rails carryover extension**: rail v2 activation path already exists and is demonstrable on carryover-complete fixtures; unresolved work is primarily coverage expansion, not a core directional law breach.
- **Replay+simulation architecture refresh**: explicitly banked in `automation/BANKED_INITIATIVES.md`; reopen conditions are not yet compelling enough to outrank an active Chapter 5/7 directional mismatch in core q semantics.
- **Calibration campaigns**: tuning before timer-direction contract correctness risks fitting around a structural behavior error.
- **PHAT movement refinement**: movement sits downstream of q semantics; correcting q timer directionality is prerequisite to trustworthy movement validation.

## Bible progression justification

### Direct Bible mismatch addressed

This initiative directly targets a written mismatch against:

- Chapter 5 (Timer Pressure): pre-plant timer should favor CT as time decreases; post-plant should favor T.
- Chapter 7.2.3 (Timer Directionality): directional behavior is an expected invariant.
- Chapter 5 hard boundary: post-plant `time_remaining < defuse_time` implies CT round-win probability should collapse to zero.

### Specific next Bible-facing step this unlocks

Once timer-direction contract behavior is in place and validated, the next Bible-facing step becomes trustworthy **replay-based directional validation at scale** (Chapter 9 replay validation + timer directionality checks) without confounding from known q-path semantic drift.

### Why this outranks PHAT / rails / calibration / replay alternatives now

- **PHAT**: PHAT movement quality cannot be trusted if q semantics violate timer-direction law.
- **Rails**: current rail initiatives have bounded activation/fallback evidence and no current invariant failures; timer-direction currently has direct semantic contradiction.
- **Calibration**: calibrating weights before contract-level timer correctness risks encoding wrong directional behavior.
- **Replay architecture**: replay contract gates and diagnostics are functioning; the blocker is semantic correctness of the computed q signal under timer pressure.

### Why this is not subsystem drift / momentum bias / safe local cleanup

This is not a local cleanup continuation of recent rail/replay plumbing. It is a separate core-model behavior correction in the q path that:

- changes model-behavior semantics required by Bible chapters 5 and 7,
- requires coordinated updates across compute, resolve diagnostics, and canonical validation surfaces,
- removes a correctness blocker that invalidates interpretation of downstream replay and calibration results.

### What remains blocked or unreliable if skipped

If skipped:

- replay trajectories may appear stable while violating timer-direction semantics;
- calibration may optimize against mis-specified timer behavior;
- any claim of Bible alignment for round-resolution dynamics remains unreliable.

## Why it exceeds bounded-fix scope

This is not a one-file bugfix. It requires staged, cross-module work across:

1. q/timer scoring semantics and hard boundary logic,
2. resolve/diagnostics contracts,
3. source-specific handling (replay/BO3/GRID reliability differences),
4. scenario + replay validation architecture and acceptance gates.

That combination is architectural/model-behavior work in the deep lane, not bounded maintenance.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `engine/compute/q_intra_cs2.py` | Current timer behavior gap on base; baseline mismatch evidence. |
| `engine/compute/midround_v2_cs2.py` | Canonical q-path timer-direction and hard-boundary contract implementation surface. |
| `engine/compute/resolve.py` | Contract diagnostics propagation and integration into per-tick debug. |
| `engine/diagnostics/invariants.py` | Required timer diagnostics keys/reason-code contract. |
| `tests/unit/test_q_intra_cs2.py` | Baseline tests currently missing timer-direction contract checks. |
| `tests/unit/test_midround_v2_cs2.py` | Stage scenario matrix (directionality + boundary + forbidden behavior). |
| `tests/unit/test_resolve_micro_adj.py` | Resolve-level timer diagnostics and boundary assertions. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Contract-level key coverage checks for timer diagnostics. |
| `tools/replay_verification_assess.py` | Replay evidence checkpointing after timer contract alignment. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Replay summary contract remains verification target after timer changes. |

## Proposed stages

1. **Stage 1 — Timer contract compute + diagnostics (bounded)**
   - Implement directional timer term and hard post-plant CT boundary in q path.
   - Emit deterministic timer reason codes and required contract diagnostics keys.
   - Add scenario tests for pre/post-plant directionality and boundary behavior.

2. **Stage 2 — Source-parity and unsupported-slice policy hardening**
   - Validate deterministic behavior across replay/BO3/GRID source classes.
   - Keep unsupported GRID post-plant defuse-capability slices explicitly reason-coded (no silent fallback).
   - Add parity tests to ensure source-class behavior is intentional and auditable.

3. **Stage 3 — Replay-scale validation + regression gate**
   - Run replay assessment fixtures and selected larger raw replay sets.
   - Add timer-direction regression summaries (pre-plant CT trend, post-plant T trend, boundary activation counts).
   - Define acceptance thresholds for promotion readiness.

## Validation checkpoints

- **Checkpoint A (Stage 1 functional contract)**:
  - Directionality matrix passes (pre-plant CT-up, pre-plant T-down, post-plant T-up, post-plant CT-down under fixed non-timer inputs).
  - Hard boundary checks: `q_A=0` when A is CT under active boundary; `q_A=1` when A is T.
  - Required diagnostics keys and reason codes always present.

- **Checkpoint B (Stage 2 source parity)**:
  - Replay/BO3 behave identically for equivalent timer states.
  - GRID unsupported slices produce explicit reason-coded skips, not implicit behavior.

- **Checkpoint C (Stage 3 replay robustness)**:
  - Replay verification remains structurally clean (no new structural/invariant violations).
  - Timer-direction summary metrics show expected sign behavior across sampled trajectories.
  - Full canonical suite remains green.

## Risks

- **Side identity ambiguity risk** (`a_side` quality): wrong side mapping can invert directionality.
- **Source reliability risk**: missing or low-confidence timer/defuse capability fields may cause over-application unless strictly reason-coded.
- **Behavior sensitivity risk**: timer term magnitude can dominate if unbounded; needs strict bounded coefficients and diagnostics.
- **Validation false-confidence risk**: passing unit tests without replay-scale directional summaries may still miss realistic drift patterns.

## Recommended branch plan

- Keep this proposal on: `deep/proposal-20260306-2321-timer-pressure-contract`.
- On approval, execute Stage 1 on:
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s1-compute` (preferred fresh stage branch from `agent-initiative-base`), or
  - promote/refresh existing candidate `deep/stage-20260306-2245-timer-pressure-s1-compute` after revalidation on current base.
- Subsequent stages should use separate stage branches:
  - `deep/stage-...-timer-pressure-s2-source-parity`
  - `deep/stage-...-timer-pressure-s3-replay-gate`
- Do not merge autonomously; each stage requires human promotion decision.

## Recommendation

- [ ] **Approve planning only** — accept proposal; do not start implementation
- [x] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later

## Deferred alternatives (explicit)

- **Replay + Simulation Validation Architecture** remains deferred per `automation/BANKED_INITIATIVES.md` unless its reopen conditions are met (replay-validation stages exhausted, simulation becomes immediate blocker, or calibration/live validation no longer outrank).
