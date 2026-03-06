branch: deep/proposal-20260306-0807-replay-contract-unification-s2
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Replay Contract Unification — Decision-Gate Memo (planning only)

## 1) Contract-gate decision to make

Point-like replay inputs currently enter runtime via a non-canonical passthrough branch (`_tick_replay_point_passthrough`) while raw BO3-shaped replay goes through canonical compute. The gate decision is:

- **Option A:** convert point-like inputs to canonical frame execution
- **Option B:** reject point-like inputs with explicit reason codes
- **Option C:** use a tightly-scoped transition mode only if needed for compatibility

No implementation is proposed in this memo.

## 2) Option analysis

### Option A — Convert point-like inputs into canonical frame path

**Bible alignment**
- Mixed. End-state goal (single canonical execution path) aligns with Chapter 6 pipeline and Chapter 9 replay contract.
- Conversion itself can violate intent if it fabricates missing round-state signals.

**Semantic risk**
- High.
- Current point structure in repo sources (`logs/history_points.jsonl`) has no raw snapshot primitives (`team_one/team_two`, clock state, player-state arrays, carryover state).
- Reconstructed "frames" would be inferred artifacts, not original round state.

**Migration/compatibility risk**
- Medium to high.
- May preserve API compatibility superficially while silently changing semantics.
- Backfilled synthetic frame assumptions can drift across model versions.

**Observability/validation implications**
- Hard to audit if conversion quality is not explicitly scored.
- Requires conversion diagnostics (source fidelity tiers, inferred-field flags, per-point confidence classes).

**Long-term maintenance cost**
- High.
- Maintains a conversion subsystem tied to evolving frame/compute schema.
- Ongoing burden each time canonical frame contract changes.

### Option B — Reject point-like inputs with explicit reason codes

**Bible alignment**
- Strongest alignment.
- Enforces single canonical replay semantics and prevents non-contract execution in replay validation workflows.

**Semantic risk**
- Low.
- No synthetic reconstruction; avoids hidden model behavior drift from inferred data.

**Migration/compatibility risk**
- High short-term.
- Existing point-oriented replay workflows (history-based replays) will fail unless transitioned.

**Observability/validation implications**
- Strong.
- Rejection reasons are deterministic and measurable.
- Replay-validation metrics become cleaner (no ambiguous mixed-mode points).

**Long-term maintenance cost**
- Low.
- Simplifies replay architecture and test matrix.

### Option C — Tightly-scoped temporary transition mode (only if necessary)

**Bible alignment**
- Acceptable only as explicitly temporary policy with an expiration gate.
- Must not be treated as canonical validation behavior.

**Semantic risk**
- Medium.
- Retains non-canonical path but can contain risk if mode is explicit, opt-in, quarantined, and excluded from canonical validation success criteria.

**Migration/compatibility risk**
- Lowest near-term.
- Preserves legacy consumers while migration is completed.

**Observability/validation implications**
- Requires strict policy instrumentation:
  - explicit mode in config/status
  - hard tagging + counters
  - deadline/deprecation checkpoints
  - default-off in validation paths

**Long-term maintenance cost**
- Medium to high if not sunset aggressively.
- Temporary modes tend to become permanent unless hard stop criteria exist.

## 3) Conversion feasibility: lossless/lossy classification

## Repo evidence used

- Raw fixtures are BO3-shaped snapshots:
  - `tools/fixtures/raw_replay_sample.jsonl`
  - `tools/fixtures/replay_multimatch_small_v1.jsonl`
- Point replay source in workspace:
  - `logs/history_points.jsonl`
- Point structure profile in current workspace (`history_points.jsonl`):
  - total records sampled: 51
  - has raw team snapshot fields (`team_one`/`team_two`): 0
  - has raw replay keys (`round_phase`/`match_fixture` at top-level): 0
  - explain phase mix includes `IN_PROGRESS`, `stale_snapshot`, `paused`, `replay_loop_boundary`, `inter_map_break`, `replay_passthrough`
  - several records missing explain or missing numeric `q_intra_total`

### Classification

- **Raw BO3 replay entries:** lossless for canonical replay execution (already canonical shape).
- **Current point-like entries from history sources:** **lossy to non-convertible** for true canonical replay:
  - no full round-state input to re-run normalize/reduce/compute faithfully
  - non-compute phases have insufficient signal content for reconstruction
  - any conversion would be inference, not recovery
- **Conditionally lossless conversion is only possible** if future point records embed a complete canonical raw snapshot payload (not currently present in repo fixtures/sources and not required by current loaders).

## 4) Minimum repo surfaces by option

### Option A (convert)

- **runner**
  - replace or bypass `_tick_replay_point_passthrough` with conversion pipeline
  - conversion diagnostics and fidelity class tagging
- **replay API/source discovery**
  - keep point sources discoverable; annotate as converted legacy source
- **config/schema**
  - explicit conversion mode and strict defaults
  - schema fields for conversion fidelity/reason flags
- **assessment/reporting**
  - add conversion counters, inferred-field counts, failure taxonomy
- **tests/fixtures**
  - conversion-path tests for each fidelity class; mixed input behavior; determinism checks

### Option B (reject)

- **runner**
  - hard reject point-like payloads with deterministic reason codes
  - no append/broadcast for rejected point input
- **replay API/source discovery**
  - either hide point files from replay candidates or mark unsupported for canonical replay
- **config/schema**
  - no conversion mode needed; optional explicit `reject_point_like_replay=true`
  - reason-code schema for rejection diagnostics
- **assessment/reporting**
  - rejection counts/reasons become first-class outputs
- **tests/fixtures**
  - tests asserting rejection and reason-code stability
  - preserve raw fixture tests unchanged

### Option C (temporary transition)

- **runner**
  - explicit opt-in legacy transition mode only
  - default behavior follows reject policy
- **replay API/source discovery**
  - point sources visible only with transition labeling/warnings
- **config/schema**
  - required explicit transition mode with sunset marker
  - surface transition mode in status payload
- **assessment/reporting**
  - transition-use counters + hard budget/threshold alerts
- **tests/fixtures**
  - tests for default-off transition, explicit-on behavior, deprecation gating

## 5) Narrowest approvable Stage 1 after policy selection

Stage 1 must not implement full architecture migration. Narrowest bounded stage:

1. **Policy lock + contract shape lock (no broad runtime migration)**
   - choose one policy (A/B/C) and freeze reason-code taxonomy
   - define canonical replay acceptance contract and non-canonical handling contract
2. **One explicit runtime gate surface**
   - one explicit config/status policy field
   - one deterministic policy decision emitted per replay point attempt
3. **One bounded evidence pack**
   - targeted tests proving policy gate behavior only
   - replay assessment output includes policy counters required by gate

This Stage 1 is reversible (single policy gate), reviewable (small surface), and testable (deterministic reason codes/counters).

## 6) Acceptance criteria to approve implementation after planning

Implementation should not be approved until all are true:

1. **Policy chosen**: A, B, or C explicitly selected with stated default mode.
2. **Reason-code contract frozen**: deterministic machine-readable reason strings and category map approved.
3. **Loss model acknowledged**: if A is chosen, conversion fidelity classes and loss assumptions explicitly approved.
4. **Validation contract frozen**:
   - canonical replay success must not include non-canonical passthrough points
   - policy counters required in replay assessment output
5. **Stage 1 boundary frozen**:
   - no broad compute/refactor work
   - no simulation-framework expansion
6. **Rollback plan defined**:
   - one-step fallback path documented (disable new policy gate behavior or revert stage branch)

## 7) Hidden assumptions currently embedded in key files

### `backend/services/runner.py`

- Replay format is inferred from first payload (`raw` vs `point`), not declared by contract.
- In raw mode, point-like mixed lines are skipped; in point mode, point-like lines are accepted and appended.
- Point passthrough appends to history and may broadcast by default (`getattr(config, "replay_emit_quarantined_points", True)`), even though this field is not in canonical `Config`.
- Point passthrough marks records as non-canonical but still treats them as replay outputs.

### `backend/api/routes_replay.py`

- Replay source discovery advertises `history_points*.jsonl` as valid `kind="point"` replay sources.
- `/replay/load` has no policy field to declare convert/reject/transition mode.
- `/replay/matches` is BO3-entry-oriented; point files are discoverable in sources but often non-actionable for match listing, creating implicit ambiguity.

### `tests/unit/test_runner_replay_contract_mode.py`

- Tests currently codify point passthrough as expected behavior (quarantine-tagged acceptance), not as exceptional transition-only behavior.
- This anchors non-canonical replay into the contract unless rewritten after policy decision.

### `tools/replay_verification_assess.py`

- Assessment reports point/non-canonical counters but does not enforce pass/fail policy gates.
- Replay mode is inferred from first payload similarly to runner injection path.
- Non-canonical replay can appear in outputs without automatic failure semantics.

## Decision recommendation from this planning pass

- Preferred long-term policy: **Option B (reject point-like replay for canonical replay execution)**.
- If compatibility risk is judged unacceptable for immediate cutover, use **Option C only as a time-boxed transition wrapper around Option B default**.
- Do not pursue Option A conversion unless a new point schema carries full canonical raw snapshot payloads (currently not true in repo evidence).
