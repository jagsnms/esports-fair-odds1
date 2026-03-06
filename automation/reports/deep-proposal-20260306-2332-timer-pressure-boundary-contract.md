branch: deep/proposal-20260306-2332-timer-pressure-boundary-contract
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Timer Pressure + Defuse-Boundary Contract Alignment (Bible Ch. 5/6/7)

## Why it outranks other major issues

Current repo evidence shows a direct Bible mismatch in round-timer behavior:

- Controlled resolve probes on this branch baseline:
  - `preplant_ct_100s: q_intra=0.500000`
  - `preplant_ct_5s: q_intra=0.500000`
  - `preplant_t_100s: q_intra=0.500000`
  - `preplant_t_5s: q_intra=0.500000`
  - `postplant_ct_8s: q_intra=0.482320`
  - `postplant_ct_0.1s: q_intra=0.481272`
- This means:
  - pre-plant timer pressure is not directional (no CT-favoring ramp as time decays),
  - no hard post-plant CT-collapse boundary exists near bomb expiry.

By contrast, other major areas are currently less urgent based on fresh evidence:

- Canonical tests are green (`python3 -m pytest -q` -> `394 passed`).
- Replay contract/rails contract paths are currently stable in fixture validation:
  - raw/multimatch remain deterministic and invariant-clean,
  - carryover-complete replay activates v2 semantics when `--prematch-map 0.55` is supplied,
  - point-like replay is deterministically rejected under default policy.
- Banked initiative (`Replay + Simulation Validation Architecture`) remains deferred unless reopen conditions are met; it is noted as a deferred alternative, not selected here.

## Bible progression justification

### Direct Bible mismatch addressed

This initiative directly targets Bible-defined timer behavior:

- Chapter 5 + 7.2.3: timer directionality must be side- and phase-correct (pre-plant CT pressure, post-plant T pressure).
- Chapter 5 hard boundary: if post-plant time remaining is below defuse feasibility, CT win probability must collapse to zero.

Current compute path does not enforce that contract:

- `engine/compute/q_intra_cs2.py` records time usage flags but does not inject timer direction into score.
- `engine/compute/midround_v2_cs2.py` uses time as symmetric urgency scaling, not a side/phase directional timer term.
- No explicit defuse-feasibility boundary is modeled in round probability outputs.

### Specific next Bible-facing step this unlocks

After this initiative, the next step becomes trustworthy replay/statistical validation of Chapter 7.2.3 (timer directionality) on canonical replay fixtures and real logs, instead of validating a timer model that is structurally non-compliant.

### Why this outranks PHAT / rails / calibration / replay right now

- **PHAT movement:** existing contract diagnostics and tests are green for current movement architecture; timer semantics are upstream of trustworthy q behavior.
- **Rails:** recent rail contract/v2 activation stages show bounded, deterministic behavior and no active invariant break in current fixtures.
- **Calibration:** calibrating weights before timer-direction semantics are correct risks fitting to wrong structure (Bible Ch. 8 forbids structural drift via calibration).
- **Replay architecture:** replay validation is functional enough to expose this timer mismatch; replay+simulation architecture is also explicitly banked/deferred.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

This is not a continuation of rail/replay local cleanup. It cuts across compute semantics, ingest contracts, diagnostics, and replay validation policy for a core Bible behavioral invariant. The trigger is a direct contract mismatch, not convenience or recent file familiarity.

### What remains blocked or unreliable if skipped

If skipped:

- timer-direction invariant results remain unreliable (false confidence from passing suites that do not encode the Bible timer contract),
- post-plant CT boundary behavior remains structurally incorrect,
- downstream calibration/replay conclusions remain confounded by missing timer semantics.

## Why it exceeds bounded-fix scope

This is major-initiative scope, not a bounded patch, because it requires coordinated changes to:

1. q semantics (timer directional term + boundary),
2. ingest/model contract for timer fields needed by the boundary,
3. diagnostics/invariant taxonomy and replay evidence surfaces,
4. cross-source parity tests (BO3/GRID/REPLAY raw).

A one-file tweak would create partial behavior and hidden drift without contract-level validation.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `engine/compute/q_intra_cs2.py` | Add explicit timer-direction term contract and hard post-plant boundary logic hooks. |
| `engine/compute/midround_v2_cs2.py` | Align urgency/timer interaction with side- and phase-aware semantics; avoid symmetric-only scaling. |
| `engine/compute/resolve.py` | Ensure target/movement path consumes canonical timer-aware q behavior consistently; expose reason codes. |
| `engine/diagnostics/invariants.py` | Add timer-direction/defuse-boundary behavioral diagnostics and violation reason codes. |
| `engine/models.py` | Extend canonical frame contract if additional timer fields are needed (bomb timer/defuse feasibility indicators). |
| `engine/normalize/bo3_normalize.py` | Ingest and normalize timer-related raw fields required for boundary logic and diagnostics. |
| `engine/ingest/grid_reducer.py` | Source-parity mapping for timer fields and phase semantics. |
| `tools/replay_verification_assess.py` | Add timer-contract summary counters for replay evidence packs. |
| `tools/schemas/replay_validation_summary.schema.json` | Add required timer-contract observability keys. |
| `tests/unit/test_q_intra_cs2.py` | Replace time-flag-only assertions with timer-directionality contract tests. |
| `tests/unit/test_midround_v2_cs2.py` | Add side/phase timer pressure monotonic tests and post-plant boundary tests. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Add timer-specific behavioral violation assertions. |
| `tests/unit/test_runner_source_contract_parity.py` | Ensure source parity for timer contract fields. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Validate replay summary timer-contract schema keys and deterministic counts. |

## Proposed stages

1. **Stage 0 — Contract lock + evidence harness (proposal-to-implementation gate)**
   - Freeze timer semantics contract (pre-plant CT pressure, post-plant T pressure, hard boundary definition).
   - Add deterministic probe fixtures and baseline evidence script outputs.
2. **Stage 1 — Compute timer-direction core**
   - Implement side/phase directional timer term in q path.
   - Implement hard post-plant CT boundary guard (explicit reason code).
   - Keep rails and non-timer PHAT semantics unchanged.
3. **Stage 2 — Ingest/source parity + diagnostics**
   - Normalize required timer inputs across BO3/GRID/REPLAY raw.
   - Add invariant diagnostics for timer direction failures and boundary violations.
4. **Stage 3 — Replay validation contract expansion**
   - Emit timer-contract counters in replay assessment artifacts/schema.
   - Validate deterministic behavior on canonical fixtures and available logs.
5. **Stage 4 — Calibration handoff gate**
   - Produce a bounded report proving timer contract compliance and stability to unlock safe calibration work.

## Validation checkpoints

- Baseline and regression:
  - `python3 -m pytest -q`
- Timer contract unit checks (new/expanded):
  - `python3 -m pytest -q tests/unit/test_q_intra_cs2.py tests/unit/test_midround_v2_cs2.py tests/unit/test_invariants_contract_diagnostics.py`
- Source parity and replay evidence:
  - `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py`
  - `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
  - `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`
  - `python3 tools/replay_verification_assess.py tools/fixtures/replay_carryover_complete_v1.jsonl --prematch-map 0.55`
- Deterministic timer probe script (new):
  - fixed-state pre/post-plant time sweep with expected directional monotonic outcomes and CT hard-boundary assertions.

## Risks

- **Data availability risk:** some replay sources may not provide explicit bomb-timer/defuse fields; boundary logic must handle missingness deterministically.
- **Over-correction risk:** aggressive timer weighting could swamp combat signals (Bible Ch. 8 hierarchy constraint).
- **Cross-source divergence risk:** BO3 vs GRID timer semantics can drift without strict parity tests.
- **Compatibility risk:** existing dashboards/consumers may assume current timer-neutral behavior in edge cases.

## Recommended branch plan

- Keep this proposal branch for planning only:
  - `deep/proposal-20260306-2332-timer-pressure-boundary-contract`
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s0-contract-lock`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s1-compute-core`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s2-ingest-parity`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s3-replay-validation`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s4-calibration-handoff`
- Do not batch multiple stages in one run; require per-stage evidence report under `automation/reports/`.

## Deferred alternatives considered

- **Replay + Simulation Validation Architecture** (banked): still relevant long-term but not selected this run; timer-contract mismatch is currently a more direct Bible violation and immediate blocker to trustworthy replay/calibration conclusions.

## Recommendation

- [x] **Approve planning only** — accept proposal; do not start implementation
- [ ] **Approve stage 1** — approve first stage for implementation
- [ ] **Defer** — do not approve; revisit later

