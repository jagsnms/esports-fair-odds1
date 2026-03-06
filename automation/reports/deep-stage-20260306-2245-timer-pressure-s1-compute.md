branch: deep/stage-20260306-2245-timer-pressure-s1-compute
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: promote

# Stage report — Timer-pressure contract (Stage 1 compute)

## Objective (locked)

Implement the frozen Stage 0 timer-direction contract and hard post-plant CT boundary in the q/timer compute path, with deterministic diagnostics and bounded tests, without rail/bounds redesign or replay-architecture work.

## Files changed

| Path | Change |
|---|---|
| `engine/compute/midround_v2_cs2.py` | Added frozen timer-contract constants/reason codes, source/timer-state classification, timer-direction term logic in q score path, hard post-plant boundary override (`q_CT=0`), and deterministic timer diagnostics fields. |
| `engine/compute/resolve.py` | Passed `frame` into midround mixture call and forwarded timer-contract diagnostics into `contract_diagnostics`. |
| `engine/diagnostics/invariants.py` | Extended `compute_phat_contract_diagnostics` to emit required timer diagnostics keys deterministically (with stable defaults). |
| `tests/unit/test_midround_v2_cs2.py` | Added S1–S10 scenario tests and forbidden-behavior tests for wrong direction, boundary leakage, and silent fallback reason-code absence. |
| `tests/unit/test_resolve_micro_adj.py` | Added required timer key assertions at resolve-level diagnostics and boundary q-contract tests (`q_A=0/1` under active hard boundary). |
| `tests/unit/test_invariants_contract_diagnostics.py` | Added explicit timer diagnostics key coverage test. |

## Behavior changed

1. **Timer-direction semantics are now explicit in q pathway**
   - Pre-plant: timer term direction favors CT.
   - Post-plant: timer term direction favors T.
   - Direction is mapped through `a_side` so `q_A` moves in the required direction under fixed non-timer inputs.
   - Timer direction responds in q-score space only (no rails/bounds timer coupling added).

2. **Hard post-plant CT boundary is implemented in q**
   - When post-plant timer is valid and below effective CT defuse threshold:
     - `q_CT = 0.0` exactly.
     - Thus `q_A = 0.0` when Team A is CT, `q_A = 1.0` when Team A is T.
   - Defuse threshold policy:
     - 5s with reliable kit-capable CT evidence,
     - 10s with reliable no-kit-only CT evidence,
     - 5s conservative floor for low-confidence/unknown kit state.

3. **Source-aware bounded behavior**
   - Replay/BO3 normalized paths support timer direction and post-plant boundary when required fields are valid.
   - GRID post-plant boundary (and associated direction action for that unsupported post-plant contract slice) is skipped deterministically with unsupported reason code when required defuse-capability reliability is unavailable.

4. **Frozen diagnostics contract emitted**
   - Required keys are emitted via `contract_diagnostics`:
     - `timer_contract_version`, `timer_state`, `timer_source_class`, `timer_remaining_s`, `timer_valid`, `a_side_used`, `timer_direction_expected`, `timer_direction_applied`, `timer_direction_term`, `timer_direction_reason_code`, `defuse_time_s`, `defuse_time_source`, `hard_boundary_active`, `hard_boundary_reason_code`.
   - Frozen reason codes are emitted deterministically for applied and skipped paths.

## Scenario results

From targeted unit scenarios added in `tests/unit/test_midround_v2_cs2.py`:

- **S1_PRE_A_CT**: pass (`q_A` non-decreasing as timer decreases pre-plant when A=CT)
- **S2_PRE_A_T**: pass (`q_A` non-increasing as timer decreases pre-plant when A=T)
- **S3_POST_A_T_NONBOUND**: pass (`q_A` non-decreasing as timer decreases post-plant, boundary inactive)
- **S4_POST_A_CT_NONBOUND**: pass (`q_A` non-increasing as timer decreases post-plant, boundary inactive)
- **S5_BOUNDARY_A_CT**: pass (active boundary sets `q_A=0.0`)
- **S6_BOUNDARY_A_T**: pass (active boundary sets `q_A=1.0`)
- **S7_POST_ABOVE_THRESHOLD**: pass (boundary inactive above threshold)
- **S8_TIMER_MISSING**: pass (direction + boundary skipped with required missing reason codes)
- **S9_TIMER_INVALID**: pass (direction + boundary skipped with required invalid reason codes)
- **S10_GRID_POST_UNSUPPORTED**: pass (deterministic unsupported reason codes)

Forbidden-behavior tests:

- wrong-direction response fails: pass (forbidden behavior absent)
- boundary leakage fails: pass (`q_A` does not leak above contract value when active)
- silent fallback without reason code fails: pass (reason codes always present)

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| Timer-direction contribution implemented in q path | ✅ met | `midround_v2_cs2.py` timer-direction term + S1–S4 tests |
| Hard post-plant CT boundary implemented in q | ✅ met | boundary override logic + S5/S6/S7 tests + resolve boundary tests |
| Frozen diagnostics keys/reason codes emitted | ✅ met | `invariants.py` diagnostics extension + resolve/invariants tests |
| S1–S10 scenario matrix covered | ✅ met | new unit scenarios in `test_midround_v2_cs2.py` |
| Forbidden behavior tests included | ✅ met | three forbidden-behavior tests added and passing |
| Structural invariants preserved | ✅ met | full suite pass (`411 passed`) |
| No unintended rails/replay contract regressions | ✅ met | targeted regression bundle (`24 passed`) |

Partially met criteria: **none**.

## Validation

- `python3 -m pytest -q tests/unit/test_midround_v2_cs2.py` → `35 passed`
- `python3 -m pytest -q tests/unit/test_resolve_micro_adj.py` → `31 passed`
- `python3 -m pytest -q tests/unit/test_invariants_contract_diagnostics.py` → `5 passed`
- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py tests/unit/test_rails_input_contract.py tests/unit/test_runner_replay_contract_mode.py tests/unit/test_replay_status_contract_gate.py tests/unit/test_replay_sources_contract_signaling.py` → `24 passed`
- `python3 -m pytest -q` → `411 passed`

## Risks / scope pressure

- Current GRID post-plant boundary support remains intentionally conservative and may skip activation when defuse-capability reliability is unavailable; this is expected under frozen Stage 0 source policy.
- Timer directional weight is bounded but still a model sensitivity lever; any future tuning must be handled as a separate approved calibration stage.

## Frozen-gate compliance statement

Stage 1 stayed inside the frozen gate:

- no rail/bounds redesign,
- no replay/simulation architecture work,
- no calibration/weight-tuning campaign,
- no broad runner/source refactor,
- no CI threshold changes,
- no unrelated cleanup/refactor bundle.

## Recommendation

**promote**
