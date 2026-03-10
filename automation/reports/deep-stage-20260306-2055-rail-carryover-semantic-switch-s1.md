branch: deep/stage-20260306-2055-rail-carryover-semantic-switch-s1
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: promote

# Stage report — Rail Carryover Semantic Switch (Stage 1)

## Objective (locked)

Switch contract rail endpoint semantics in `engine/compute/rails_cs2.py` from Stage-1 lock (`observe_v2_use_v1_endpoints`) to Stage 1 policy-gated semantics with:

- only `force_v1` and `v2_strict` policy states,
- deterministic fallback reason codes,
- v2 activation only when all required fields are valid,
- preserved structural invariants and bounded rollout.

## Files changed

| Path | Change |
|---|---|
| `engine/compute/rails_cs2.py` | Implemented semantic-switch policy gate (`force_v1` / `v2_strict`), deterministic reason codes, v2 activation marker, required-field validity enforcement, and carryover-conditioned endpoint adjustment when activated. |
| `tests/unit/test_rails_input_contract.py` | Reworked tests for policy-state constraints, activation/fallback determinism, carryover sensitivity, forbidden transient invariance, and BO3/GRID/REPLAY parity for equivalent inputs. |
| `tests/unit/test_rails_cs2_basic.py` | Added v2-active map-point alignment test to confirm structural map-point boundary behavior remains intact. |
| `tools/replay_verification_assess.py` | Added deterministic rail policy observability counters: policy counts, semantics counts, reason-code counts, and v2-activated vs v1-fallback point counts. |
| `tools/schemas/replay_validation_summary.schema.json` | Added required schema keys for rail policy/reason-code observability outputs. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertions for deterministic rail policy/reason observability in replay summary artifacts. |

## Behavior changed

1. **Policy states are now explicit and bounded**
   - Supported states:
     - `force_v1`
     - `v2_strict`
   - Unsupported values deterministically fall back to v1 with reason code `V2_POLICY_UNSUPPORTED`.

2. **v2 activation is strict**
   - In `v2_strict`, v2 semantics activate only when all required fields are present and valid.
   - Otherwise deterministic fallback to v1 with:
     - `V2_REQUIRED_FIELDS_MISSING` or
     - `V2_REQUIRED_FIELDS_INVALID`.
   - Activation marker when active:
     - `V2_STRICT_ACTIVATED`.

3. **force_v1 is deterministic**
   - `force_v1` always uses v1 semantics with reason:
     - `POLICY_FORCE_V1`.

4. **Contract endpoint semantics now switch when v2 is active**
   - Under v2 activation, contract endpoint calculations use carryover-conditioned adjustment based on allowed persistent fields (`cash_totals`, `loadout_totals`, `armor_totals`, plus score/series/prematch context).
   - Forbidden transient signals remain excluded from endpoint semantics.

5. **Structural safeguards preserved**
   - Rail ordering (`rail_low <= rail_high`)
   - bounds/[0,1] containment
   - map-point alignment checks
   - contract min-width epsilon behavior

6. **Replay artifact observability expanded**
   - Replay assessment now emits deterministic rail policy/reason-code counters for before/after comparison.

## Before/after evidence

### Baseline pack (before)

1. `python3 -m pytest -q tests/unit/test_rails_cs2_basic.py tests/unit/test_rails_input_contract.py`  
   - **30 passed**
2. `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py`  
   - **2 passed**
3. `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`  
   - `raw_contract_points=3`, `structural_violations_total=0`, no rail policy/reason summary keys yet.
4. `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`  
   - `raw_contract_points=6`, `structural_violations_total=0`, no rail policy/reason summary keys yet.

### Baseline pack (after)

1. `python3 -m pytest -q tests/unit/test_rails_cs2_basic.py tests/unit/test_rails_input_contract.py`  
   - **31 passed**
2. `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py`  
   - **2 passed**
3. `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`  
   - `raw_contract_points=3`, `structural_violations_total=0`  
   - new deterministic outputs:
     - `rail_input_contract_policy_counts={"v2_strict":3}`
     - `rail_input_active_endpoint_semantics_counts={"v1":3}`
     - `rail_input_reason_code_counts={"V2_REQUIRED_FIELDS_MISSING":3}`
     - `rail_input_v2_activated_points=0`
     - `rail_input_v1_fallback_points=3`
4. `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`  
   - `raw_contract_points=6`, `structural_violations_total=0`  
   - new deterministic outputs:
     - `rail_input_contract_policy_counts={"v2_strict":6}`
     - `rail_input_active_endpoint_semantics_counts={"v1":6}`
     - `rail_input_reason_code_counts={"V2_REQUIRED_FIELDS_MISSING":6}`
     - `rail_input_v2_activated_points=0`
     - `rail_input_v1_fallback_points=6`

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| Only `force_v1` and `v2_strict` policy states | ✅ met | `rails_cs2.py` policy constants + unsupported-policy fallback test |
| v2 activates only when all required fields valid | ✅ met | `test_v2_strict_activates_when_required_complete` + strict validation path |
| Deterministic fallback/activation reason codes | ✅ met | `force_v1`, missing, invalid, unsupported, activated tests |
| Forbidden transient invariance for contract endpoints | ✅ met | `test_forbidden_perturbation_invariance` |
| Carryover sensitivity under fixed score/series | ✅ met | `test_carryover_sensitivity_when_v2_active` |
| BO3/GRID/replay raw parity for equivalent required inputs | ✅ met | `test_bo3_vs_grid_contract_parity` (includes REPLAY source parity at compute contract level) |
| Replay artifact/schema emits policy + reason observability deterministically | ✅ met | replay assessment script/schema updates + replay summary test assertions |
| Structural invariants remain green | ✅ met | post-change baseline pack (all tests green; replay structural violations remain 0) |

Partially met criteria: **none**.

## Validation

- `python3 -m pytest -q tests/unit/test_rails_cs2_basic.py tests/unit/test_rails_input_contract.py` → `31 passed`
- `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py` → `2 passed`
- `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl` → deterministic summary with rail policy/reason counters
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` → deterministic summary with rail policy/reason counters

## Risks / scope pressure

- Current replay fixtures still trigger `V2_REQUIRED_FIELDS_MISSING` fallback (expected) because they do not provide full required carryover vectors; this is an observability outcome, not a structural failure.
- Scope pressure to expand replay ingestion/backfill was explicitly resisted in Stage 1 to stay inside the frozen gate.

## Frozen-gate compliance statement

Stage 1 stayed inside the frozen Stage 0 gate:

- no PHAT resolve/movement changes,
- no calibration work,
- no replay/simulation architecture redesign,
- no broad runner refactor,
- no partial-v2 mode.

## Recommendation

**promote**
