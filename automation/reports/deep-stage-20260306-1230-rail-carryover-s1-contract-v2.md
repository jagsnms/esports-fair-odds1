# Deep Stage 1: Rail carryover S1 contract v2 (observability only)

**Branch:** deep/stage-20260306-1230-rail-carryover-s1-contract-v2  
**Base:** agent-initiative-base  
**Date:** 2026-03-06

## Objective

Define rail input contract v2 (carryover-observability) and emit deterministic coverage/fallback diagnostics, while keeping current rail endpoint math exactly as-is (v1 semantics). Stage 1 proves whether v2 carryover inputs are present/trustworthy for later semantic migration.

## Files changed

- **engine/compute/rails_cs2.py**
  - Added v2 contract constants: `RAIL_INPUT_V2_CONTRACT_VERSION`, `RAIL_INPUT_V2_POLICY`, `RAIL_INPUT_V2_REQUIRED_FIELDS`, `RAIL_INPUT_V2_OPTIONAL_FIELDS`, `RAIL_INPUT_V2_FORBIDDEN_FIELDS`, and fallback reason codes.
  - Added `_v2_required_field_valid()` for deterministic required-field validation.
  - Added `_rail_input_v2_provenance()` returning all required v2 observability keys (version, policy, required/optional/forbidden lists, present/missing/invalid required, present optional, coverage ratio, required_complete, v1_fallback_used, fallback_reason_code, active_endpoint_semantics, source, replay_kind).
  - In `compute_rails_cs2()`, added optional kwargs `source=None`, `replay_kind=None`; call `_rail_input_v2_provenance()` and merge into debug after v1 provenance. v1 provenance keys retained for backward compatibility.
- **tests/unit/test_rails_input_contract.py**
  - Updated for v2: assert `rail_input_contract_version` == v2-observe-stage1, v1 keys still present, v2 keys (policy, fallback_used, active_endpoint_semantics, v2_required_fields, coverage_ratio, fallback_reason_code) present.
  - Updated BO3/GRID parity to assert v2 version and fallback reason/coverage parity.
  - Added: `test_v2_fallback_always_used`, `test_v2_fallback_reason_missing_when_required_absent`, `test_v2_fallback_reason_stage1_locked_when_required_complete`, `test_v2_fallback_reason_invalid_when_required_bad_type`.

## Behavior changed

- **Observability only.** Every `compute_rails_cs2` evaluation now also emits v2 diagnostics: `rail_input_contract_version` = "v2-observe-stage1", `rail_input_contract_policy` = "observe_v2_use_v1_endpoints", v2 required/optional/forbidden field lists, `rail_input_v2_present_required_fields`, `rail_input_v2_missing_required_fields`, `rail_input_v2_invalid_required_fields`, `rail_input_v2_present_optional_fields`, `rail_input_v2_required_coverage_ratio`, `rail_input_v2_required_complete`, `rail_input_v1_fallback_used` = True, `rail_input_v1_fallback_reason_code` (V2_REQUIRED_FIELDS_MISSING | V2_REQUIRED_FIELDS_INVALID | STAGE1_LOCKED_NO_SEMANTIC_SWITCH), `rail_input_active_endpoint_semantics` = "v1", `rail_input_source`, `rail_input_replay_kind`. v1 provenance keys (allowed_fields, forbidden_fields, allowed_consumed, forbidden_ignored) remain in debug.
- **Rail endpoint behavior unchanged.** Contract rail formulas and returned `(rail_lo, rail_hi)` are identical for the same inputs. Stage 1 always uses v1 endpoints; fallback reason is deterministic (missing → invalid → stage-lock).

## Acceptance criteria status

- v2 required/optional/forbidden sets explicit in code: **met**
- Every rail evaluation emits v2 contract version/policy and presence/missing/invalid/fallback fields: **met**
- Fallback reason codes deterministic for missing vs invalid vs stage-lock: **met** (tests cover all three)
- BO3/GRID/replay raw produce deterministic classification/coverage: **met** (same inputs → same v2 diagnostics; parity test)
- Existing rail endpoint outputs unchanged for fixed inputs: **met**
- Existing canonical rail/resolve tests remain green: **met**

## Validation

- `python -m pytest -q tests/unit/test_rails_input_contract.py`: **10 passed**
- `python -m pytest -q tests/unit/test_runner_source_contract_parity.py`: **1 passed**
- `python -m pytest -q tests/unit/test_replay_verification_assess_stage1.py`: **1 passed**
- `python -m pytest -q tests/unit/test_rails_cs2_basic.py tests/unit/test_resolve_micro_adj.py`: **48 passed**
- `python tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`: **OK**
- `python -m pytest -q`: **374 passed, 5 failed** (pre-existing test_runner_map_identity.py asyncio/Windows; unrelated to this stage)

## Risks

- **None.** No formula or endpoint changes. v2 diagnostics are additive. `rail_input_source` and `rail_input_replay_kind` are None unless callers pass them (rails.py/runner unchanged in Stage 1).

## Out-of-scope preserved

- No rail endpoint formula redesign.
- No PHAT/movement/timer changes.
- No calibration/tuning.
- No replay/simulation architecture expansion.
- No changes to backend/services/runner.py, engine/ingest/grid_reducer.py, tools/replay_verification_assess.py, or tests/unit/test_replay_verification_assess_stage1.py (beyond existing tests).

## Rail endpoint semantics

Rail endpoint semantics remained v1. Outputs were not silently changed: same inputs produce the same `(rail_lo, rail_hi)` and same contract rail values; only debug dict is extended with v2 observability.

## Recommendation

**Promote.** Stage 1 is bounded, reviewable, and testable. Rail endpoint semantics are unchanged.
