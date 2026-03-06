# Deep Stage 1: Rails carryover S1 contract (observability only)

**Branch:** deep/stage-20260306-1145-rails-carryover-s1-contract  
**Base:** agent-initiative-base  
**Date:** 2026-03-06

## Objective

Define a versioned, explicit rail input contract for contract rails and expose per-evaluation provenance (allowed/forbidden fields, consumed/ignored) without changing endpoint math.

## Files changed

- **engine/compute/rails_cs2.py**
  - Added `RAIL_INPUT_CONTRACT_VERSION` ("v1"), `RAIL_INPUT_ALLOWED_FIELDS`, `RAIL_INPUT_FORBIDDEN_FIELDS`.
  - Added `_rail_input_provenance()` helper returning the five provenance keys.
  - In `compute_rails_cs2()`, call provenance helper and merge result into the returned `debug` dict for every evaluation.
- **tests/unit/test_rails_input_contract.py** (new)
  - Tests: provenance keys emitted, forbidden-input perturbation invariance, allowed consumed list, forbidden ignored when present, allowed input change changes rails, BO3 vs GRID contract parity.

## Behavior changed

- **Observability only.** Every call to `compute_rails_cs2` now includes in `debug`:
  - `rail_input_contract_version`: "v1"
  - `rail_input_allowed_fields`: list of allowed field names
  - `rail_input_forbidden_fields`: list of forbidden field names
  - `rail_input_allowed_consumed`: list of allowed inputs actually used for contract rails (bounds.low/high, frame.scores, frame.series_score, frame.series_fmt, config.prematch_map)
  - `rail_input_forbidden_ignored`: list of forbidden fields present on the frame (and thus ignored for contract rails)
- **Rail endpoint behavior unchanged.** Contract rail formulas, clamp logic, and returned `(rail_lo, rail_hi)` are identical for the same allowed inputs. Heuristic rails path unchanged.

## Acceptance criteria status

- Contract v1 explicit in code with named allowed and forbidden sets: **met**
- Every contract-rail evaluation emits the five provenance keys: **met**
- Perturbing forbidden inputs while holding allowed fixed does not change rail_low/rail_high: **met** (test_forbidden_perturbation_invariance)
- Perturbing allowed inputs reflected in provenance as consumed and may change rails: **met** (test_allowed_input_change_changes_rails, test_allowed_consumed_reflects_contract)
- BO3 vs GRID parity: identical allowed/forbidden classification and rails: **met** (test_bo3_vs_grid_contract_parity)
- Existing rail/resolve contract tests remain green: **met**
- Rail outputs unchanged before/after for fixed allowed inputs: **met**

## Validation

- `python -m pytest -q tests/unit/test_rails_input_contract.py`: **6 passed**
- `python -m pytest -q tests/unit/test_rails_cs2_basic.py`: **20 passed**
- `python -m pytest -q tests/unit/test_resolve_micro_adj.py`: **28 passed**
- `python -m pytest -q tests/unit/test_runner_source_contract_parity.py`: **1 passed**
- `python -m pytest -q`: **370 passed, 5 failed** (failures are pre-existing: test_runner_map_identity.py asyncio event loop on Windows; unrelated to this stage)

## Risks

- **None.** No formula or endpoint changes. Provenance is additive debug only. v1 classification is documented as current policy, not a permanent guarantee.

## Out-of-scope preserved

- No endpoint redesign.
- No PHAT/timer movement changes.
- No calibration work.
- No replay/simulation expansion.
- No source-specific hacks.
- No changes to engine/ingest/grid_reducer.py or backend/services/runner.py (provenance is in rails debug; runner already receives and can surface it if needed).

## Recommendation

**Promote.** Stage 1 is bounded, reviewable, testable, and reversible. Rail endpoint behavior was not redesigned.
