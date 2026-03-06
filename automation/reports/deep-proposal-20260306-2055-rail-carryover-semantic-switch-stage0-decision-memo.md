branch: deep/proposal-20260306-2055-rail-carryover-semantic-switch
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve stage 1

STAGE 0 DECISION MEMO
- Semantic switch target:
  - Switch only the **contract endpoint derivation** in `engine/compute/rails_cs2.py` from Stage-1-locked v1 semantics to v2 carryover-conditioned semantics.
  - Current (v1-locked) behavior to replace: v2 observability is emitted, but endpoints remain v1 (`rail_input_contract_policy="observe_v2_use_v1_endpoints"`, `rail_input_active_endpoint_semantics="v1"`, `rail_input_v1_fallback_used=True`).
  - After switch (Stage 1 target): `rails_cf_low` / `rails_cf_high` must be conditioned on allowed carryover inputs, not score/prematch-only proxies.
  - Semantics that remain unchanged:
    - `rail_low <= rail_high`
    - rails clamped to series bounds and [0,1]
    - map-point boundary behavior (A map-point aligns to series high; B map-point aligns to series low)
    - contract min-width safety epsilon behavior
    - PHAT coupling and resolve movement semantics remain unchanged
  - No semantic change to heuristic/debug-only rails path beyond parity/observability needed for migration evidence.

- Allowed v2 inputs:
  - Required carryover-capable inputs (must be allowed to influence v2 endpoints):
    - `frame.cash_totals`
    - `frame.loadout_totals`
    - `frame.armor_totals`
    - `frame.scores`
    - `frame.series_score`
    - `frame.series_fmt`
    - `config.prematch_map`
  - Optional enrichments (may influence diagnostics and optional endpoint refinement, but must not block activation):
    - `frame.wealth_totals`
    - `frame.loadout_source`
    - `frame.loadout_ev_count_a`
    - `frame.loadout_ev_count_b`
    - `frame.loadout_est_count_a`
    - `frame.loadout_est_count_b`

- Forbidden inputs:
  - Must remain forbidden for v2 endpoint semantics:
    - `frame.hp_totals`
    - `frame.alive_counts`
    - `frame.round_time_remaining_s`
    - `frame.round_time_s`
    - `frame.bomb_phase_time_remaining`
  - Any transient intraround signals (positioning, utility state, site control, sub-round clock pressure) remain forbidden even if present in upstream payloads.
  - Forbidden-field perturbation with all allowed fields fixed must not change `rails_cf_low`/`rails_cf_high`.

- Required activation fields:
  - Stage 1 activates v2 semantics only when **all** required fields are present and valid:
    - `config.prematch_map`: numeric, finite, `1e-6 <= value <= 1 - 1e-6`
    - `frame.scores`: tuple/list len>=2; both entries castable to int and non-null
    - `frame.series_score`: tuple/list len>=2; both entries castable to int and non-null
    - `frame.series_fmt`: non-empty string
    - `frame.cash_totals`: tuple/list len>=2; both entries castable to float and non-null
    - `frame.loadout_totals`: tuple/list len>=2; both entries castable to float and non-null
    - `frame.armor_totals`: tuple/list len>=2; both entries castable to float and non-null
  - Activation is binary (all-required-valid or fallback); no probabilistic activation.

- Optional fields:
  - Optional fields are enrichments only:
    - `frame.wealth_totals`: tuple/list len>=2 numeric if used; invalid values ignored
    - `frame.loadout_source`: expected enum `"ev" | "weapon_est" | "mixed"`; invalid treated as unknown
    - `frame.loadout_ev_count_*`, `frame.loadout_est_count_*`: expected non-negative ints; invalid treated as unknown
  - Optional-field invalidity must not disable v2 when required fields are valid.
  - Optional usage must be observable in debug/provenance so before/after comparisons are deterministic.

- Fallback policy:
  - Stage 1 policy states must be explicit and finite (no hidden mixed mode):
    - `force_v1` (explicit lock to v1 semantics)
    - `v2_strict` (activate v2 only when all required fields are valid; otherwise deterministic fallback to v1)
  - `observe_v2_use_v1_endpoints` is pre-Stage-1 baseline only; it is not an implementation-end-state for Stage 1.
  - Partial v2 is **not allowed** in Stage 1.
  - Deterministic reason codes required when v2 is not used:
    - `POLICY_FORCE_V1`
    - `V2_REQUIRED_FIELDS_MISSING`
    - `V2_REQUIRED_FIELDS_INVALID`
    - `V2_POLICY_UNSUPPORTED`
  - Deterministic activation marker required when v2 is used:
    - `V2_STRICT_ACTIVATED`

- Acceptance criteria for Stage 1:
  - Pre-implementation gate:
    - Stage 0 memo approved with no open semantic ambiguities.
    - Baseline evidence pack captured on `agent-initiative-base` with reproducible artifacts.
  - Implementation-pass criteria:
    - Structural invariants remain green (`rail_low <= rail_high`, bounds containment, map-point alignment checks).
    - v2 activation occurs only under `v2_strict` + all-required-valid.
    - Forbidden/transient perturbation invariance holds.
    - Carryover sensitivity is demonstrated: with score/series fixed, changing required carryover vectors changes at least one contract endpoint in designated scenarios.
    - Cross-source parity evidence passes for BO3/GRID/replay raw when equivalent required fields are provided.
    - Replay summary/schema includes policy state, activation/fallback reason counts, and deterministic outputs.
  - Failure criteria:
    - Any structural regression
    - Any implicit mixed-mode path
    - Any activation without all required-valid fields
    - Any failed forbidden-input invariance check
    - Missing required migration evidence artifacts

- Baseline evidence pack:
  - Fixed fixtures/replays:
    - `tools/fixtures/raw_replay_sample.jsonl`
    - `tools/fixtures/replay_multimatch_small_v1.jsonl`
  - Required baseline test set (before migration and after migration, same commands):
    - `python3 -m pytest -q tests/unit/test_rails_cs2_basic.py tests/unit/test_rails_input_contract.py`
    - `python3 -m pytest -q tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py`
  - Required replay artifacts (before and after):
    - `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
    - `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`
  - Required evidence dimensions:
    - structural invariant checks
    - carryover sensitivity checks (fixed-score controlled scenarios)
    - transient invariance checks (forbidden perturbation scenarios)
    - cross-source parity checks (BO3 vs GRID vs replay raw contract diagnostics)
    - replay artifact/schema checks (deterministic summary + required keys + policy/reason counts)

- Narrowest safe Stage 1:
  - Must include only:
    - Endpoint semantic switch implementation in `engine/compute/rails_cs2.py` (v1->v2_strict logic and explicit policy state handling)
    - Minimal test updates/additions in:
      - `tests/unit/test_rails_input_contract.py`
      - `tests/unit/test_rails_cs2_basic.py`
      - `tests/unit/test_runner_source_contract_parity.py`
      - `tests/unit/test_replay_verification_assess_stage1.py`
    - Minimal replay assessment/schema key additions needed to expose activation/fallback reason counts
  - Must NOT include:
    - replay/simulation architecture redesign
    - calibration tuning campaigns
    - PHAT movement/resolve semantics changes
    - broad runner refactor beyond strict diagnostics plumbing required for reason-code visibility
    - map_fair or broader compute architecture rewrite

- Risks:
  - Endpoint behavior shift may move PHAT trajectories; strict before/after evidence is required to separate expected semantic shift from regressions.
  - Source-data completeness for carryover fields may increase fallback frequency; reason-code observability is required to avoid silent degradation.
  - Cross-source normalization gaps can masquerade as semantic bugs; parity tests are mandatory.
  - Over-expansion risk (replay/calibration work) is controlled by Stage 1 boundary lock.

- Recommendation:
  - Approve Stage 1 implementation **only** with this Stage 0 gate frozen, explicit `force_v1`/`v2_strict` policy states, no partial-v2 mode, deterministic reason codes, and the fixed evidence pack executed before and after migration.
