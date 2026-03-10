branch: deep/stage-20260306-2203-rail-v2-activation-s1
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: promote

# Stage report — Rail V2 Activation Backbone (Stage 1)

## Objective (locked)

Deliver the narrowest replay/source backbone needed to demonstrate non-zero rail-v2 activation on one bounded carryover-complete raw replay fixture class, while preserving deterministic fallback behavior on existing sparse fixture classes.

## Files changed

| Path | Change |
|---|---|
| `tools/fixtures/replay_carryover_complete_v1.jsonl` | Added one bounded carryover-complete raw replay fixture class (3 raw BO3-shaped snapshots with explicit player-state carryover values). |
| `tools/replay_verification_assess.py` | Added minimal optional `prematch_map` plumbing for assessment runs; exposed `assessment_prematch_map`; added CLI support `--prematch-map` without changing replay policy or compute semantics. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added focused tests for sparse fallback-only classes, carryover-complete activation class, and point-like replay rejection/exclusion behavior. |
| `automation/reports/deep-stage-20260306-2203-rail-v2-activation-s1.md` | This implementation report. |

## Behavior changed

1. **One new bounded activation fixture class introduced**
   - New raw fixture class: `replay_carryover_complete_v1`.
   - Encodes explicit same-snapshot `player_states` carryover inputs (balance/equipment_value/armor), enabling `cash_totals/loadout_totals/armor_totals` to be valid after normalization.

2. **Replay assessment can now run with explicit valid prematch map**
   - `run_assessment(..., prematch_map=<float>)` and CLI `--prematch-map`.
   - This is assessment-only plumbing to satisfy frozen activation contract; no replay policy changes and no rail/PHAT formula changes.

3. **Focused validation contract expanded**
   - Sparse classes remain fallback-only with deterministic missing-field reason.
   - Carryover-complete class must show v2 activation with valid prematch_map.
   - Point-like replay remains rejected/excluded from activation denominator.

## Before/after evidence

### Baseline (before Stage 1 edits; current base behavior)

- `tools/fixtures/raw_replay_sample.jsonl`
  - `total_points_captured=3`
  - `rail_input_v2_activated_points=0`
  - `rail_input_v1_fallback_points=3`
  - `rail_input_reason_code_counts={"V2_REQUIRED_FIELDS_MISSING":3}`
  - `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
- `tools/fixtures/replay_multimatch_small_v1.jsonl`
  - `total_points_captured=6`
  - `rail_input_v2_activated_points=0`
  - `rail_input_v1_fallback_points=6`
  - `rail_input_reason_code_counts={"V2_REQUIRED_FIELDS_MISSING":6}`
  - `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
- `logs/history_points.jsonl` (point-like class)
  - `total_points_captured=0`
  - `point_like_inputs_seen>0`, `point_like_inputs_rejected=point_like_inputs_seen`
  - excluded from activation denominator

### Stage 1 after

- Sparse fallback classes preserved:
  - `raw_replay_sample`: unchanged fallback profile (`v2_activated=0`, reason missing, invariants clean).
  - `replay_multimatch_small_v1`: unchanged fallback profile (`v2_activated=0`, reason missing, invariants clean).
- Carryover-complete class:
  - `replay_carryover_complete_v1` with no prematch map (`assessment_prematch_map=null`):
    - `total_points_captured=3`
    - `rail_input_v2_activated_points=0`
    - `rail_input_v1_fallback_points=3`
    - `rail_input_reason_code_counts={"V2_REQUIRED_FIELDS_MISSING":3}`
  - `replay_carryover_complete_v1` with `--prematch-map 0.55`:
    - `total_points_captured=3`
    - `rail_input_v2_activated_points=3`
    - `rail_input_v1_fallback_points=0`
    - `rail_input_reason_code_counts={"V2_STRICT_ACTIVATED":3}`
    - `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
- Point-like class remains rejected/excluded:
  - `history_points`: `total_points_captured=0`, rejected counts tracked, no activation/fallback denominator impact.

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| Add one bounded carryover-complete raw replay fixture class | ✅ met | `tools/fixtures/replay_carryover_complete_v1.jsonl` (3 lines only) |
| Keep existing sparse classes fallback-only | ✅ met | unchanged reason code `V2_REQUIRED_FIELDS_MISSING` on both sparse classes |
| Demonstrate non-zero v2 activation on carryover-complete class | ✅ met | `replay_carryover_complete_v1` with `prematch_map=0.55`: `v2_activated=3/3` |
| Point-like replay rejected/quarantined and excluded from activation denominator | ✅ met | `history_points`: `total_points_captured=0`, rejected counts > 0 |
| No structural/invariant regressions | ✅ met | all fixture-class evidence shows zero structural/behavioral/invariant totals |
| Canonical tests remain green | ✅ met | `python3 -m pytest -q` -> `394 passed` |
| Stay inside frozen Stage 0 gate (no rail/PHAT/replay architecture redesign) | ✅ met | only fixture + assessment parameter plumbing + focused tests changed |

Partially met criteria: **none**.

## Validation

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `3 passed`
- `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_carryover_complete_v1.jsonl --prematch-map 0.55`
- `python3 - <<'PY' ... run_assessment(...) across sparse/complete/point-like classes ... PY` (class-specific evidence pack)
- `python3 -m pytest -q` -> `394 passed`

## Risks / scope pressure encountered

- `logs/history_points.jsonl` rejection counters are > line count due loader path behavior; this does not affect gate semantics (still rejected and excluded), but warrants future bounded observability clarification if needed.
- Pressure to broaden fixture corpus was explicitly avoided; only one new activation fixture class was added.

## Frozen-gate compliance statement

Stage 1 stayed inside the frozen Stage 0 gate:

- no rail formula changes,
- no PHAT/movement/timer changes,
- no replay policy redesign,
- no runner architecture redesign,
- no GRID redesign,
- no replay+simulation architecture reopening,
- no fixture sprawl beyond one new carryover-complete raw fixture class.

## Recommendation

**promote**
