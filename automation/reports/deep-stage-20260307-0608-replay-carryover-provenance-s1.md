# Stage report — Replay carryover provenance (Stage 1)

branch: deep/stage-20260307-0608-replay-carryover-provenance-s1
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: promote

## Objective (locked)

Plumb source/replay_kind end-to-end into rail debug and replay evidence, and emit deterministic completeness diagnostics for required carryover inputs per source, without changing rail formulas, PHAT semantics, or replay policy.

## Files changed

| Path | Change |
|------|--------|
| `engine/compute/rails.py` | Added optional `source` and `replay_kind` kwargs to `compute_rails()`; forward to `compute_rails_cs2()`. |
| `backend/services/runner.py` | BO3 path: pass `source=getattr(config, "source", None)`, `replay_kind=None` into `compute_rails`. GRID path: pass `source="GRID"`, `replay_kind=None`. Replay raw path: pass `source="REPLAY"` (or config.source), `replay_kind=_replay_format` or `"raw"`. |
| `tools/replay_verification_assess.py` | Import `RAIL_INPUT_V2_REQUIRED_FIELDS`. Aggregate per-point `rail_input_source` / `rail_input_replay_kind` into `carryover_evidence_by_source_class` (points, v2_activated, v1_fallback, reason_code_counts, required_complete/incomplete). Emit `carryover_completeness_required_fields` in summary. |
| `tools/schemas/replay_validation_summary.schema.json` | Added optional properties `carryover_evidence_by_source_class` and `carryover_completeness_required_fields`. |
| `tests/unit/test_rails_input_contract.py` | Added `test_rail_input_source_replay_kind_in_provenance`, `test_compute_rails_forwards_source_replay_kind`. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Fixed indentation in `test_point_like_replay_...`. Added `test_carryover_evidence_by_source_class_present_and_structured`, `test_carryover_complete_fixture_source_class_shows_activation`. |

## Behavior changed

1. **Rail debug/provenance**  
   `rail_input_source` and `rail_input_replay_kind` are now set end-to-end when the runner calls `compute_rails` with `source`/`replay_kind`. Replay raw path uses `REPLAY` + `raw`; GRID uses `GRID` + `None`; BO3 uses config.source + `None`. Existing v2 provenance (present/missing/invalid required fields, required_complete, reason codes) is unchanged and now attributable by source class.

2. **Completeness diagnostics**  
   Required carryover fields (cash_totals, loadout_totals, armor_totals, scores, series_score, series_fmt, prematch_map) remain defined in `rails_cs2.RAIL_INPUT_V2_REQUIRED_FIELDS`. Per-point completeness is already emitted in rail debug (`rail_input_v2_required_complete`, `rail_input_v2_present_required_fields`, etc.). No new engine-level completeness fields were added; assessment aggregates them by source class.

3. **Replay assessment output**  
   Summary now includes:
   - `carryover_evidence_by_source_class`: for each `{source}_{replay_kind}` key, counts of points, v2_activated_points, v1_fallback_points, reason_code_counts, required_complete_points, required_incomplete_points.
   - `carryover_completeness_required_fields`: list of the seven required field names for reference.
   Sparse raw fixtures show `REPLAY_raw` with fallback only and required_incomplete; carryover-complete fixture with prematch_map shows `REPLAY_raw` with v2 activated and required_complete.

4. **Activation/fallback semantics**  
   Unchanged. No changes to v2 activation logic, fallback policy, or rail formulas.

## Before/after evidence

- **raw_replay_sample.jsonl**: Before: no source/replay_kind in rail debug; no by-source-class breakdown. After: `carryover_evidence_by_source_class.REPLAY_raw` has points=3, v1_fallback_points=3, v2_activated_points=0, required_incomplete_points=3, reason_code_counts `V2_REQUIRED_FIELDS_MISSING`: 3.
- **replay_multimatch_small_v1.jsonl**: Same pattern; REPLAY_raw with 6 points, all fallback, required incomplete.
- **replay_carryover_complete_v1.jsonl --prematch-map 0.55**: After: REPLAY_raw has points=3, v2_activated_points=3, v1_fallback_points=0, required_complete_points=3, reason_code_counts `V2_STRICT_ACTIVATED`: 3.

## Acceptance criteria status

| Criterion | Status |
|-----------|--------|
| Rail debug/provenance includes meaningful source/replay_kind where available | Met: runner passes source/replay_kind; rails_cs2 already emitted them when provided; now wired end-to-end. |
| Required carryover completeness visible deterministically in replay evidence | Met: rail debug has rail_input_v2_* completeness fields; assessment aggregates by source class. |
| Replay assessment can distinguish sparse fallback from carryover-complete by source/class | Met: carryover_evidence_by_source_class provides activation/fallback/completeness per source class. |
| Existing activation/fallback behavior unchanged | Met: no changes to rails_cs2 activation or fallback logic. |
| Existing sparse raw fixtures still fallback deterministically | Met: tests and tool runs confirm. |
| Existing carryover-complete fixture still activates v2 with valid prematch_map | Met: tests and tool runs confirm. |
| Full canonical unit suite remains green for scope | Met: requested tests pass; 5 pre-existing failures in test_runner_map_identity (asyncio). |

## Validation

- test_replay_verification_assess_stage1.py: 6 passed (includes 2 new carryover evidence tests)
- test_runner_source_contract_parity.py: 1 passed
- test_rails_input_contract.py: 10 passed (includes 2 new source/replay_kind tests)
- tools/replay_verification_assess.py on raw_replay_sample, replay_multimatch_small_v1, replay_carryover_complete_v1 --prematch-map 0.55: all succeeded with expected output
- pytest -q: 410 passed, 5 failed (test_runner_map_identity only; pre-existing)

## Risks

- None identified. Changes are additive (provenance plumbing and assessment aggregation); no rail math or policy changed.

## Out-of-scope preserved

- No rail formula changes. No PHAT/movement/timer changes. No replay policy redesign. No runner architecture redesign beyond minimal provenance plumbing. No GRID redesign. No corpus uplift. No calibration. No replay+simulation architecture work.
