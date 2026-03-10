# Stage report — PHAT Coupling and Movement Realignment (Stage 4C)

## Stage scope (approved)

Stage 4C — raw-contract replay verification:
- Extend/adapt the bounded replay verification assessment so it processes a real raw-contract replay input path (not only point_passthrough).
- Use invariant_diagnostics=True so raw replay output includes contract_diagnostics.
- Strictly a raw replay verification stage; no compute, calibration, or policy changes.

## Deliverables completed

1. **Assessment script and raw replay support**
   - Added `tools/replay_verification_assess.py` (4B-style pre-load + inject; 4C: supports any path including raw BO3-shaped snapshots).
   - Default path for no-arg run: `tools/fixtures/raw_replay_sample.jsonl` so raw-contract verification can be run by default.

2. **Raw replay fixture**
   - Added `tools/fixtures/raw_replay_sample.jsonl`: three minimal BO3-shaped snapshots (team_one, team_two, round_phase, round_time_remaining). Each line is a single JSON object (generic JSONL). Loaded via load_generic_jsonl when BO3-wrapped entries are absent; first payload passes _is_raw_bo3_snapshot so runner uses raw path (normalize → bounds → rails → resolve with invariant_diagnostics).

3. **Raw replay assessment run**
   - Ran: `python tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`

## Assessment results (raw replay input)

| Metric | Value |
|--------|--------|
| Replay path | tools/fixtures/raw_replay_sample.jsonl |
| Replay path exists | true |
| direct_load_payload_count | 3 |
| replay_payload_count_loaded | 3 |
| total_points_captured | 3 |
| raw_contract_points | 3 |
| point_passthrough_points | 0 |
| points_with_contract_diagnostics | 3 |
| structural_violations_total | 0 |
| behavioral_violations_total | 0 |
| invariant_violations_total | 0 |
| p_hat_min | ~0.500 |
| p_hat_max | ~0.505 |
| rail_low_min | ~0.480 |
| rail_high_max | ~0.539 |

## Conclusion

- **Mismatch confirmed:** No.
- **Not confirmed:** Yes. Raw-contract path was exercised; contract_diagnostics were present on all three points; zero structural, behavioral, and invariant violations observed.
- **Still blocked:** No. Raw replay verification is unblocked; the assessment successfully processes real raw-contract replay input and reports diagnostics.

## Files changed

| Path | Change |
|------|--------|
| tools/replay_verification_assess.py | New: pre-load payloads, inject into runner, aggregate raw_contract/point_passthrough and contract_diagnostics/violations; default path for 4C = raw fixture. |
| tools/fixtures/raw_replay_sample.jsonl | New: 3-line generic JSONL of minimal BO3-shaped snapshots for raw-contract path. |
| automation/reports/initiative-20260306-phat-coupling-realignment-stage-4c.md | This report. |

## Forbidden changes (confirmed not done)

- engine/compute/resolve.py, midround_v2_cs2.py, q_intra_cs2.py, rails_cs2.py, bounds.py: not modified.
- Calibration, BUY_TIME/FREEZETIME/non-IN_PROGRESS semantics, broad runner refactors, replay policy redefinition: not modified.

## Recommended next step

Use the same assessment with larger or live BO3 replay files (if available) to gather more evidence; optional calibration/drift assessment when desired.
