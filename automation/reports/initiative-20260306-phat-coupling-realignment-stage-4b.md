# Stage report — PHAT Coupling and Movement Realignment (Stage 4B)

## Stage scope (approved)

Stage 4B — Replay verification unblock (no compute, calibration, or PHAT semantic changes):
- Diagnose why the Stage 4 assessment script did not cause the runner to load replay payloads.
- Implement the smallest bounded fix so the assessment can process real replay input and capture points.
- Re-run the bounded assessment and report conclusion.

## Root cause of 0-payload issue

The assessment script called the runner’s `_tick_replay(config)` with a config that had `replay_path` set to an absolute path. The runner’s lazy-load block (inside `_tick_replay`) uses `path = getattr(config, "replay_path", None) or "logs/bo3_pulls.jsonl"` and then calls `load_bo3_jsonl_entries(path)` and `load_generic_jsonl(path)`. In the script context, that lazy-load consistently resulted in 0 payloads (`replay_payload_count_loaded == 0`) even when the same path was readable by the script and `load_generic_jsonl(replay_path_str)` returned 12k+ lines. So the runner’s internal path resolution or file open in that context did not see the file (e.g. path/cwd or process context difference). The script had no way to change runner internals, so the fix was to **avoid relying on the runner’s lazy-load**: the script now loads payloads itself and injects them into the runner before the first tick.

## Fix implemented

1. **Pre-load payloads in the script** using the same logic as the runner: `load_bo3_jsonl_entries(path)` then `iter_payloads(..., None)`; if empty, `load_generic_jsonl(path)`.
2. **Inject into the runner** before the first `_tick_replay`: set `runner._replay_payloads`, `runner._replay_index`, `runner._replay_path`, `runner._replay_match_id`, and `runner._replay_format` (from first payload: raw vs point). The runner then uses these and does not re-enter the lazy-load block (because `path == self._replay_path`).
3. **Async broadcaster** in the script was updated so `await self._broadcaster.broadcast(...)` is valid (no-op async method).

No changes to `backend/services/runner.py` or any compute/calibration code.

## Replay inputs used

| Input | Path exists | Payloads loaded | Points captured (cap 500) |
|-------|-------------|------------------|---------------------------|
| `logs/bo3_pulls.jsonl` | no | 0 | 0 |
| `logs/history_points.jsonl` | yes | 12,232 | 500 |

## Metrics gathered (logs/history_points.jsonl)

| Metric | Value |
|--------|--------|
| replay_payload_count_loaded | 12,232 |
| total_points_captured | 500 (capped) |
| raw_contract_points | 0 |
| point_passthrough_points | 500 |
| points_with_contract_diagnostics | 0 |
| structural_violations_total | 0 |
| behavioral_violations_total | 0 |
| invariant_violations_total | 0 |
| p_hat_min / p_hat_max | 0.55 / 0.89 |
| rail_low_min / rail_high_max | 0.35 / 0.92 |

## Contract diagnostics and violations

- **Contract diagnostics:** Not present for this run. The replay file used was wire-format (point-like) so the runner correctly used point-passthrough; the full resolve pipeline (and thus `contract_diagnostics`) is only used for raw replay.
- **Violations observed:** None (structural, behavioral, invariant all 0).

## Conclusion

**Replay mismatch not confirmed.**

- The assessment successfully processed real replay input and captured 500 points (point_passthrough).
- No violations were observed. PHAT and rail metrics are in a plausible range.
- Verification is no longer blocked for point replay. Raw replay (with `invariant_diagnostics` and contract_diagnostics) would require a BO3-format replay file; the same script and inject approach would work once such a file is available.

## Recommended next step

- Use the same script with a BO3-format replay file (or fixture) to run raw replay and collect contract_diagnostics and violation counts if desired.
- Otherwise proceed with the next initiative (e.g. calibration/drift or further verification) as planned.

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added `_load_replay_payloads()`; pre-load and inject payloads into runner; set `_replay_format` from first payload; async no-op broadcaster. |
| `automation/reports/initiative-20260306-phat-coupling-realignment-stage-4b.md` | This report. |

## Compute / calibration semantics

No changes to resolve.py, midround_v2_cs2.py, q_intra_cs2.py, rails_cs2.py, bounds.py, BUY_TIME/FREEZETIME behavior, or calibration.
