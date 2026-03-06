# Stage report — PHAT Coupling and Movement Realignment (Stage 4)

## Stage scope (approved)

Stage 4 — Replay/Verification Assessment (assessment/reporting only; no compute or calibration changes):
- Run bounded replay/verification assessment on current PHAT behavior after Stages 1–3.
- Use invariant_diagnostics=True where appropriate.
- Collect evidence on structural/behavioral violation rates, PHAT/rail behavior, replay mode separation.
- Produce a concrete assessment report with conclusion and recommended next step.

## Replay inputs used

| Input | Result |
|-------|--------|
| `logs/bo3_pulls.jsonl` | Path did not exist; no payloads. |
| `logs/history_points.jsonl` | Path exists; file readable (direct load: 12k+ JSONL lines). Runner lazy-load in script context reported 0 payloads loaded. |

Default script path: `logs/bo3_pulls.jsonl`. Secondary run: `logs/history_points.jsonl` (wire-format point-like lines).

## Diagnostics enabled/disabled

- **invariant_diagnostics:** Enabled (`True`) in the config passed to the runner for replay.
- **contract_testing_mode:** Wired from invariant_diagnostics at resolve call sites (Stage 3A); replay raw path would emit contract_diagnostics when payloads are processed through the full pipeline.

## Metrics gathered

From `tools/replay_verification_assess.py` (run from project root):

| Metric | Value (logs/bo3_pulls.jsonl) | Value (logs/history_points.jsonl) |
|--------|-------------------------------|------------------------------------|
| replay_path_exists | false | true |
| direct_load_payload_count | 0 | 12,000+ |
| replay_payload_count_loaded | 0 | 0 |
| total_points_captured | 0 | 0 |
| raw_contract_points | 0 | 0 |
| point_passthrough_points | 0 | 0 |
| points_with_contract_diagnostics | 0 | 0 |
| structural_violations_total | 0 | 0 |
| behavioral_violations_total | 0 | 0 |
| invariant_violations_total | 0 | 0 |
| p_hat_min / p_hat_max / rail stats | — | — |

No points were appended by the runner in the script run; therefore no violation counts or PHAT/rail distributions could be computed from live replay output.

## Violation counts observed

None (no points processed).

## Baseline comparison

Not performed. No prior baseline run or stored metrics were available; comparison would require a separate baseline capture (e.g. pre–Stage 2 replay run or golden snapshot).

## Conclusion

**Verification incomplete / blocked.**

- No confirmed replay mismatch (no replay output was produced in the assessment run).
- No confirmed absence of mismatch (raw pipeline and contract_diagnostics were not exercised in this run).
- Cause: In the standalone script context, the runner’s lazy-load did not load payloads from the given replay path (replay_payload_count_loaded remained 0 even when the file existed and was readable via direct load). Therefore the assessment could not gather metrics from actual replay output.

## Recommended next step

1. **Run replay verification in an environment where the runner successfully loads replay data** (e.g. via the API with source=REPLAY, invariant_diagnostics enabled, and a known replay path), then re-run the assessment logic on the resulting history/points, or
2. **Add a small BO3-format or wire-format replay fixture** that the runner loads in the same process (e.g. in a unit test or a script that injects payloads), run the assessment, and record structural/behavioral violation counts and PHAT/rail metrics, or
3. **Leave Stage 4 as “assessment attempted; incomplete”** and proceed to the next initiative (e.g. calibration/drift assessment or further verification) with the understanding that replay verification evidence is still pending.

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | New: minimal script to run replay with invariant_diagnostics=True, capture append_point, aggregate metrics (violations, PHAT/rails, replay_mode). |
| `automation/reports/initiative-20260306-phat-coupling-realignment-stage-4.md` | This report. |

## Validation

- `python tools/replay_verification_assess.py logs/bo3_pulls.jsonl` → JSON output; 0 points (no file).
- `python tools/replay_verification_assess.py logs/history_points.jsonl` → JSON output; direct_load 12k+; replay_payload_count_loaded 0; 0 points captured.
- No changes to engine/compute, calibration, or runner semantics.

## Compute / calibration semantics

No changes to resolve.py, midround_v2_cs2.py, q_intra_cs2.py, rails_cs2.py, bounds.py, BUY_TIME/FREEZETIME behavior, or calibration.
