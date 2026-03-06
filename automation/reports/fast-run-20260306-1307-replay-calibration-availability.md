# Promotion report

branch: fast/run-20260306-1307-replay-calibration-availability
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing calibration availability diagnostics in replay validation summary
- **Title:** `tools/replay_verification_assess.py` did not expose Chapter 9.5 calibration-metric status fields (available/unavailable + reason), blocking unambiguous calibration diagnosis from replay artifacts

## Why it outranked alternatives

Issue ladder was applied on current `agent-base` evidence:

1. Structural invariant violations: none (`structural_violations_total=0`).
2. Failing canonical tests: none (`375 passed`).
3. Confirmed replay mismatches: none (`raw_contract_points>0`, `unknown_replay_mode_points=0`, no mismatch signal).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With ranks 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): replay summaries silently omitted calibration-status fields, so consumers could not distinguish "not computed" from "not implemented."

## Baseline evidence

- Baseline branch: `fast/run-20260306-1307-replay-calibration-availability` (branched from `agent-base`)
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.19s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Calibration-key probe:
    - `python3 - <<'PY' ... run_assessment(...) ... PY`
    - result: `{'calibration_metrics_available': False, 'calibration_metrics_unavailable_reason': False, 'calibration_brier_score': False, 'calibration_log_loss': False, 'calibration_reliability_bins': False}` (all absent)

## Files changed

| Path | Change |
|------|--------|
| `tools/replay_verification_assess.py` | Added explicit calibration diagnostics fields to replay summary: availability flag, unavailable reason, and placeholder metric outputs (`brier`, `log_loss`, `reliability_bins`). |
| `tools/schemas/replay_validation_summary.schema.json` | Added new required schema properties/types for calibration diagnostics fields. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertions for deterministic calibration diagnostics values on current fixture (unavailable + null/empty metrics). |
| `automation/reports/fast-run-20260306-1307-replay-calibration-availability.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1307-replay-calibration-availability.json` | Added machine-readable run report. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` -> `1 passed`
- `python3 - <<'PY' ... run_assessment('tools/fixtures/replay_multimatch_small_v1.jsonl') ... PY` -> calibration fields present with expected values:
  - `calibration_metrics_available=False`
  - `calibration_metrics_unavailable_reason='series_outcome_labels_missing_in_replay_points'`
  - `calibration_brier_score is None`
  - `calibration_log_loss is None`
  - `calibration_reliability_bins == []`
- `python3 tools/replay_verification_assess.py` -> diagnostics remain clean (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`) and now include calibration status fields
- `python3 -m pytest -q` -> `375 passed in 7.23s`

## Before/after evidence

- **Before:** replay summary did not contain any calibration status/metric keys, making calibration observability ambiguous.
- **After:** replay summary always emits explicit calibration diagnostics status fields; current fixtures clearly report metrics as unavailable with deterministic reason, while preserving all existing invariant and diagnostics outputs.

## Unresolved risks

- Current replay fixtures still do not provide explicit series-outcome labels, so numeric calibration metrics remain unavailable (`null`/empty) by design.
- This run adds observability only; it does not introduce calibration scoring logic on labeled datasets.

## Stop reason

Stopped after resolving the selected rank-5 diagnostics gap with minimal instrumentation-only edits and repeated validation; additional work would require a broader labeled-replay pipeline beyond this bounded maintenance scope.

## Recommendation

- `promote`
