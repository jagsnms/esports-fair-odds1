# Promotion report

branch: fast/run-20260306-1407-contract-bombtimer-diag
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Missing bomb timer field in contract diagnostics/replay completeness contract
- **Title:** `contract_diagnostics` omitted `bomb_time_remaining_s`, blocking full post-plant timer-state diagnosis

## Why it outranked alternatives

Current evidence on `agent-base` showed no unresolved rank 1-4 issues:

1. Structural invariant violations: none (`structural_violations_total=0` on replay assessment).
2. Failing canonical tests: none (`375 passed`).
3. Confirmed replay mismatches: none (`raw_contract_points` only, `unknown_replay_mode_points=0`).
4. High-frequency diagnostic invariant failures: none (`behavioral_violations_total=0`, `invariant_violations_total=0`).

With rank 1-4 clear, the highest unresolved bounded issue was rank 5 (missing instrumentation blocking diagnosis): diagnostics included round timer and planted state, but not bomb timer remaining required for full timer-state diagnosis in post-plant scenarios.

## Baseline evidence

- Baseline branch: `fast/run-20260306-1407-contract-bombtimer-diag` (created from `agent-base`)
- Baseline commit: `6458ba6bdc5c9df6ed975cc29bdb2befca1dabd4`
- Baseline checks:
  - `python3 -m pytest -q` -> `375 passed in 7.28s`
  - `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`
  - Contract-diagnostics bomb-timer probe:
    - `python3 - <<'PY' ... resolve_p_hat(...) ... PY`
    - result: `has_bomb_time_remaining_s False`
  - Replay completeness contract baseline:
    - `contract_diagnostics_required_keys` did not contain `bomb_time_remaining_s`

## Files changed

| Path | Change |
|------|--------|
| `engine/diagnostics/invariants.py` | Added `bomb_time_remaining_s` to contract diagnostics input/output payload. |
| `engine/compute/resolve.py` | Extracted bomb timer remaining from `Frame.bomb_phase_time_remaining` and passed it into contract diagnostics. |
| `engine/normalize/bo3_normalize.py` | Normalized bomb timer source fields (`bomb_time_remaining`/`bomb_timer_remaining`/nested fallback) into canonical seconds and stored in `bomb_phase_time_remaining`. |
| `tools/replay_verification_assess.py` | Added `bomb_time_remaining_s` to required diagnostics completeness keys. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Updated contract diagnostics unit tests for new bomb timer field and assertions. |
| `tests/unit/test_runner_source_contract_parity.py` | Added parity assertion that diagnostics include `bomb_time_remaining_s` key. |
| `tests/unit/test_time_norm.py` | Added normalization assertion that BO3 snapshots propagate normalized bomb timer into frame bomb-phase state. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Added assertion that replay completeness contract requires `bomb_time_remaining_s`. |
| `automation/reports/fast-run-20260306-1407-contract-bombtimer-diag.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1407-contract-bombtimer-diag.json` | Added machine-readable run report artifact. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_invariants_contract_diagnostics.py tests/unit/test_runner_source_contract_parity.py tests/unit/test_replay_verification_assess_stage1.py tests/unit/test_time_norm.py` -> `19 passed`
- `python3 tools/replay_verification_assess.py` -> includes `bomb_time_remaining_s` in required keys with presence rate `1.0` and missing count `0`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl` -> includes `bomb_time_remaining_s` with presence rate `1.0` and missing count `0`
- `python3 - <<'PY' ... resolve_p_hat(...) ... PY` -> `has_bomb_time_remaining_s True`, value propagated (`32.5`)
- `python3 -m pytest -q` -> `375 passed in 7.21s`

## Before/after evidence

- **Before:** `contract_diagnostics` omitted `bomb_time_remaining_s` even when bomb timer data existed in `Frame.bomb_phase_time_remaining`; replay completeness contract could not enforce this key.
- **After:** `contract_diagnostics` now emits `bomb_time_remaining_s`, normalization captures bomb timer from raw snapshot fields, and replay completeness contract enforces/observes full presence (`rate=1.0`, missing `0`) for the new key.

## Unresolved risks

- Current canonical replay fixtures are mostly non-planted and do not stress a variety of live bomb-timer source formats; broader fixture diversity may reveal additional source-key variants.
- This run adds field-level instrumentation and completeness enforcement, not semantic calibration of timer pressure behavior.

## Stop reason

Stopped after the selected rank-5 instrumentation gap was fixed with minimal bounded edits and repeated validation; further timer-pressure calibration work would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
