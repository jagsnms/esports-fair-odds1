# Promotion report

## Selected issue

- **Issue:** failing canonical tests (`tests/unit/test_bo3_multisession_isolation.py`, `tests/unit/test_multisession_state_isolation.py`)
- **Title:** BO3 multi-session isolation regression from strict `get_snapshot` call signature

## Why it outranked alternatives

This run selected a **failing canonical tests** issue (rank #2 on the required ladder). No structural invariant failure evidence was observed, and canonical pytest had reproducible failures in the BO3 multi-session path, so this was the highest-ranked unresolved issue.

## Baseline evidence

- Branch at baseline: `agent-base`
- Baseline test command: `python3 -m pytest -q`
- Baseline result: `2 failed, 339 passed`
- Failing tests:
  - `tests/unit/test_bo3_multisession_isolation.py::test_bo3_multisession_isolation`
  - `tests/unit/test_multisession_state_isolation.py::test_multisession_state_isolation`
- Observed failure pattern:
  - session BO3 buffers stayed empty (`bo3_buf_raw is None`)
  - non-primary BO3 ticks called `store.get_state()` unexpectedly

## Files changed

| Path | Change |
|------|--------|
| `backend/services/runner.py` | Added a narrow compatibility fallback in `_bo3_fetch_into_buffer` to retry `get_snapshot(mid)` when debug kwargs are unsupported by the client/mock. |
| `automation/reports/report-20260306-bo3-multisession-isolation.md` | Added this run report. |

## Validation performed

- `python3 -m pytest -q tests/unit/test_bo3_multisession_isolation.py tests/unit/test_multisession_state_isolation.py` → `2 passed`
- `python3 -m pytest -q` → `341 passed`
- Repeated stability check:
  - `python3 -m pytest -q tests/unit/test_bo3_multisession_isolation.py tests/unit/test_multisession_state_isolation.py` → `2 passed`

## Before/after evidence

- Before:
  - Full canonical suite: `2 failed, 339 passed`
  - BO3 isolation tests failed due to empty per-session buffer and non-primary state-store access.
- After:
  - Targeted BO3 isolation tests: `2 passed` (twice)
  - Full canonical suite: `341 passed`
  - Selected issue no longer reproducible.

## Unresolved risks

- The fallback is keyed to `TypeError` messages containing unsupported debug kwarg names. If a future client raises different `TypeError` text for signature mismatch, this path may not trigger.
- No replay-specific mismatch evidence was part of this issue scope; replay checks remain for separate ranked runs.

## Stop reason

Stopped after the selected highest-ranked issue was fixed and validated with targeted tests plus full canonical suite; additional edits would expand beyond the bounded single-issue scope.

## Recommendation

- [x] **Promote** — ready for human review/merge
- [ ] **Hold** — keep branch for later review
- [ ] **Discard** — do not promote
