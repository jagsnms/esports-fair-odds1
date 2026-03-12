# Current Status - `master`

Last updated: 2026-03-12

## Snapshot
- **Promoted `master` initiative:** Backend BO3 corpus continuity protection against git/worktree hazards is the current promoted `master` state.
- **Active corpus continuity contract:** the default active BO3 corpus path now lives outside ordinary repo worktree hazard at `%LOCALAPPDATA%\EsportsFairOdds\corpus\bo3_backend_live_capture_contract.jsonl` (or `BO3_BACKEND_CAPTURE_PATH` when deliberately overridden). The old in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path is no longer the trusted continuity-protected active store.
- **Frozen artifact boundary remains separate:** repo-visible frozen cuts still live under `automation/reports/`, including `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl` and the point-in-time readiness report artifact. Those are review/analysis surfaces, not the continuity-protected active corpus.
- **Current readiness-tool boundary remains intact:** `tools/run_backend_bo3_corpus_readiness_analyzer.py` still exists as a separate corpus-level analyzer, and `tools/run_backend_bo3_live_parity_diagnostic.py` still exists as the separate bounded one-match diagnostic reading the frozen snapshot path.

## Main red flags
1. **This continuity correction is not calibration implementation.** It only protects the active corpus from silent rollback hazards.
2. **This continuity correction is not live parity implementation.** It does not change parity math, replay/live linkage, or decision logic.
3. **The bounded one-match diagnostic remains separate on purpose.** This stage does not turn it into a corpus tool.
4. **Repo-visible artifacts are still secondary to active corpus continuity.** Snapshots and reports may stay in the repo, but they must not be confused with the active growing store.

## Most recent completed checks
- `tests/unit/test_backend_bo3_capture_contract.py` is the focused continuity-contract test surface for the backend writer path and confirms the default active corpus path resolves outside the repo worktree.
- `tests/unit/test_run_backend_bo3_corpus_readiness_analyzer.py` confirms the corpus analyzer follows the same continuity-protected active corpus default while preserving separation from the bounded diagnostic.
- `tests/unit/test_run_backend_bo3_live_parity_diagnostic.py` continues to confirm that the bounded diagnostic remains a one-match tool on the frozen snapshot path.

## Current initiative status
- **Actual current runtime BO3 ingestion path:** `backend/services/runner.py`.
- **Promoted `master` continuity contract:** active corpus now defaults outside ordinary repo worktree hazard; frozen snapshots remain repo-visible and separate; the old in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path is no longer the continuity-protected active store.
- **Truth boundary:** `master` now carries one narrow continuity protection only. It does not redesign broader storage, calibration, parity, replay/live logic, or reporting platforms.

## Next likely step
- Use the new external-path default consistently and avoid pointing `BO3_BACKEND_CAPTURE_PATH` back into the repo unless you deliberately want to accept the old worktree hazard.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit only when a branch actually has newer work than promoted `master`.
