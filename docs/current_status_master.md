# Current Status - `master`

Last updated: 2026-03-12

## Snapshot
- **Promoted `master` BO3 lane state:** the active BO3 corpus defaults outside ordinary repo worktree hazard, one-time alignment and divergent recovery tools exist as special repair workflows, the bounded one-match diagnostic remains separate, the corpus-readiness analyzer remains separate, persisted BO3 `round_result` history rows now carry top-level `match_id`, and a narrow BO3 live labeled calibration exporter exists for round-level `q_intra_total` vs `round_result` only.
- **Active corpus continuity contract:** the default active BO3 corpus path lives at `%LOCALAPPDATA%\EsportsFairOdds\corpus\bo3_backend_live_capture_contract.jsonl` (or `BO3_BACKEND_CAPTURE_PATH` when deliberately overridden). The old in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path is legacy only and no longer the trusted continuity-protected active store.
- **One-time repair tools remain special workflows:** `tools/align_backend_bo3_active_corpus.py` is only for prefix/superset continuity alignment, and `tools/recover_backend_bo3_divergent_corpus.py` is only for one-off divergent corpus recovery with refusal on unresolved conflicts. Steady-state writer/analyzer flow does not depend on rerunning them.
- **Frozen artifact boundary remains separate:** repo-visible frozen cuts still live under `automation/reports/`, including the BO3 snapshot artifact and point-in-time analyzer/export reports. Those remain review/analysis surfaces, not the active growing corpus.
- **Current branch-local BO3 ingestion audit instrumentation note (not promoted `master` truth):** this branch adds narrow per-session fetch -> suppression -> propagation visibility in `backend/services/runner.py` and exposes it through `/debug/telemetry/sessions` as `bo3_pipeline`. It is instrumentation only and does not claim a poller or ingestion fix.

## Main red flags
1. **This branch-local BO3 pipeline instrumentation is not a poller or ingestion fix.** It only exposes where BO3 updates are fetched, suppressed, accepted, and propagated.
2. **This branch-local instrumentation is not exporter or calibration work.** It does not change BO3 evidence labeling, calibration math, or downstream judgment.
3. **The promoted exporter remains narrow.** It is still round-level `q_intra_total` vs `round_result` only and not a calibration-quality verdict.
4. **Point-in-time live observations remain local only.** Fresh BO3 lag/suppression observations still depend on the current live match and current local runner session state.

## Most recent completed checks
- `tests/unit/test_backend_bo3_capture_contract.py` now covers the default active corpus path, normal BO3 append behavior, stable same-match identity continuity, and explicit refusal/quarantine of a forced mid-session team flip.
- `tests/unit/test_runner_map_identity.py` still confirms the runner's per-map identity cache behavior stays stable for normal BO3 map identity establishment.
- `tests/unit/test_run_backend_bo3_corpus_readiness_analyzer.py` still confirms the corpus analyzer follows the continuity-protected active corpus default while preserving separation from the bounded diagnostic.
- `tests/unit/test_run_backend_bo3_live_parity_diagnostic.py` still confirms that the bounded diagnostic remains a one-match tool on the frozen snapshot path.
- `tests/unit/test_runner_bo3_hold.py`, `tests/unit/test_state_corridor_labels.py`, and `tests/unit/test_memory_store_score_diag.py` remain the focused checks confirming BO3 `round_result` emission and persisted history wire output preserve `match_id` without dropping existing fields.
- `tests/unit/test_export_backend_bo3_live_round_calibration_evidence.py` remains the focused exporter test surface covering same-match join, wrong-match refusal via `match_id`, duplicate collapse, strict-later leakage refusal, conflicting round_result refusal, malformed-row accounting, and artifact/report shape.
- `tests/unit/test_telemetry_session.py` now confirms BO3 session diagnostics expose the new `bo3_pipeline` fetch -> suppression -> propagation view for `/debug/telemetry/sessions`.

## Current initiative status
- **Actual current runtime BO3 ingestion path:** `backend/services/runner.py`.
- **Promoted `master` upstream/downstream label truth:** persisted BO3 `round_result` history rows carry `match_id`, and the promoted exporter can build a same-match, leakage-aware BO3 live labeled round-level evidence surface.
- **Current branch-local BO3 audit truth:** this branch adds narrow per-session BO3 pipeline diagnostics that show fetch attempt/success timing, source snapshot identifiers, suppression decisions, and emit/store/broadcast timing through `/debug/telemetry/sessions`.
- **Truth boundary:** this branch-local stage would prove that the repo now exposes where a BO3 update was fetched, suppressed, accepted, and propagated. It would not prove the poller is healthy, the source is timely, or that a fix has already been made for any specific lag class.

## Next likely step
- Review this branch strictly for whether the new `bo3_pipeline` diagnostics are narrow, truthful, and sufficient to localize BO3 fetch vs suppression vs propagation loss during a live match.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit only when a branch actually has newer work than promoted `master`.
