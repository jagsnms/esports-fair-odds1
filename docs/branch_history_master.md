# Branch History - `master`

## 2026-03-12 - [LOCAL STAGE] Backend BO3 corpus continuity protection against git/worktree hazards
- **Branch:** `codex/backend-bo3-corpus-continuity-protection` (local stage; not promoted)
- **Initiative / phase:** Local-stage continuity-protection step after confirming that stash/worktree state could silently roll the active BO3 corpus back to an older baseline.
- **Summary of local stage work:** Moved the default active BO3 corpus path out of ordinary repo worktree hazard to `%LOCALAPPDATA%\EsportsFairOdds\corpus\bo3_backend_live_capture_contract.jsonl` (or `BO3_BACKEND_CAPTURE_PATH` if deliberately overridden), repointed the corpus-readiness analyzer to that same active path, and updated reset/docs wording so the old in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path is no longer treated as the continuity-protected active store.
- **Why this local stage matters:** Previously collected rows should no longer silently disappear from the active corpus just because git/worktree state puts an older in-repo file back on disk.
- **Continuity truth:** the active corpus now lives outside ordinary repo-state hazard by default; repo-visible frozen snapshots remain separate under `automation/reports/`; the bounded one-match diagnostic remains separate and still reads the frozen snapshot path.
- **Risks / red flags:** This is continuity protection only. It is not broader storage redesign, not calibration work, not live parity, and not replay/live linkage.

## 2026-03-12 - [LOCAL STAGE] Backend BO3 corpus-level evidence readiness analyzer
- **Branch:** `codex/backend-bo3-corpus-readiness-analyzer` (local stage; not promoted)
- **Initiative / phase:** Local-stage corpus-readiness step on top of the promoted BO3 corpus-contract correction
- **Summary of local stage work:** Added one separate corpus-level readiness analyzer at `tools/run_backend_bo3_corpus_readiness_analyzer.py`, added one narrow shared helper at `tools/backend_bo3_capture_analysis_common.py`, added focused unit coverage at `tests/unit/test_run_backend_bo3_corpus_readiness_analyzer.py`, added one machine-readable local-stage report artifact at `automation/reports/backend_bo3_corpus_readiness_report.json`, and kept the existing bounded one-match diagnostic intact as its own truth surface.
- **Current local-stage report artifact note:** the carried corpus-readiness report is a point-in-time run against the mutable local corpus state available when the analyzer was executed on this branch. Its counts are useful local evidence, but they are not frozen promoted-`master` corpus truth.
- **Current local-stage analyzer result captured in the carried report artifact:** the report records `blockage_assessment = easing_multi_match`, `distinct_match_count = 2`, `eligible_compared_row_count = 298`, `distinct_raw_event_count = 93`, and `distinct_signal_active_raw_event_count = 90`, with `not_in_progress_phase` as the dominant blocker (`330` excluded rows).
- **Why this local stage matters:** The lane now has one narrow corpus-level answer to the question that matters next: whether the growing multi-match live corpus is becoming useful enough to unblock meaningful next engine work.
- **Truth boundary:** This analyzer is separate from `tools/run_backend_bo3_live_parity_diagnostic.py`. The old tool remains the bounded single-match sanity surface; the new tool is corpus-level readiness only.
- **Risks / red flags:** This is still not calibration implementation, not live parity, not replay/live linkage, and not a broader analytics platform.

## 2026-03-12 - [LOCAL STAGE] Backend BO3 capture corpus contract correction
- **Branch:** `codex/backend-bo3-corpus-contract-correction` (local stage; not promoted)
- **Initiative / phase:** Local-stage correction step after the promoted lifecycle split over-weighted snapshot neatness relative to the actual corpus-growth mission
- **Summary of local stage work:** Restored `logs/bo3_backend_live_capture_contract.jsonl` as the canonical persistent accumulating BO3 corpus, kept `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl` only as an optional frozen cut of that corpus, and changed reset/docs/diagnostic framing so snapshots support the corpus instead of replacing it.
- **Why this local stage matters:** The lane now protects accumulation first. Normal collection workflow no longer treats the main BO3 capture corpus like disposable runtime state.
- **Reset / git truth:** reset preserves the corpus path; frozen snapshots stay separate; the bounded diagnostic still reads a snapshot cut, but that snapshot is secondary to the corpus.
- **Risks / red flags:** This is corpus-contract correction only. It does not redesign broader logging, parity, or replay/live linkage.
## 2026-03-11 - [LOCAL STAGE] Backend BO3 capture artifact lifecycle contract clarification
- **Branch:** `codex/backend-bo3-lifecycle-contract` (local stage; not promoted)
- **Initiative / phase:** Local-stage lifecycle-clarification step on top of the promoted backend-native BO3 capture contract
- **Summary of local stage work:** Split the old ambiguous single-path role so the real backend runtime capture log now writes to `logs/runtime/bo3_backend_live_capture_contract.jsonl`, the deliberate versioned evidence snapshot is `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl`, and the bounded diagnostic consumer now points at that snapshot role instead of the disposable runtime path.
- **Why this local stage matters:** The repo stops pretending the same filesystem path is both a disposable runtime log and a durable committed evidence artifact.
- **Reset / git truth:** normal reset flow continues to treat the runtime log as disposable, while the versioned evidence snapshot is non-runtime and no longer silently shares the runtime path.
- **Risks / red flags:** This is lifecycle-contract clarification only. It does not change live parity, replay/live linkage, or broader logging architecture.
