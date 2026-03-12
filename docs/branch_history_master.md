# Branch History - `master`

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

## 2026-03-10 - [LOCAL STAGE] Bounded real-runtime BO3 live parity diagnostic on the committed backend capture artifact
- **Branch:** `stage/backend-bo3-full-live-capture` (local stage; not promoted)
- **Initiative / phase:** Local-stage bounded live comparison-surface diagnostic on top of the promoted backend-native BO3 capture contract
- **Summary of local stage work:** Added one thin consumer for `logs/bo3_backend_live_capture_contract.jsonl` that reconstructs bounded comparison inputs from committed backend capture rows, excludes unfit rows explicitly, compares truthfully eligible `IN_PROGRESS` rows from the dominant captured match against the bounded V2 reference target, and emits one machine-readable report at `automation/reports/backend_bo3_live_parity_diagnostic_report.json`.
- **Diagnostic result:** the committed artifact spans match ids `111953` (`4` rows) and `113437` (`459` rows); the diagnostic selected dominant match `113437`, excluded `409` rows explicitly, kept `54` per-tick eligible rows visible, counted only `17` distinct truthfully comparable raw events as independent evidence, and returned `decision = inconclusive`.
- **Why this local stage matters:** The branch can now do more than store real runtime capture data. It can ask whether the current live lane looks wrong, close, or inconclusive on the committed artifact.
- **Risks / red flags:** This is still one bounded diagnostic on one committed artifact. It is not live parity implementation, not replay/live linkage, not broad representativeness, and not universal proof of truth.


## 2026-03-10 - [LOCAL STAGE] Full backend BO3 live capture artifact from one real local run
- **Branch:** `stage/backend-bo3-full-live-capture` (local stage; not promoted)
- **Initiative / phase:** Local-stage full artifact availability step on top of the promoted backend-native BO3 capture contract
- **Summary of local stage work:** Commit `34ec589` adds the full backend-native BO3 capture artifact file `logs/bo3_backend_live_capture_contract.jsonl` to the branch so the collected live data is directly visible in git rather than only on disk.
- **Committed artifact truth:** the committed artifact contains `463` rows; local validation on that committed file was `total=463`, `valid=463`, `pct=100%`.
