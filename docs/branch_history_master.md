# Branch History - `master`

## 2026-03-13 - [LOCAL STAGE] BO3 live capture canonical team identity establishment fix
- **Branch:** `codex/backend-bo3-canonical-identity-establishment` (local stage; not promoted)
- **Initiative / phase:** Local-stage runtime/capture integrity step after confirming that the promoted continuity guard stops later drift but fresh BO3 matches can still start with the wrong team identity established from the outset.
- **Summary of local stage work:** Hardened the BO3 capture start condition in `backend/services/runner.py`, added focused deterministic coverage in `tests/unit/test_backend_bo3_capture_contract.py`, and now skip normal BO3 capture rows until per-session raw match identity is established for that `(match_id, game_number, map_index)`.
- **Why this local stage matters:** Fresh BO3 accumulation should not begin from a weak first frame or another match's cached team identity. This stage narrows the fix to trustworthy identity establishment before normal capture starts.
- **Truth boundary:** This stage only protects fresh BO3 capture rows from entering the normal corpus with wrongly established canonical team identity. It does not repair historical contaminated rows, does not change exporter logic, and does not redesign BO3 runtime identity handling broadly.
- **Risks / red flags:** This stage still depends on the raw BO3 identity surface being the trustworthy source for initial per-session identity establishment. If raw identity is itself wrong, this stage will correctly refuse weak starts but will not retroactively repair already-written history.
## 2026-03-12 - [LOCAL STAGE] BO3 live capture team-identity continuity integrity fix
- **Branch:** `codex/backend-bo3-team-identity-continuity-fix` (local stage; not promoted)
- **Initiative / phase:** Local-stage runtime/capture integrity step after confirming that fresh BO3 capture rows could flip `team_one_id`, `team_two_id`, and `team_a_is_team_one` under the same `match_id`.
- **Summary of local stage work:** Added a same-match identity continuity guard in `backend/services/bo3_capture_contract.py`, added focused deterministic coverage in `tests/unit/test_backend_bo3_capture_contract.py`, and now refuse conflicting later BO3 capture rows by keeping them out of the normal corpus and writing a visible `_identity_conflicts.jsonl` sidecar record instead.
- **Why this local stage matters:** Fresh BO3 accumulation should not continue silently appending contradictory team identity slices under one match id. This stage narrows the defect to a visible refusal/quarantine path rather than letting corrupted rows look normal.
- **Truth boundary:** This stage only protects fresh BO3 capture accumulation from mid-session identity drift at the write boundary. It does not repair historical contaminated rows, does not change exporter logic, and does not redesign BO3 runtime identity handling broadly.
- **Risks / red flags:** This stage assumes the first accepted identity for a `match_id` is the canonical one for subsequent writes, including after restart by seeding from the existing corpus file. If upstream runtime identity selection is still wrong, the guard will preserve continuity by refusing later drift, not by retroactively correcting earlier bad rows.

## 2026-03-12 - [LOCAL STAGE] Backend BO3 live labeled calibration evidence bridge
- **Branch:** `codex/backend-bo3-live-labeled-calibration-evidence-bridge-v2` (local stage; not promoted)
- **Initiative / phase:** Local-stage downstream evidence/export step resumed only after promoted `master` gained BO3 `round_result.match_id` emission.
- **Summary of local stage work:** Added `tools/export_backend_bo3_live_round_calibration_evidence.py`, added focused deterministic coverage in `tests/unit/test_export_backend_bo3_live_round_calibration_evidence.py`, and exported one narrow round-level BO3 live labeled calibration evidence surface for `q_intra_total` vs `round_result` only.
- **Why this local stage matters:** The repo can now attempt a same-match, leakage-aware BO3 live labeled evidence export instead of guessing label identity from round/team shape alone.
- **Current local artifact note:** The branch-local exporter wrote point-in-time local artifacts at `automation/reports/backend_bo3_live_round_calibration_evidence_v1.json` and `automation/reports/backend_bo3_live_round_calibration_evidence_report_v1.json`. The current local run labeled `0` records because many persisted `round_result` rows in the local `history_points.jsonl` still predate the promoted `match_id` emission bridge; those local counts are not promoted repo truth.
- **Truth boundary:** This stage only adds a narrow exporter for round-level `q_intra_total` vs `round_result`, with strict later-than timing, conservative duplicate collapse, same-match `match_id` join, and explicit malformed-row accounting. It does not prove calibration quality, add `p_hat` / `segment_result`, or change runtime BO3 behavior.
- **Risks / red flags:** Current local export volume still depends on how much persisted history was recorded before label-side `match_id` existed. A truthful exporter can still produce very little labeled evidence if the local history surface is mostly pre-bridge data.

## 2026-03-12 - [LOCAL STAGE] Backend BO3 round_result match-identity emission bridge
- **Branch:** `codex/backend-bo3-round-result-match-identity-bridge` (local stage; not promoted)
- **Initiative / phase:** Local-stage upstream identity-surface step created specifically to unblock the previously blocked BO3 live labeled calibration evidence exporter.
- **Summary of local stage work:** Added `match_id` to persisted BO3 `round_result` history rows, preserved the field through history wire/persistence output, and added focused unit coverage proving BO3 round_result emission still carries the existing round/map/team fields while now also carrying per-match identity.
- **Why this local stage matters:** Downstream BO3 calibration evidence cannot be promotion-safe while `round_result` labels lack a true per-match discriminator. This stage closes that upstream blocker narrowly.
- **Truth boundary:** This stage only adds BO3 `round_result.match_id` emission/persistence. It does not change steady-state collection semantics, calibration math, replay/live architecture, or exporter promotion readiness by itself.
- **Risks / red flags:** The blocked downstream exporter still needs its own small truth-fix pass after this lands; this stage only supplies the missing upstream identity surface.

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

