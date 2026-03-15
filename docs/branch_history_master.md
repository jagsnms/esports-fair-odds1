# Branch History - `master`


## 2026-03-14 - BO3 payload-diff observability packet
- **Initiative / phase:** Promoted BO3 observability extension after live continuity audits showed the need to distinguish repeated payloads, payload microstate changes, and stale-input conditions without changing runtime behavior.
- **Summary of promoted work:** Extended `backend/services/runner.py` with BO3 payload-diff hashes and change flags plus stale-input snapshot-status diagnostics, added focused deterministic coverage in `tests/unit/test_telemetry_session.py`, and kept the packet limited to session telemetry only.
- **Why this promoted stage matters:** BO3 audits can now see whether a fetched payload changed, which parts changed, and which stale-input conditions were present when the runner evaluated snapshot freshness, without guessing from raw repeated states alone.
- **Truth boundary:** This promoted stage is observability only. It does not change BO3 runtime behavior, does not weaken stale/fresh or safety-gate decisions, and does not claim a fix for upstream coarse progression.
- **Risks / red flags:** The legacy in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path was explicitly excluded from this promoted packet and should not be treated as part of the stage.

## 2026-03-13 - BO3 live freshness-gate clock-rewind diagnosis/fix
- **Promoted from:** `codex/backend-bo3-clock-rewind-freshness-fix`
- **Initiative / phase:** Narrow BO3 ingestion behavior correction after promoted pipeline instrumentation showed live snapshots were fetched on time but dying at `freshness_gate_reject: clock_rewind` during an active round.
- **Summary of promoted work:** Tightened `engine/ingest/bo3_freshness.py`, added focused deterministic coverage in `tests/unit/test_runner_bo3_monotonic_gate.py`, and now accept a `clock_rewind` snapshot only when it also shows explicit meaningful live advancement such as score progression, alive-count drop, bomb-planted transition, or known round-phase progression.
- **Why this promoted stage matters:** The app should not stay stuck on older accepted state solely because one clock surface rewinds if the same BO3 snapshot clearly advances the live round in other meaningful ways.
- **Truth boundary:** This promoted stage is a narrow BO3 freshness-gate behavior correction only. It does not redesign the live path, does not change exporter logic, and does not claim that all BO3 lag classes are solved.
- **Risks / red flags:** Raw snapshot logs still dedupe unchanged payloads, and this promoted stage remains conservative by preserving rejection for truly stale/regressing snapshots and for clock rewinds with no explicit advancement signal.
## 2026-03-13 - BO3.gg poller and live ingestion pipeline audit instrumentation
- **Promoted from:** `codex/backend-bo3-poller-ingestion-audit`
- **Initiative / phase:** Narrow BO3 live audit/instrumentation step after confirming that live lag concerns needed fetch -> suppression -> propagation evidence instead of guesswork.
- **Summary of promoted work:** Added narrow per-session BO3 pipeline diagnostics in `backend/services/runner.py`, added focused coverage in `tests/unit/test_telemetry_session.py`, and now expose fetch attempt/success timing, source snapshot identifiers, suppression decisions, and emit/store/broadcast timing through `/debug/telemetry/sessions` as `bo3_pipeline`.
- **Why this promoted stage matters:** The repo can now localize whether a BO3 update was fetched, suppressed, accepted, or propagated instead of inferring lag from raw capture files alone.
- **Truth boundary:** This stage is instrumentation only. It does not prove the poller is healthy, does not fix any BO3 ingestion defect, and does not redesign the live pipeline.
- **Risks / red flags:** Current raw snapshot logs still dedupe unchanged payloads, so they remain an incomplete fetch-cadence surface even after this instrumentation stage.

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
