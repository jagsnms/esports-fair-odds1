# Branch History - `master`

## 2026-03-20 - [LOCAL STAGE] Q-Intra Real-World Calibration Program: BO3 segment_result match_id propagation fix
- **Branch:** `codex/bo3-segment-result-match-id-propagation-fix-stage1` (local stage; not promoted)
- **Larger project:** `Q-Intra Real-World Calibration Program`
- **Current bounded stage:** `BO3 segment_result match_id propagation fix`
- **Initiative / phase:** Narrow history-truth correction after live map-final diagnostics proved the audited BO3 `segment_result` was emitted but lacked top-level `match_id`, so normal match-scoped inspection treated it as absent.
- **Summary of local stage work:** Updated `backend/services/runner.py`, added focused deterministic coverage in `tests/unit/test_runner_bo3_hold.py`, and now propagate top-level `match_id` onto emitted BO3 `segment_result` history points while leaving credible increment detection and existing emit timing unchanged.
- **Why this local stage matters:** Round-result evidence is now healthier on `master`, but the larger q-intra project still benefits from truthful map-result history surfaces. This stage fixes a localized visibility defect in that live history narrative without reopening broader BO3 finality work.
- **Checks run on the branch:** `tests/unit/test_runner_bo3_hold.py` passed; `tests/unit/test_runner_map_identity.py` passed.
- **Truth boundary:** This stage only fixes `segment_result` top-level `match_id` propagation. It does not redesign round_result logic, does not redesign map/series finalization, does not touch exporter/gate/measurement tooling, does not retune q, and does not calibrate p_hat or rails.
- **Current state:** Committed on the project branch and awaiting final promotion decision.

## 2026-03-19 - [LOCAL STAGE] Q-Intra Real-World Calibration Program: BO3 stale score-baseline round_result seeding fix
- **Branch:** `codex/bo3-stale-score-baseline-seeding-fix-stage1` (local stage; not promoted)
- **Larger project:** `Q-Intra Real-World Calibration Program`
- **Current bounded stage:** `BO3 stale score-baseline round_result seeding fix`
- **Initiative / phase:** Narrow evidence-truth defect correction after runner-only diagnostics localized the audited round-12 wrong-winner path to stale prior-round score baseline contaminating new-round winner seeding.
- **Summary of local stage work:** Updated `backend/services/runner.py`, added focused deterministic coverage in `tests/unit/test_runner_bo3_hold.py`, and now require BO3 score-delta seeding for the current round to use a baseline valid for the current round context so stale prior-round terminal score does not poison new-round pending winner state.
- **Why this local stage matters:** The larger q-intra program depends on truthful explicit `round_result` rows for labeled evidence export. This local stage fixes a repo-side wrong-winner contamination path in that label surface without redesigning broader BO3 outcome handling.
- **Checks run on the branch:** `tests/unit/test_runner_bo3_hold.py` passed; `tests/unit/test_runner_map_identity.py` passed.
- **Truth boundary:** This stage preserves normal same-round score-first reconciliation and existing lawful emit timing only. It does not redesign map-result logic, does not redesign series/match finalization, does not touch exporter/gate/measurement tooling, does not retune q, and does not calibrate p_hat or rails.
- **Current state:** Committed on the project branch and awaiting final promotion decision.

## 2026-03-19 - [LOCAL STAGE] Q-Intra Real-World Calibration Program: BO3 round_result score-delta carryover fix
- **Branch:** `codex/bo3-round-result-carryover-fix-stage1` (local stage; not promoted)
- **Larger project:** `Q-Intra Real-World Calibration Program`
- **Current bounded stage:** `BO3 round_result score-delta carryover fix`
- **Initiative / phase:** Narrow evidence-accumulation defect correction after live match audit showed repeated missed `round_result` emissions when BO3 score delta became knowable before the lawful emit boundary.
- **Summary of local stage work:** Updated `backend/services/runner.py`, added focused deterministic coverage in `tests/unit/test_runner_bo3_hold.py`, and now preserve BO3 round winner information across same-round ticks until the existing lawful emit boundary so the demonstrated same-round score-delta loss mode no longer drops completed rounds before labeling.
- **Why this local stage matters:** The larger q-intra program depends on explicit `round_result` rows for labeled evidence export. This local stage fixes a repo-side suppression defect in that label path instead of broadening into more measurement tooling.
- **Checks run on the branch:** `tests/unit/test_runner_bo3_hold.py` passed; `tests/unit/test_runner_map_identity.py` passed.
- **Truth boundary:** This stage preserves existing lawful emit timing only. It does not redesign map-result logic, does not redesign series/match finalization, does not touch exporter/gate/measurement tooling, does not retune q, and does not calibrate p_hat or rails.
- **Current state:** Committed on the project branch and awaiting final promotion decision.

## 2026-03-19 - [LOCAL STAGE] Q-Intra Real-World Calibration Program: BO3 live q_intra measurement orchestration runner
- **Branch:** `codex/bo3-live-q-intra-measurement-runner-stage1` (local stage; not promoted)
- **Larger project:** `Q-Intra Real-World Calibration Program`
- **Current bounded stage:** `BO3 live q_intra measurement orchestration runner`
- **Initiative / phase:** Second bounded measurement/evidence stage after the BO3 live q_intra reliability/sufficiency artifact landed on `master`.
- **Summary of local stage work:** Added `tools/run_backend_bo3_live_q_intra_measurement.py`, added focused deterministic coverage in `tests/unit/test_run_backend_bo3_live_q_intra_measurement.py`, and now provide one canonical end-to-end q_intra measurement run path that executes the existing BO3 live labeled evidence exporter and then the existing q_intra reliability gate with stdout-only orchestration output.
- **Why this local stage matters:** The larger q-intra program now has one standard way to run its current measurement loop instead of relying on manual exporter-then-gate chaining.
- **Checks run on the branch:** `tests/unit/test_run_backend_bo3_live_q_intra_measurement.py` passed; `tests/unit/test_run_backend_bo3_live_q_intra_reliability_gate.py` passed; `tests/unit/test_export_backend_bo3_live_round_calibration_evidence.py` passed.
- **Truth boundary:** This stage is orchestration only. It does not add a new artifact layer, does not add new calibration metrics or readiness semantics, does not retune q, does not calibrate p_hat or rails, and does not change BO3 runtime/upstream behavior.
- **Current state:** Committed on the project branch and awaiting final promotion decision.

## 2026-03-19 - [LOCAL STAGE] Q-Intra Real-World Calibration Program: BO3 live q_intra reliability sufficiency artifact
- **Branch:** `codex/bo3-live-q-intra-reliability-gate-stage1` (local stage; not promoted)
- **Larger project:** `Q-Intra Real-World Calibration Program`
- **Current bounded stage:** `BO3 live q_intra reliability sufficiency artifact`
- **Initiative / phase:** First bounded measurement-only stage after promoted `master` gained a narrow BO3 live labeled round-level exporter for `q_intra_total` vs `round_result`.
- **Summary of local stage work:** Added `tools/run_backend_bo3_live_q_intra_reliability_gate.py`, added `tools/schemas/backend_bo3_live_q_intra_reliability_gate.schema.json`, added focused deterministic coverage in `tests/unit/test_run_backend_bo3_live_q_intra_reliability_gate.py`, and now emit one machine-readable BO3 live q_intra reliability/sufficiency artifact from the existing detailed labeled evidence export.
- **Why this local stage matters:** The repo can now turn promoted BO3 live labeled q_intra evidence into one explicit calibration-measurement artifact with sufficiency signaling instead of stopping at raw exported labeled rows.
- **Checks run on the branch:** `tests/unit/test_run_backend_bo3_live_q_intra_reliability_gate.py` passed; `tests/unit/test_export_backend_bo3_live_round_calibration_evidence.py` passed.
- **Truth boundary:** This stage is measurement-only. It does not retune q, does not calibrate p_hat, does not calibrate rails, does not certify calibration quality, does not change engine math, and does not reopen replay/canonical matching or BO3 upstream/coarse-progression work.
- **Current state:** Committed on the project branch and awaiting final promotion decision.

## 2026-03-19 - [LOCAL STAGE] Kalshi robustness packet recovery and clean promotion path
- **Branch:** `codex/kalshi-client-robustness-recovery` (local stage; not promoted)
- **Initiative / phase:** Bounded Kalshi quote fetch/parsing recovery after live runtime status showed polling was active with a selected ticker but quote acquisition was failing on current Kalshi response shapes.
- **Summary of local stage work:** Hardened `engine/market/kalshi_client.py` to accept current dollar-denominated market quote fields and `orderbook_fp` / `yes_dollars` / `no_dollars` orderbook fallback shapes, added focused deterministic coverage in `tests/unit/test_kalshi_client.py`, and preserved honest failure when no lawful two-sided quote can be derived.
- **Why this local stage matters:** `master` currently exposes market runtime truth but still carries the older Kalshi quote parser. This local stage is the bounded recovery packet intended to restore lawful live quote acquisition without touching runner/chart logic.
- **Checks run on the branch:** `tests/unit/test_kalshi_client.py` passed; `tests/unit/test_market_runtime_status.py` passed.
- **Truth boundary:** This stage is Kalshi client robustness only. It does not change runner behavior, chart rendering, replay/simulation surfaces, or BO3 behavior.
- **Current state:** Committed on the project branch and awaiting final push/promotion decision.

## [BACKFILLED] 2026-03-15 - Bounded replay point-source contract
- **Branch:** `master`
- **Initiative / phase:** Bounded replay-side source-contract step
- **Project commit:**
  - `cd289f7942ea17415136e19928fed513c0b9d92d` `Add bounded replay point-source contract`
- **Summary of push:** Promoted one bounded replay-side `replay_point_source` contract under the replay validation summary artifact, limited to lawful point fields already captured by the replay assessment path.
- **Key files/subsystems touched:**
  - `tools/replay_verification_assess.py`
  - `tools/schemas/replay_validation_summary.schema.json`
  - `tools/validate_replay_validation_summary.py`
  - `tests/unit/test_replay_verification_assess_stage1.py`
  - `tests/unit/test_validate_replay_validation_summary.py`
- **Truth boundary:** This stage excludes `point.event`, broad `derived`, and `derived.debug`. It does not introduce comparison logic, alignment work, or broader replay/simulation architecture changes.

## [BACKFILLED] 2026-03-15 - Common point-source basis metadata
- **Branch:** `master`
- **Initiative / phase:** Bounded lawful shared-vocabulary metadata step
- **Project commit:**
  - `7c57ea2fbaeb315f0f2079c45bbae185b25875c1` `Add common point-source basis metadata`
- **Summary of push:** Promoted one shared `common_point_source_basis` descriptor on replay-side and canonical-side source surfaces for the six lawful overlapping fields only.
- **Key files/subsystems touched:**
  - `tools/replay_verification_assess.py`
  - `tools/schemas/replay_validation_summary.schema.json`
  - `tools/validate_replay_validation_summary.py`
  - `engine/simulation/phase2.py`
  - `tests/unit/test_replay_verification_assess_stage1.py`
  - `tests/unit/test_validate_replay_validation_summary.py`
  - `tests/simulation/test_phase2_trace_export.py`
- **Truth boundary:** This stage is metadata-only and explicitly sets `record_matching_implied = false`, `alignment_implied = false`, and `scoring_or_selection_implied = false`.

## [BACKFILLED] 2026-03-15 - Common point-source projection support
- **Branch:** `master`
- **Initiative / phase:** Bounded shared-field projection contract step
- **Project commit:**
  - `90a25e9e41b08f1d2578b5ff989eaf53885b7112` `Add common point-source projection support`
- **Summary of push:** Promoted side-local `common_point_source_projection` surfaces on replay and canonical sides, each emitting only the six lawful shared fields without implying matching or alignment.
- **Key files/subsystems touched:**
  - `tools/replay_verification_assess.py`
  - `engine/simulation/phase2.py`
  - `tools/schemas/replay_validation_summary.schema.json`
  - `tools/validate_replay_validation_summary.py`
  - `tests/unit/test_replay_verification_assess_stage1.py`
  - `tests/unit/test_validate_replay_validation_summary.py`
  - `tests/simulation/test_phase2_trace_export.py`
- **Truth boundary:** This stage adds side-local projections only. It does not create joined records, lawful replay-to-canonical point matching, or comparison logic.

## [BACKFILLED] 2026-03-15 - Market runtime status visibility
- **Branch:** `master`
- **Initiative / phase:** Bounded market runtime visibility step
- **Project commit:**
  - `1948ab27849e3b46cd8c340d06ac07daf6cf3b24` `Add market runtime status visibility`
- **Summary of push:** Promoted machine-readable market runtime status in the backend and Market panel so selected ticker state, polling activity, inactive reason, quote status, and last error are visible to the operator.
- **Key files/subsystems touched:**
  - `backend/api/routes_market.py`
  - `backend/services/market_buffer.py`
  - `backend/services/runner.py`
  - `frontend/src/App.tsx`
  - `tests/unit/test_market_runtime_status.py`
- **Truth boundary:** This stage is runtime visibility only. It does not add ticker persistence, does not redesign the chart, and does not fix quote acquisition by itself.


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

## 2026-03-10 - Replay-anchored multi-source decision contract (`balanced_v1` vs `eco_bias_v1`)
- **Branch:** `master`
- **Initiative / phase:** Bounded replay-anchored two-source decision contract (fixed seed `20260310`)
- **Summary of push:** Promoted one replay-primary decision surface that evaluates exactly two canonical simulation sources, `balanced_v1` and `eco_bias_v1`, against one replay input and emits one combined machine-readable decision artifact.
- **Project commits:**
  - `81fec184069f2fc04b30d5b209a0a72314e95e42` `Add replay anchored multisource decision contract`
  - `11fe43ee1ffe938524b8db70095c674fec8bfe75` `Tighten replay multisource no-difference rule`
  - `a77922fcadfd353b7164606b32647670cfd5fae9` `Refresh replay multisource stage docs`
- **Key files/subsystems touched:**
  - `tools/run_replay_simulation_validation_pilot.py`
  - `tools/run_replay_multisource_decision.py`
  - `tests/unit/test_run_replay_multisource_decision.py`
  - `automation/reports/replay_multisource_decision_balanced_v1_vs_eco_bias_v1_seed20260310.json`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Bounded decision-contract output:** one combined replay-anchored artifact with exactly two source blocks (`balanced_v1` and `eco_bias_v1`), explicit source contracts and fixed seed/shape basis, per-source local sanity/alignment/failed-check state, and one explicit decision block with `decision_basis = "replay_anchored_multi_source"`.
- **Tests/checks run and result:** `tests/unit/test_run_replay_simulation_validation_pilot.py` passed; `tests/unit/test_run_replay_multisource_decision.py` passed, including the focused equal-count / different-failed-check regression case; the direct `.venv311` replay-anchored multi-source decision CLI validation completed successfully after bypassing the local environment/sandbox launcher quirk and emitted the combined artifact; the bounded direct `.venv311` single-source replay/simulation pilot CLI validation also completed successfully and remained supported.
- **Current replay-anchored outcome:** the promoted combined decision artifact remains truthful but `inconclusive` on `tools/fixtures/replay_carryover_complete_v1.jsonl` because neither canonical source satisfies the bounded alignment threshold on that replay slice.
- **Risks / red flags:** This is still one bounded replay-anchored two-source pressure test only. It is not broad replay resolution, not broad representativeness, and not general simulation/calibration completion.
- **Why this push matters:** `master` can now ask the right question in the right order: not just whether the two canonical sources differ, but whether replay truth distinguishes them enough to matter.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality rather than assuming more replay/simulation expansion automatically.

## [BACKFILLED] 2026-03-10 - Master catch-up merge from `agent-initiative-base`
- **Branch:** `master`
- **Initiative / phase:** Deliberate source-of-truth catch-up merge
- **Summary of push:** Merged the approved branch-only groundwork from `agent-initiative-base` into `master` so future final landings can target `master` directly.
- **Why it mattered:** `master` had been missing the canonical engine/replay/simulation/doc surfaces needed for current work.
- **Risks / red flags:** Large merge by scope; future landings still need normal staged review discipline.
- **Checks at that time:** full canonical pytest passed before and after merge.
- **Next likely step (at that time):** Continue only the highest-leverage approved project work directly on `master`.

## [BACKFILLED] 2026-03-10 - Phase 2 bounded policy-driven canonical simulation baseline
- **Branch:** `master`
- **Initiative / phase:** Phase 2 Stage 1 bounded policy-driven canonical simulation contract (`balanced_v1` only)
- **Summary of push:** Landed the first bounded policy-driven canonical simulation artifact on one deterministic `balanced_v1` slice with truthful synthetic replay metadata.
- **Why it mattered:** Opened the first replay-comparable Phase 2 simulation contract instead of leaving simulation policy work in a separate synthetic lane.
- **Risks / red flags:** At that point the slice still had `rail_input_v2_activated_points = 0`, so richer carryover-complete semantics were not established yet.
- **Checks at that time:** focused Phase 2 simulation tests passed; CLI emitted deterministic machine-readable JSON.
- **Next likely step (at that time):** Decide whether to make the bounded `balanced_v1` slice carryover-complete enough to activate V2 before any profile expansion.

## 2026-03-10 - Phase 2 bounded V2 activation on the landed `balanced_v1` slice
- **Branch:** `master`
- **Initiative / phase:** Phase 2 bounded carryover-completeness correction (`balanced_v1` only)
- **Summary of push:** Added the minimum missing contract input needed for the existing landed `balanced_v1` Phase 2 slice to satisfy the carryover-completeness gate through the real replay assessment path, and tightened the focused Phase 2 test to lock that behavior down explicitly.
- **Key files/subsystems touched:**
  - `engine/simulation/phase2.py`
  - `tests/simulation/test_phase2_policy_contract.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Tests/checks run and result:** focused Phase 2 unittest passed; repeated CLI runs for seed `20260310` remained deterministic; emitted artifact showed `policy_profile="balanced_v1"`, `assessment_prematch_map=0.55`, `rail_input_v2_activated_points=159`, `rail_input_v1_fallback_points=0`, no `V2_REQUIRED_FIELDS_MISSING`, `required_complete_points=159`, `required_incomplete_points=0`, and zero structural / behavioral / invariant violations.
- **Risks / red flags:** This is bounded V2 activation on one slice only. It does not broaden profiles, seeds, calibration/export integration, or complete all Phase 2 semantics.
- **Why this push matters:** It turns the landed Phase 2 slice from policy-driven-but-fallback-only into policy-driven-with-real-V2-activation through the canonical path.
- **Next likely step (at this time):** Re-rank whether a bounded next Phase 2 semantic extension still beats pausing or another Bible-level project.

## 2026-03-10 - Canonical simulation trace export with per-point prediction/outcome labels
- **Branch:** `master`
- **Initiative / phase:** Bounded canonical trace-export/source-contract step (`balanced_v1` only)
- **Summary of push:** Landed one deterministic machine-readable trace-export path for the canonical `balanced_v1` simulation lane so prediction points are paired only to truthful runner-emitted `round_result` labels, and removed the duplicate canonical execution introduced in the initial implementation by reusing the existing canonical assessment pass.
- **Project commits:**
  - `897f97400e21e6099ba4abc887d9d55eaca0c9cb` `Add canonical Phase 2 trace export contract`
  - `374485d9df58f12213076ffb4716cf0729f061c6` `Reuse assessment pass for Phase 2 trace export`
  - `d1dfab139698fc31ef65c9e71fc50207ccf3b99c` `Update master docs for validated trace export stage`
- **Key files/subsystems touched:**
  - `engine/simulation/phase2.py`
  - `tools/replay_verification_assess.py`
  - `tests/simulation/test_phase2_trace_export.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Tests/checks run and result:** approved validations passed: `tests/unit/test_run_replay_simulation_validation_pilot.py`, `tests/simulation/test_phase2_policy_contract.py`, and `tests/simulation/test_phase2_trace_export.py`; repeated `tools/simulate_phase2.py --seed 20260310` runs emitted identical machine-readable output.
- **Risks / red flags:** This is still one bounded `balanced_v1` slice only. The export is a source-contract step, not a calibration lane, and unlabeled final-round prediction points are excluded rather than given pseudo-labels.
- **Why this push matters:** It opens the truthful raw source contract that later downstream evidence work was missing, without faking calibration-ready outputs or broadening simulation semantics.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality rather than continuing by inertia.

## 2026-03-10 - Bounded true simulation calibration evidence from canonical Phase 2 trace export
- **Branch:** `master`
- **Initiative / phase:** Bounded simulation evidence-path step (`balanced_v1` only, fixed seed `20260310`)
- **Summary of push:** Landed one truthful simulation calibration evidence path sourced only from explicit canonical `balanced_v1` trace inputs, replacing the prior hard-disabled simulation export state for this bounded source with gate/schema-compatible baseline/current simulation evidence records.
- **Project commit:**
  - `4b0147761780c64e919c97d5b4eab1303714f283` `Add bounded canonical simulation calibration evidence path`
- **Key files/subsystems touched:**
  - `tools/export_calibration_reliability_evidence.py`
  - `tools/calibration_reliability_evidence_gate.py`
  - `tools/fixtures/calibration_reliability_simulation_exported_v1.json`
  - `tools/fixtures/canonical_phase2_balanced_v1_trace_baseline_v1.json`
  - `tools/fixtures/canonical_phase2_balanced_v1_trace_current_v1.json`
  - `automation/reports/calibration_reliability_evidence_export_manifest_v1.json`
  - `tests/unit/test_export_calibration_reliability_evidence.py`
  - `tests/unit/test_run_calibration_reliability_evidence_gate.py`
  - `tests/unit/test_calibration_reliability_evidence_schema.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Explicit baseline/current input method:** Two explicit canonical trace-export JSON inputs are used, one for `baseline` and one for `current`. In this bounded stage they are separate files but identical `balanced_v1`/seed `20260310` traces, so the lane is truthful about source identity without claiming comparative improvement.
- **Tests/checks run and result:** `tests/unit/test_export_calibration_reliability_evidence.py`, `tests/unit/test_run_calibration_reliability_evidence_gate.py`, `tests/unit/test_calibration_reliability_evidence_schema.py`, and `tests/simulation/test_phase2_trace_export.py` all passed; repeated `tools/simulate_phase2.py --seed 20260310` runs remained deterministic; the bounded calibration export path emitted simulation evidence records and truthful manifest provenance including unlabeled-point exclusion counts.
- **Risks / red flags:** This is still bounded `balanced_v1` evidence only, not calibration redesign or general simulation-calibration completion. Baseline/current are explicit but identical at this stage, and final-round prediction points remain unlabeled under current canonical semantics and are excluded explicitly rather than imputed.
- **Why this push matters:** The repo now has one truthful downstream simulation evidence path derived from promoted canonical trace records instead of a hard-disabled simulation side.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality without assuming more Phase 2 or broader calibration work automatically.

## 2026-03-10 - Bounded `eco_bias_v1` second source and source-vs-source comparison pressure
- **Branch:** `master`
- **Initiative / phase:** Bounded second-source pressure step (`balanced_v1` vs `eco_bias_v1`, fixed seed `20260310`)
- **Summary of push:** Promoted exactly one bounded second canonical Phase 2 source using `eco_bias_v1` on the same fixed seed/shape/truthfulness rules as `balanced_v1`, and landed one thin machine-readable source-vs-source comparison artifact that keeps source identity explicit instead of abusing baseline/current semantics.
- **Project commits:**
  - `5a50f4781099eb78309d943e467037c1da437ffc` `Add bounded eco bias Phase 2 second source`
  - `9cbaad2f6430c3e71744bb616facf2bd2c100bcd` `Correct second source validation status docs`
- **Key files/subsystems touched:**
  - `engine/simulation/phase2.py`
  - `tools/simulate_phase2.py`
  - `tools/compare_phase2_sources.py`
  - `tools/schemas/simulation_phase2_policy_summary.schema.json`
  - `tests/simulation/test_phase2_policy_contract.py`
  - `tests/simulation/test_phase2_trace_export.py`
  - `automation/reports/phase2_source_comparison_balanced_v1_vs_eco_bias_v1_seed20260310.json`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Bounded second-source contract:** `eco_bias_v1`, seed `20260310`, `round_count = 32`, `ticks_per_round = 4`, same canonical engine path, same replay-comparable assessment path, same labeled-point-only trace export rule, same explicit unlabeled-point exclusion count reporting, and zero structural / behavioral / invariant violations.
- **Tests/checks run and result:** `tests/simulation/test_phase2_policy_contract.py` and `tests/simulation/test_phase2_trace_export.py` passed; the approved `balanced_v1` CLI simulation run completed deterministically; the direct `.venv311` `eco_bias_v1` CLI validation completed successfully; the direct `.venv311` comparison CLI validation completed successfully and emitted a machine-readable artifact with explicit left/right source identity, same seed/shape basis, preserved safety floor, and non-zero family-distribution deltas.
- **Risks / red flags:** This is still only one extra bounded source and one fixed seed. It creates decision pressure, but it does not answer broader representativeness by itself and must not be misread as broad simulation/calibration completion.
- **Why this push matters:** The canonical simulation lane is no longer stuck with a single truthful source and no comparison pressure; `master` can now test whether a materially different bounded source changes the observed lane enough to justify further work.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality rather than assuming more Phase 2 expansion automatically.
