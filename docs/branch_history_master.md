# Branch History - `master`

## 2026-03-10 - [LOCAL STAGE] Bounded BO3 live-capture/source contract for replay-anchored parity work
- **Branch:** `master` (local stage; not promoted)
- **Initiative / phase:** Stage 1 bounded BO3-authoritative live-capture/source-contract step
- **Summary of local stage work:** Added one bounded BO3-authoritative canonical live-capture contract so real BO3 auto-pull runs now produce one append-only persisted artifact row in `data/processed/cs2_replay_snapshots.parquet` with explicit raw-event linkage, replay-anchorable round identity, normalized engine-consumed frame fields, and derived intraround/parity diagnostics.
- **Project changes in scope:**
  - `bo3.gg/poller.py`
  - `legacy/app/app35_ml.py`
  - `legacy/fair_odds/logs.py`
  - `legacy/fair_odds/paths.py`
  - `tests/unit/test_bo3_live_capture_contract.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Bounded contract decision:** BO3 is the only authoritative live source for this stage. The existing `data/processed/cs2_replay_snapshots.parquet` artifact remains the canonical persisted path, but now carries explicit BO3 live-capture contract fields instead of relying on scattered raw JSONL plus optional snapshot persistence.
- **Default capture behavior change:** BO3 live auto activation now always records raw pulls to `logs/bo3_pulls.jsonl`, and canonical BO3 live snapshot persistence no longer depends on the old broad `cs2_inplay_persist` toggle.
- **Checks run and result (local stage):** `tests/unit/test_bo3_live_capture_contract.py` passed (`4 passed`), including raw-linkage mapping, append-only artifact generation, live-only persistence gating, and a parse smoke check for `legacy/app/app35_ml.py`; the focused artifact-generation check `tests/unit/test_bo3_live_capture_contract.py::test_bo3_live_capture_contract_persists_append_only_artifact` also passed and confirmed that the canonical BO3 artifact is produced, append-only, preserves raw-event linkage, includes normalized frame fields, includes derived intraround/parity diagnostics, and does not rely on the old broad toggle.
- **Risks / red flags:** This is capture-contract work only. It is not live parity implementation, not replay/live comparison logic, not BO3+GRID unification, and not proof that BO3 is sufficient for eventual full parity work.
- **Why this local stage matters:** The repo can now collect real BO3 matches into one reusable linked artifact instead of a fragmented mix of overwrite-only feed state, raw pulls, optional parquet snapshots, and downstream history logs.
- **Next likely step (from this local stage):** Review whether this bounded BO3 live-capture/source-contract step is clean enough for promotion before opening any replay/live parity project.

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

## 2026-03-10 - Replay/simulation decision layer bound to canonical `balanced_v1` Phase 2 slice
- **Branch:** `master`
- **Initiative / phase:** Bounded replay/simulation decision-layer binding (`balanced_v1` only)
- **Summary of push:** Rebound the decision layer so its simulation side now comes from the landed canonical Phase 2 `balanced_v1` slice instead of the older direct raw synthetic generator path, and preserved explicit alignment honesty by resolving the fixed unaligned slice to `inconclusive`.
- **Key files/subsystems touched:**
  - `tools/run_replay_simulation_validation_pilot.py`
  - `tests/unit/test_run_replay_simulation_validation_pilot.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Tests/checks run and result:** focused pilot pytest passed (`8 passed`); bounded pilot command against `tools/fixtures/replay_carryover_complete_v1.jsonl` with seed `20260310` emitted a machine-readable artifact showing canonical Phase 2 provenance, `alignment_achieved = false`, `selected_synthetic_rounds = null`, the fixed-slice note as both stop reason and decision reason, and `decision = inconclusive`.
- **Risks / red flags:** This binds one canonical `balanced_v1` slice only. It does not expand profiles or seeds, and it does not complete broad replay/simulation comparison.
- **Why this push matters:** Replay/simulation decisions are now about the real landed canonical Phase 2 slice rather than a separate in-progress synthetic raw lane.
- **Next likely step (at this time):** Re-rank whether a bounded next comparison/alignment step on the canonical slice beats pausing or a different Bible-level gap.

## 2026-03-10 - Bounded canonical round-alignment search for the landed `balanced_v1` Phase 2 slice
- **Branch:** `master`
- **Initiative / phase:** Bounded canonical alignment-search step (`balanced_v1` only)
- **Summary of push:** Added a tiny fixed canonical round-candidate search so the replay/simulation decision layer can try nearby canonical `balanced_v1` slices through the real Phase 2 path instead of assuming one fixed 32-round slice is the only comparison candidate.
- **Key files/subsystems touched:**
  - `engine/simulation/phase2.py`
  - `tools/run_replay_simulation_validation_pilot.py`
  - `tests/unit/test_run_replay_simulation_validation_pilot.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Tests/checks run and result:** focused pilot pytest passed (`9 passed`); bounded pilot command against `tools/fixtures/replay_carryover_complete_v1.jsonl` with seed `20260310` emitted a machine-readable artifact showing canonical Phase 2 provenance, attempted candidates `[32, 31, 33, 30, 34]`, `selected_synthetic_rounds = null`, `alignment_achieved = false`, truthful stop and decision reasons, and `decision = inconclusive`.
- **Risks / red flags:** This search surface is still tiny and fixed. It does not broaden profiles or seeds, and it does not guarantee alignment for every replay fixture.
- **Why this push matters:** The canonical slice is now decision-useful when an approved nearby canonical candidate aligns, without reopening the old raw synthetic lane or widening the project into a matrix search.
- **Next likely step (at this time):** Re-rank whether a bounded next comparison/alignment step still beats pausing or a different Bible-level gap.

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
- **Risks / red flags:** This is still one bounded `balanced_v1` slice only. The export is a source-contract step, not a calibration lane, and unlabeled final-round prediction points are excluded rather than given pseudo-labels. The earlier pytest launcher issue was an environment/sandbox quirk, not a product failure.
- **Why this push matters:** It opens the truthful raw source contract that later downstream evidence work was missing, without faking calibration-ready outputs or broadening simulation semantics.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality rather than continuing by inertia.

## 2026-03-10 - [LOCAL STAGE] Bounded true simulation calibration evidence from canonical Phase 2 trace export
- **Branch:** `codex/sim-calibration-evidence-stage1`
- **Initiative / phase:** Bounded simulation evidence-path step (`balanced_v1` only, fixed seed `20260310`)
- **Summary of local stage work:** Enabled one truthful simulation calibration evidence path sourced only from explicit canonical `balanced_v1` trace inputs, emitted bounded baseline/current simulation evidence records that match the existing gate/schema contract, and replaced the prior hard-disabled simulation export state for this bounded source only.
- **Project changes in scope:**
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
- **Checks run and result (local stage):**
  - `tests/unit/test_export_calibration_reliability_evidence.py` passed
  - `tests/unit/test_run_calibration_reliability_evidence_gate.py` passed
  - `tests/unit/test_calibration_reliability_evidence_schema.py` passed
  - `tests/simulation/test_phase2_trace_export.py` passed
  - repeated `tools/simulate_phase2.py --seed 20260310` runs remained deterministic
  - bounded calibration export path emitted simulation evidence records and truthful manifest provenance, including unlabeled-point exclusion counts
- **Risks / red flags:** This is still bounded `balanced_v1` evidence only, not a calibration redesign or broad simulation-calibration solution. Final-round prediction points remain unlabeled under current canonical semantics and are excluded explicitly rather than imputed.
- **Why this local stage matters:** The repo now has one truthful downstream simulation evidence path derived from promoted canonical trace records instead of a hard-disabled simulation side.
- **Next likely step (from this local stage):** Review whether this bounded evidence-path step should be promoted, without overstating it as broad simulation calibration completion.
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

## 2026-03-10 - [LOCAL STAGE] Bounded `eco_bias_v1` second source and source-vs-source comparison pressure
- **Branch:** `codex/stage1-eco-bias-second-source`
- **Initiative / phase:** Stage 1 bounded second-source pressure step (`balanced_v1` vs `eco_bias_v1`, fixed seed `20260310`)
- **Summary of local stage work:** Added exactly one second canonical Phase 2 source using `eco_bias_v1` on the same fixed seed/shape/truthfulness rules as `balanced_v1`, preserved the same replay-comparable carryover-complete safety floor, and emitted one thin machine-readable source-vs-source comparison artifact that keeps source identity explicit instead of abusing baseline/current semantics.
- **Project changes in scope:**
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
- **Checks run and result (local stage):** focused Phase 2 policy-contract and trace-export pytest passed; the approved `balanced_v1` CLI run completed deterministically; the direct `.venv311` `eco_bias_v1` CLI validation completed successfully; the direct `.venv311` comparison CLI validation completed successfully and emitted `automation/reports/phase2_source_comparison_balanced_v1_vs_eco_bias_v1_seed20260310.json` with explicit left/right source identity, same seed/shape basis, preserved safety floor, and non-zero family-distribution deltas; the earlier launcher issue proved to be an environment/sandbox execution quirk rather than a product failure.
- **Risks / red flags:** This is still only one extra bounded source and one fixed seed. It creates decision pressure, but it does not answer broader representativeness by itself and must not be misread as a generic multi-source framework.
- **Why this local stage matters:** The canonical simulation lane is no longer stuck with a single truthful source and no comparison pressure; the repo can now tell whether a materially different bounded source changes the observed lane enough to justify further work.
- **Next likely step (from this local stage):** Review whether the new source-vs-source pressure is strong enough to justify promotion or whether the correct answer is still pause.

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
- **Tests/checks run and result:** `tests/simulation/test_phase2_policy_contract.py` and `tests/simulation/test_phase2_trace_export.py` passed; the approved `balanced_v1` CLI simulation run completed deterministically; the direct `.venv311` `eco_bias_v1` CLI validation completed successfully; the direct `.venv311` comparison CLI validation completed successfully and emitted a machine-readable artifact with explicit left/right source identity, same seed/shape basis, preserved safety floor, and non-zero family-distribution deltas; the earlier launcher trouble proved to be an environment/sandbox execution quirk rather than a product failure.
- **Risks / red flags:** This is still only one extra bounded source and one fixed seed. It creates decision pressure, but it does not answer broader representativeness by itself and must not be misread as broad simulation/calibration completion.
- **Why this push matters:** The canonical simulation lane is no longer stuck with a single truthful source and no comparison pressure; `master` can now test whether a materially different bounded source changes the observed lane enough to justify further work.
- **Next likely step (at this time):** Re-rank the next meaningful project from current `master` reality rather than assuming more Phase 2 expansion automatically.



