# Branch History - `master`

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

## 2026-03-10 - [LOCAL STAGE] Canonical simulation trace export with per-point prediction/outcome labels
- **Branch:** `codex/phase2-trace-export-stage1`
- **Initiative / phase:** Bounded canonical trace-export/source-contract step (`balanced_v1` only)
- **Summary of local stage work:** Added one deterministic machine-readable trace-export path for the landed canonical `balanced_v1` simulation lane so prediction points can be paired only with truthful runner-emitted `round_result` labels, then removed the duplicate canonical execution introduced in the initial implementation by reusing the existing canonical assessment pass.
- **Project commits:**
  - `897f97400e21e6099ba4abc887d9d55eaca0c9cb` `Add canonical Phase 2 trace export contract`
  - `374485d9df58f12213076ffb4716cf0729f061c6` `Reuse assessment pass for Phase 2 trace export`
- **Key files/subsystems touched:**
  - `engine/simulation/phase2.py`
  - `tools/replay_verification_assess.py`
  - `tests/simulation/test_phase2_trace_export.py`
  - `docs/branch_history_master.md`
  - `docs/current_status_master.md`
- **Checks run and result (local stage):** approved validations now pass: `tests/unit/test_run_replay_simulation_validation_pilot.py`, `tests/simulation/test_phase2_policy_contract.py`, and `tests/simulation/test_phase2_trace_export.py`; repeated `tools/simulate_phase2.py --seed 20260310` runs emitted identical machine-readable output.
- **Risks / red flags:** This is still one bounded `balanced_v1` slice only. The export is a source-contract step, not a calibration lane, and unlabeled final-round prediction points are excluded rather than given pseudo-labels. The earlier pytest launcher issue was an environment/sandbox quirk, not a product failure.
- **Why this local stage matters:** It opens the truthful raw source contract that later downstream evidence work was missing, without faking calibration-ready outputs or broadening simulation semantics.
- **Next likely step (from this local stage):** Decide whether this bounded source-contract step should be promoted, without overstating it as calibration-lane work.

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
