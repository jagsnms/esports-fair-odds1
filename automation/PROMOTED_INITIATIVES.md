# Promoted initiatives registry

Canonical registry of initiatives/stages that are already promoted into shared truth.
Proposal and planning runs must check this file before proposing new work.

## A) Rail / replay / carryover line

- **Stage 1 rail v2 strict semantic switch**
  - status: promoted/shared truth
  - evidence hint: `origin/deep/stage-20260306-2055-rail-carryover-semantic-switch-s1`
- **Approved carryover scenario suite**
  - status: promoted/shared truth
  - evidence hint: carryover scenario coverage in promoted deep carryover line
- **Replay v2 activation backbone**
  - status: promoted/shared truth
  - evidence hint: `origin/deep/stage-20260306-2203-rail-v2-activation-s1`
- **Replay carryover provenance/completeness**
  - status: promoted/shared truth
  - evidence hint: `automation/reports/deep-stage-20260307-0608-replay-carryover-provenance-s1.md`
- **Replay carryover baseline artifact**
  - status: promoted/shared truth
  - evidence hint: promoted deep carryover baseline stage/report set
- **Stage 3A redistribution revise**
  - status: promoted/shared truth
  - evidence hint: `agent-base` / `agent-initiative-base` at `eab12368f8a1c35567cf6a33fb1a711fabe2c108`

## B) Timer-pressure line

- **Timer-pressure Stage 1 compute contract**
  - status: promoted/shared truth
  - contract points:
    - pre-plant timer favors CT
    - post-plant timer favors T
    - hard post-plant CT impossible-defuse boundary
    - diagnostics emitted
    - q-path only
    - no rail/bounds redesign
  - evidence hint: `automation/reports/deep-stage-20260306-2245-timer-pressure-s1-compute.md`

## C) Synthetic subsystem

- **Stage 1 metadata/deterministic selection**
  - status: promoted/shared truth
  - evidence hint: promoted Stage 1 harvested line
- **Stage 2A execute/retake planted-state shaping**
  - status: promoted/shared truth
  - evidence hint: promoted Stage 2A line
- **Stage 2B clutch/eco_force shaping**
  - status: promoted/shared truth
  - evidence hint: promoted Stage 2B line
- **Stage 3A distribution control + evidence gate**
  - status: promoted/shared truth
  - evidence hint: promoted Stage 3A line including redistribution revise

## D) Canonical simulation / bounded evidence line

- **Seeded simulation Phase 1 contract**
  - status: promoted/shared truth
  - evidence hint: `engine/simulation/phase1.py`, `tools/simulate_phase1.py`
- **Bounded Phase 2 canonical simulation contract (`balanced_v1`)**
  - status: promoted/shared truth
  - evidence hint: `engine/simulation/phase2.py`, `tests/simulation/test_phase2_policy_contract.py`
- **Bounded Phase 2 V2 activation on landed `balanced_v1` slice**
  - status: promoted/shared truth
  - evidence hint: `engine/simulation/phase2.py` with replay-assessed V2 activation on the canonical `balanced_v1` slice
- **Canonical Phase 2 trace export (`balanced_v1`)**
  - status: promoted/shared truth
  - evidence hint: `tests/simulation/test_phase2_trace_export.py`, canonical `round_result` trace export in `engine/simulation/phase2.py`
- **Bounded canonical simulation calibration evidence path (`balanced_v1`, seed `20260310`)**
  - status: promoted/shared truth
  - evidence hint: `tools/export_calibration_reliability_evidence.py`, `automation/reports/calibration_reliability_evidence_export_manifest_v1.json`
- **Bounded `eco_bias_v1` second source + source-vs-source comparison artifact**
  - status: promoted/shared truth
  - evidence hint: `tools/compare_phase2_sources.py`, `automation/reports/phase2_source_comparison_balanced_v1_vs_eco_bias_v1_seed20260310.json`
- **Replay-anchored two-source decision contract (`balanced_v1` vs `eco_bias_v1`)**
  - status: promoted/shared truth
  - evidence hint: `tools/run_replay_multisource_decision.py`, `tests/unit/test_run_replay_multisource_decision.py`, `automation/reports/replay_multisource_decision_balanced_v1_vs_eco_bias_v1_seed20260310.json`
