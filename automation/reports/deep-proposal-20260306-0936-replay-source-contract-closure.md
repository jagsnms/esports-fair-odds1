branch: deep/proposal-20260306-0936-replay-source-contract-closure
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative proposal

## Initiative title

Replay Source Contract Closure: align replay source discovery, runner policy gates, and validation artifacts so canonical replay surfaces only contract-valid inputs.

## Why it outranks other major issues

This is the highest-value unresolved major issue in current repo evidence because replay contract policy and replay source surfacing are now out of sync:

- Canonical suite is currently healthy (`python3 -m pytest -q` -> `365 passed`), so no higher-ranked structural/test-failure blocker is active.
- Canonical raw replay fixtures are clean under current policy:
  - `tools/fixtures/replay_multimatch_small_v1.jsonl` -> `raw_contract_points=6`, `point_like_inputs_rejected=0`.
  - `tools/fixtures/raw_replay_sample.jsonl` -> `raw_contract_points=3`, `point_like_inputs_rejected=0`.
- But default policy hard-rejects point-like replay inputs:
  - `python3 tools/replay_verification_assess.py logs/history_points.jsonl` -> `point_like_inputs_seen=46`, `point_like_inputs_rejected=46`, reason `POINT_REPLAY_REJECTED_DEFAULT_POLICY`.
- Replay source discovery still advertises point logs as selectable replay sources (`backend/api/routes_replay.py` discovers `history_points*.jsonl` with `kind="point"`).

That mismatch is architectural, cross-module, and directly affects whether replay validation workflows are contract-faithful (Bible Chapter 6 pipeline + Chapter 9 replay primacy).

Deferred alternative (banked): **Replay + Simulation Validation Architecture** remains banked in `automation/BANKED_INITIATIVES.md` and is not re-proposed here because reopen conditions are not clearly met by current evidence.

## Why it exceeds bounded-fix scope

This is not a one-file maintenance fix. Closing the gap requires coordinated design and staged migration across:

- replay source discovery and selection contracts,
- runner policy/gate behavior and transition handling,
- replay API status/load semantics,
- replay validation artifact schema and pass/fail gates,
- fixtures/tests that currently encode both canonical and legacy semantics.

Because it spans runner + API + tooling + tests + migration policy, this exceeds bounded-fix scope and belongs in the deep initiative lane.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/api/routes_replay.py` | Source discovery currently surfaces both `raw` and `point` inputs; load/status policy surface must be made contract-explicit. |
| `backend/services/runner.py` | Enforces replay policy gate and still contains transition passthrough path (`_tick_replay_point_passthrough`) with policy counters. |
| `engine/models.py` | Holds replay policy fields (`replay_contract_policy`, transition flags) that define runtime contract behavior. |
| `engine/config.py` | Coercion/default logic for replay policy settings and any staged deprecation flags. |
| `tools/replay_verification_assess.py` | Emits replay-mode and policy counters; must become a strict contract-conformance reporter for closure stages. |
| `tools/schemas/replay_validation_summary.schema.json` | Needs schema evolution for explicit contract-conformance verdict fields/checks. |
| `tests/unit/test_runner_replay_contract_mode.py` | Core tests for reject/transition behavior and deterministic reason-code contract. |
| `tests/unit/test_replay_status_contract_gate.py` | API observability tests for replay policy exposure. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Validation summary schema/determinism tests to extend for closure gates. |
| `logs/history_points.jsonl` and `logs/bo3_raw_match_*.jsonl` | Practical evidence classes for point-like legacy vs canonical raw replay sources. |

## Proposed stages

1. **Stage 1 — Contract/source matrix freeze (planning+tests only)**
   - Freeze canonical policy matrix for each source kind (`raw`, `point`, mixed) with explicit expected load/accept/reject behavior.
   - Lock deterministic reason-code taxonomy and conformance counters as contractual outputs.

2. **Stage 2 — Replay source discovery alignment**
   - Align `/replay/sources` and `/replay/load` behavior with active policy:
     - either hide unsupported point sources by default, or
     - surface them as unsupported/legacy with explicit non-canonical status and load gate outcomes.
   - Prevent UI/operator ambiguity where selectable sources are guaranteed to reject under default policy.

3. **Stage 3 — Runner transition policy hardening**
   - Keep default reject behavior as canonical path.
   - Make transition passthrough strictly time-boxed and explicitly non-canonical in status/reporting.
   - Define deterministic sunset handling and stop conditions for transition mode removal.

4. **Stage 4 — Validation artifact contract upgrade**
   - Extend replay summary schema to include explicit conformance verdicts (not only counters).
   - Add gate checks so canonical validation cannot pass when non-canonical paths are used without explicit transition mode.

5. **Stage 5 — Deprecation closure**
   - Remove or fully disable legacy point passthrough path after approved migration window.
   - Preserve raw-contract replay behavior and diagnostics parity.

## Validation checkpoints

- **Checkpoint A (pre-implementation baseline):**
  - `python3 -m pytest -q` remains green.
  - Replay assessment outputs for:
    - `tools/fixtures/replay_multimatch_small_v1.jsonl`
    - `tools/fixtures/raw_replay_sample.jsonl`
    - `logs/history_points.jsonl`
  - Baseline policy matrix captured (seen/rejected/transition counts + reason-code map).

- **Checkpoint B (post Stage 2):**
  - Replay source discovery/load tests assert contract-explicit behavior for point sources.
  - No "silent selectable but always rejected" source state remains.

- **Checkpoint C (post Stage 3):**
  - Runner tests prove deterministic reject-vs-transition behavior and sunset enforcement.
  - Status endpoint exposes policy and counters unambiguously.

- **Checkpoint D (post Stage 4):**
  - Replay summary schema validation passes with conformance verdict fields.
  - Assessment fails/flags non-canonical usage in canonical validation mode.

- **Checkpoint E (post Stage 5):**
  - Full suite remains green.
  - Replay contract path is single-semantic by default (raw canonical compute).

## Risks

- **Compatibility risk:** teams relying on historical point logs may lose ad-hoc replay workflows unless migration guidance is staged.
- **Operational risk:** changing source discoverability can impact UI/consumer expectations and requires explicit messaging.
- **Policy drift risk:** temporary transition mode can become permanent unless hard sunset gates are test-enforced.
- **Validation risk:** counters without hard verdict gates can still permit ambiguous success claims.
- **Scope risk:** closure work can drift into broad replay redesign unless stage boundaries are enforced.

## Recommended branch plan

- Proposal branch (this run): `deep/proposal-20260306-0936-replay-source-contract-closure`.
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-replay-source-contract-closure-s1`
  - `deep/stage-YYYYMMDD-HHMM-replay-source-contract-closure-s2`
  - `deep/stage-YYYYMMDD-HHMM-replay-source-contract-closure-s3`
  - `deep/stage-YYYYMMDD-HHMM-replay-source-contract-closure-s4`
  - `deep/stage-YYYYMMDD-HHMM-replay-source-contract-closure-s5`
- Rebase each stage from updated `agent-initiative-base` after prior stage review outcome.
- No self-merge; promotion is human-gated.

## Recommendation

- [x] **Approve planning only**
- [ ] **Approve stage 1**
- [ ] **Defer**
