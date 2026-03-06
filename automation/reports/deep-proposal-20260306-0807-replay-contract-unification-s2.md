branch: deep/proposal-20260306-0807-replay-contract-unification-s2
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Replay Contract Unification Stage 2: eliminate semantic split between raw-contract replay and point-passthrough replay.

## Why it outranks other major issues

Current evidence shows canonical tests are green (`362 passed`) and bounded replay fixtures are clean under raw-contract mode, but the architecture still carries two replay semantics:

- canonical replay (`replay_mode="raw_contract"`) through normalize -> reduce -> bounds/rails -> resolve
- non-canonical replay (`replay_mode="point_passthrough"`, `replay_contract_class="non_canonical_point"`) that bypasses canonical compute

This split is still active in runtime code and test contract:

- `backend/services/runner.py` has `_tick_replay_point_passthrough(...)`
- `tests/unit/test_runner_replay_contract_mode.py` explicitly asserts passthrough + quarantine tags
- `backend/api/routes_replay.py` still discovers and advertises point replay sources (`history_points*.jsonl`)

Compared with alternatives, this is the highest-value unresolved major issue because it directly affects whether replay validation evidence is contract-faithful (Bible Chapter 6/9 alignment). A replay architecture that mixes canonical and non-canonical semantics can hide true model behavior behind format-dependent execution paths.

Deferred alternative (banked): **Replay + Simulation Validation Architecture** remains banked in `automation/BANKED_INITIATIVES.md`; reopen conditions were not triggered by this run, so it is not re-proposed here.

## Why it exceeds bounded-fix scope

This cannot be closed safely as a one-file or one-test maintenance fix. It requires coordinated policy and implementation changes across:

- replay ingestion/detection policy
- runner execution paths
- replay API source discovery/selection behavior
- config schema and controls
- fixtures, replay assessment outputs, and tests

It also requires a transition strategy to avoid abruptly breaking legacy point-based replay consumers.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Primary replay execution split (`_tick_replay` vs `_tick_replay_point_passthrough`) and quarantine tagging behavior. |
| `backend/api/routes_replay.py` | Replay source discovery currently publishes both raw and point sources; policy enforcement surface. |
| `engine/models.py` | Runtime config contract may need explicit replay contract mode fields. |
| `engine/config.py` | Config merge defaults/allowlist for any new replay contract mode and deprecation controls. |
| `tools/replay_verification_assess.py` | Validation metrics/reporting for replay mode usage and non-canonical counts. |
| `tools/schemas/replay_validation_summary.schema.json` | Schema updates for stricter replay contract reporting. |
| `tests/unit/test_runner_replay_contract_mode.py` | Convert from "passthrough allowed" assertions to approved Stage 2 policy assertions. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Extend to require policy-conformant replay-mode distributions. |
| `tools/fixtures/*.jsonl` | Fixture classes used to validate migration policy and backward-compat boundaries. |

## Proposed stages

1. **Stage 1 (planning/contract gate): finalize replay policy decision**
   - Decide one policy for point-like replay inputs:
     - convert point-like inputs into canonical frame path, or
     - reject point-like inputs with explicit reason codes.
   - Lock acceptance criteria and migration timeline.

2. **Stage 2 (bounded runtime contract wiring)**
   - Add explicit config-level replay contract mode and enforce it in runner.
   - Keep compatibility telemetry, but prevent implicit silent passthrough behavior.
   - Ensure API/status surfaces active contract mode.

3. **Stage 3 (migration mechanics)**
   - Implement chosen convert-or-reject behavior end-to-end.
   - Update replay source discovery and load semantics to avoid ambiguous source selection.
   - Update replay assessment and schema so non-canonical counts are tracked as policy exceptions only.

4. **Stage 4 (test/fixture contract hardening)**
   - Replace tests that currently validate passthrough-as-normal.
   - Add deterministic coverage for mixed-input files and explicit failure/convert outcomes.
   - Add regression checks that canonical replay outputs remain stable.

5. **Stage 5 (deprecation closure)**
   - Remove or fully disable legacy passthrough path after migration window.
   - Finalize documentation/reporting contracts around replay validation provenance.

## Validation checkpoints

- **Checkpoint A (pre-implementation baseline):**
  - `python3 -m pytest -q` remains green.
  - Replay assessment on canonical fixtures remains all `raw_contract`.
  - Baseline non-canonical-path inventory documented.

- **Checkpoint B (Stage 2 runtime wiring):**
  - Unit tests prove contract mode is explicit and observable in runtime debug/status.
  - No implicit fallback into passthrough without explicit mode.

- **Checkpoint C (Stage 3 migration behavior):**
  - Mixed replay inputs produce deterministic, policy-compliant outcomes (convert or reject).
  - Replay assessment outputs include measurable policy counters and zero ambiguous mode points.

- **Checkpoint D (Stage 4 hardening):**
  - Contract tests for replay mode policy pass.
  - Existing compute/invariant suites remain unchanged and passing.

- **Checkpoint E (Stage 5 closure):**
  - Legacy passthrough path removed/disabled with passing full suite and replay regression evidence.

## Risks

- **Compatibility risk:** legacy workflows using point replay could break if rejection is immediate.
- **Semantic drift risk:** conversion from point-like records to canonical frames may inject assumptions if mapping is lossy.
- **Operational risk:** mixed raw/point files may produce unexpected runtime behavior without strict deterministic handling.
- **Validation risk:** replay metrics can appear healthy while hiding non-canonical execution unless policy counters are mandatory.
- **Rollout risk:** API/UI replay source behavior changes may require coordinated consumer updates.

## Recommended branch plan

- Current proposal branch (this run): `deep/proposal-20260306-0807-replay-contract-unification-s2`
- If approved, execute one stage per branch:
  - `deep/stage-<timestamp>-replay-contract-unification-s2-s1`
  - `deep/stage-<timestamp>-replay-contract-unification-s2-s2`
  - `deep/stage-<timestamp>-replay-contract-unification-s2-s3`
  - etc.
- Keep stage branches narrowly scoped and independently reviewable; no cross-stage bundling.
- Promote only through human review; no self-merge.

## Recommendation

- [x] **Approve planning only**
- [ ] **Approve stage 1**
- [ ] **Defer**
