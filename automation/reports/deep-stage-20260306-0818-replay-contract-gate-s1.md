branch: deep/stage-20260306-0818-replay-contract-gate-s1
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: approve stage 1

# Stage report — Replay Contract Gate (Stage 1)

## Objective (approved)

Implement a bounded replay contract gate so canonical replay defaults to rejecting point-like inputs with deterministic reason codes, while exposing active policy and conformance counters.

## Files changed

| Path | Change |
|---|---|
| `engine/models.py` | Added explicit replay contract policy fields: `replay_contract_policy`, `replay_point_transition_enabled`, `replay_point_transition_sunset_epoch`. |
| `engine/config.py` | Added defaults + merge/coercion for replay policy fields; policy locked to `reject_point_like` for Stage 1. |
| `backend/services/runner.py` | Added point-like replay contract gate with deterministic reject reasons, transition-window enforcement, and replay contract counters/status method; gated legacy passthrough path. |
| `backend/api/routes_replay.py` | Added replay policy fields to `/replay/load` and `/replay/status`; status now includes runner contract counters when available. |
| `tools/replay_verification_assess.py` | Added policy conformance fields and reason-count map to assessment summary output. |
| `tools/schemas/replay_validation_summary.schema.json` | Extended schema required keys for replay contract policy and conformance counters. |
| `tests/unit/test_runner_replay_contract_mode.py` | Replaced passthrough-as-default tests with Stage 1 gate tests (default reject, explicit transition allow, expired transition reject). |
| `tests/unit/test_replay_verification_assess_stage1.py` | Extended schema conformance assertions for new policy/counter fields. |
| `tests/unit/test_replay_status_contract_gate.py` | New narrow status/load tests for replay contract policy observability. |

## Behavior changed

1. **Default policy is explicit and enforced**
   - Replay config now includes a contract policy surface.
   - Stage 1 policy resolves to `reject_point_like`.

2. **Point-like replay path is contract-gated**
   - If replay payloads are point-like and transition mode is not explicitly enabled, runner rejects them (no append, no broadcast).
   - Deterministic reason codes are emitted into runner counters:
     - `POINT_REPLAY_REJECTED_DEFAULT_POLICY`
     - `POINT_REPLAY_REJECTED_UNSUPPORTED_POLICY`
     - `POINT_REPLAY_REJECTED_TRANSITION_SUNSET_MISSING`
     - `POINT_REPLAY_REJECTED_TRANSITION_SUNSET_EXPIRED`

3. **Transition passthrough is explicit and sunset-bound**
   - Legacy passthrough only occurs when:
     - `replay_point_transition_enabled=True`
     - `replay_point_transition_sunset_epoch` exists and is in the future
   - Legacy passthrough path remains present but gated.

4. **Status/assessment observability added**
   - `/replay/status` now reports active contract policy fields and replay contract counters.
   - Replay assessment summary now includes policy/counter outputs and reason-count map.

5. **Raw canonical replay behavior preserved**
   - Raw replay remains on canonical normalize->compute path (`replay_mode="raw_contract"`).

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| Default config resolves explicitly to `reject_point_like` | ✅ met | `engine/models.py` default + `engine/config.py` coercion |
| Point-like replay under default policy does not append/broadcast | ✅ met | `tests/unit/test_runner_replay_contract_mode.py::test_point_replay_default_policy_rejects_without_append_or_broadcast` |
| Rejects emit deterministic frozen reason codes | ✅ met | Runner constants + counter map assertions in replay contract tests |
| Transition passthrough only when explicit + valid sunset | ✅ met | `tests/unit/test_runner_replay_contract_mode.py` transition allow + expired rejection tests |
| `/replay/status` reports active policy and counters | ✅ met | `tests/unit/test_replay_status_contract_gate.py` |
| Assessment emits policy counters + reason-count map | ✅ met | `tests/unit/test_replay_verification_assess_stage1.py` + command output on `logs/history_points.jsonl` |
| Raw canonical replay behavior unchanged | ✅ met | Assessment run on raw fixture + full pytest pass |
| No compute-path files changed | ✅ met | No `engine/compute/*` modifications in file list |

## Validation

Executed exactly as requested:

1. `python3 -m pytest -q tests/unit/test_runner_replay_contract_mode.py` → `6 passed`
2. `python3 -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` → `1 passed`
3. `python3 -m pytest -q tests/unit/test_replay_status_contract_gate.py` → `2 passed`
4. `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
   - raw contract points captured; point-like counters all zero
5. `python3 tools/replay_verification_assess.py logs/history_points.jsonl`
   - point-like inputs seen/rejected tracked; reason-count map populated with `POINT_REPLAY_REJECTED_DEFAULT_POLICY`
6. `python3 -m pytest -q` → `365 passed`

## Risks / scope pressure

- Point-like replay now rejects by default; legacy consumers depending on implicit passthrough must explicitly opt into transition mode and set sunset.
- Stage 1 uses runner counters for observability; this is bounded but does not yet include broader historical telemetry export.

No scope expansion pressure triggered beyond approved Stage 1 surfaces.

## Out-of-scope preserved

- No compute-path changes (`engine/compute/*` untouched).
- No conversion pipelines added.
- No replay migration mechanics implemented.
- No replay source discovery redesign beyond policy visibility.
- No legacy path deletion (passthrough retained, gated only).
- No Stage 2+ architecture work.
