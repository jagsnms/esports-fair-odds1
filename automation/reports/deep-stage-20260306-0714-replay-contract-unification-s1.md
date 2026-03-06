branch: deep/stage-20260306-0714-replay-contract-unification-s1
base_branch: agent-initiative-base
lane: deep
run_type: implementation
status: implemented
recommendation: approve stage 1

# Stage report — Replay Contract Unification (Stage 1)

## Approved Stage 1 scope (implemented only)

- one baseline replay-mode usage matrix over known replay inputs/fixtures/logs
- one explicit replay contract policy output evaluating:
  - convert point-like replay inputs
  - reject point-like replay inputs
  - quarantine/tag non-canonical point replay inputs
- one bounded runtime tagging/quarantine mechanism for non-canonical point replay outputs
- one focused test proving tagging/quarantine behavior

Out-of-scope items were not implemented (no full migration, no runner rewrite, no schema migration, no fixture ecosystem expansion).

## Pre-coding contract restatement (what this stage touched)

### Exact files touched

1. `backend/services/runner.py`
2. `tools/replay_verification_assess.py`
3. `tests/unit/test_runner_replay_contract_mode.py`
4. `automation/reports/deep-stage-20260306-0714-replay-contract-unification-s1.md`

### Exact allowed changes applied

- Added minimal replay point-output quarantine tags in runner (legacy point path retained).
- Added baseline replay-mode matrix fields to replay assessment output.
- Added focused test assertions proving quarantine/tagging fields on point replay output.
- Added this stage report.

### Exact forbidden changes honored (not done)

- No changes in `engine/compute/**`
- No changes in `engine/diagnostics/**`
- No conversion of point-like replay into canonical frames
- No deletion of legacy point replay path
- No broad replay architecture cleanup
- No calibration changes
- No simulation work

### Quarantine/tagging behavior implemented (one paragraph)

When replay input is detected as legacy point-like and processed by `_tick_replay_point_passthrough`, runner now explicitly classifies output as non-canonical and quarantine-tagged by adding debug fields: `replay_contract_class="non_canonical_point"`, `replay_quarantine_status="quarantined_tagged"`, and `replay_quarantine_reason="point_payload_bypasses_canonical_pipeline"` (while keeping `replay_mode="point_passthrough"`). The point is still appended as before for compatibility, and broadcast behavior remains bounded/optional via existing config access (`replay_emit_quarantined_points`, defaulting to emit).

## Baseline replay-mode usage matrix (known inputs/fixtures/logs)

Evidence commands:

- `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`
- runtime sample for non-canonical point path (mocked point replay payload through runner)
- `logs/*.jsonl` discovery check

| Input source | Exists | raw_contract | point_passthrough | non_canonical_point | unknown | Notes |
|---|---:|---:|---:|---:|---:|---|
| `tools/fixtures/raw_replay_sample.jsonl` | yes | 3 | 0 | 0 | 0 | Raw BO3-shaped snapshots; contract diagnostics present on all points. |
| `tools/fixtures/replay_multimatch_small_v1.jsonl` | yes | 6 | 0 | 0 | 0 | Raw-contract multimatch fixture. |
| `logs/*.jsonl` | no | n/a | n/a | n/a | n/a | No replay log jsonl files present in workspace at run time. |
| Runtime point-path sample (mock entry) | synthetic | 0 | 1 | 1 | 0 | Debug output confirms quarantine tags for non-canonical point replay output. |

## Policy output (Stage 1 evaluation)

### Option A — Convert point-like replay inputs to canonical frames (not implemented in Stage 1)

- Pros: one semantic replay model; strongest Chapter 6/9 alignment.
- Cons: requires mapping assumptions and broader migration risk; exceeds Stage 1 bounds.

### Option B — Reject point-like replay inputs (not implemented in Stage 1)

- Pros: strict contract purity.
- Cons: breaks legacy replay artifacts immediately without transition path.

### Option C — Quarantine/tag non-canonical point replay inputs (implemented in Stage 1)

- Pros: bounded, reversible, preserves compatibility while making non-canonical behavior explicit and measurable.
- Cons: legacy non-canonical path still exists until later approved stages.

**Stage 1 recommendation:** keep Option C as the active interim policy and defer convert-vs-reject final decision to Stage 2 planning gate.

## Runtime tagging/quarantine behavior implemented (exact)

- Location: `backend/services/runner.py` in `_tick_replay_point_passthrough`.
- Added debug contract fields on point outputs:
  - `replay_mode = "point_passthrough"`
  - `replay_contract_class = "non_canonical_point"`
  - `replay_quarantine_status = "quarantined_tagged"`
  - `replay_quarantine_reason = "point_payload_bypasses_canonical_pipeline"`
- Compatibility preserved:
  - point replay still appends output to history
  - no conversion to canonical frame compute path yet
  - no deletion of legacy point path yet

## Focused test proof

- Updated `tests/unit/test_runner_replay_contract_mode.py`:
  - point replay test now asserts the three quarantine fields above in runner-produced debug.
- Validation run:
  - `python3 -m pytest -q tests/unit/test_runner_replay_contract_mode.py` -> `5 passed`

## Additional validation evidence

- `python3 tools/replay_verification_assess.py tools/fixtures/raw_replay_sample.jsonl`
  - `raw_contract_points=3`, `point_passthrough_points=0`, `non_canonical_point_points=0`
- `python3 tools/replay_verification_assess.py tools/fixtures/replay_multimatch_small_v1.jsonl`
  - `raw_contract_points=6`, `point_passthrough_points=0`, `non_canonical_point_points=0`
- runtime point-path sample output:
  - `{"replay_contract_class":"non_canonical_point","replay_mode":"point_passthrough","replay_quarantine_reason":"point_payload_bypasses_canonical_pipeline","replay_quarantine_status":"quarantined_tagged"}`

## Files changed

| Path | Change |
|---|---|
| `backend/services/runner.py` | Added bounded non-canonical point replay quarantine tags in debug output for point passthrough path. |
| `tools/replay_verification_assess.py` | Added replay-mode usage matrix and quarantine count fields for baseline reporting. |
| `tests/unit/test_runner_replay_contract_mode.py` | Added focused assertions proving quarantine tagging behavior on point replay output. |
| `automation/reports/deep-stage-20260306-0714-replay-contract-unification-s1.md` | This stage report. |

## Stop reason

Stopped at approved Stage 1 boundary after baseline matrix, policy evaluation, minimal runtime tagging/quarantine mechanism, and focused test evidence were completed. No migration/rewrite work was started.
