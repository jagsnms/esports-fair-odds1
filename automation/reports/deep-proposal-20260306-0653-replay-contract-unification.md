branch: deep/proposal-20260306-0653-replay-contract-unification
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative proposal: Replay Contract Unification (eliminate non-contract replay semantics)

## Initiative title

Replay Contract Unification: remove semantic drift between replay ingestion modes by converging all replay processing onto the canonical normalize -> bounds -> rails -> resolve pipeline.

## Why it outranks other major issues

This issue is selected as the highest-value unresolved major issue because current code still permits replay points that bypass canonical compute:

- `backend/services/runner.py` contains `_tick_replay_point_passthrough(...)` and routes replay mode `"point"` to passthrough (`_tick_replay`, point branch), which does not execute normalize/bounds/rails/resolve.
- `tools/replay_verification_assess.py` explicitly tracks `point_passthrough_points` separately from `raw_contract_points`, proving mixed semantic modes are first-class in current architecture.
- Bible Chapter 6 defines a canonical calculation flow, and Chapter 9.2.3 states replay validation is the primary source of truth. A replay path that bypasses canonical compute weakens that source of truth and can mask contract/invariant behavior.

Alternatives considered:

1. Replay + simulation validation architecture (banked): remains a deferred alternative per `automation/BANKED_INITIATIVES.md`. It can be reopened later, but this run prioritizes replay semantic correctness first because simulation quality depends on contract-faithful replay behavior.
2. Additional PHAT movement/calibration redesign: recent Stage 1-4C reports show bounded progress and no active blocker evidence exceeding this replay contract gap right now.
3. Automation self-infrastructure refinement: deprioritized by policy and not a core model/architecture blocker.

## Why it exceeds bounded-fix scope

This is not a single-file maintenance patch. It requires cross-module architectural work and migration policy:

- replay ingestion contract decisions (accept/convert/reject point-like payloads),
- runner behavior changes across replay path semantics and diagnostics,
- replay validation summary schema/version strategy,
- fixture/test migration and compatibility gates,
- likely staged deprecation timeline for legacy point replay inputs.

That breadth makes it a major initiative rather than a bounded maintenance fix.

## Affected modules/files

| Module / path | Role in initiative |
|---------------|--------------------|
| `backend/services/runner.py` | Replay mode dispatch and current point passthrough behavior; primary architecture change site. |
| `engine/replay/bo3_jsonl.py` | Replay load/iteration contract and payload classification boundaries. |
| `tools/replay_verification_assess.py` | Validation summary logic and contract-mode metrics/reporting. |
| `tools/schemas/replay_validation_summary.schema.json` | Schema evolution for unified replay contract metrics. |
| `tests/unit/test_runner_replay_contract_mode.py` | Contract tests for replay mode semantics and migration behavior. |
| `tests/unit/test_replay_verification_assess_stage1.py` | Determinism/schema assertions that must be updated for new contract version. |
| `tools/fixtures/*.jsonl` | Canonical replay fixtures for migration and compatibility checkpoints. |

## Proposed stages

1. **Stage 1 - Contract policy lock and baseline matrix (planning + instrumentation)**
   - Freeze explicit replay contract policy (convert point-like to canonical frame vs hard-reject legacy points).
   - Produce a baseline matrix over existing fixtures/logs: counts of raw-contract vs point-passthrough, diagnostics coverage, and invariant visibility.
   - No broad behavior migration yet.

2. **Stage 2 - Replay ingestion unification adapter**
   - Implement one canonical replay ingestion path in runner.
   - Replace direct passthrough with adapter behavior per approved policy (normalize to canonical frame and run compute, or reject with explicit reason codes).
   - Keep compatibility guardrails behind clearly documented mode flags if transitional operation is required.

3. **Stage 3 - Validation artifact/schema upgrade**
   - Version replay summary schema (v2) to encode unified contract expectations.
   - Tighten success gates: expected contract diagnostics coverage and explicit handling of non-conformant payloads.
   - Ensure deterministic outputs remain stable.

4. **Stage 4 - Test and fixture migration**
   - Update contract tests to assert one replay semantic model.
   - Add fixtures covering legacy point-like inputs and expected converted/rejected outcomes.
   - Preserve regression coverage for raw BO3 snapshots and inter-map-break handling.

5. **Stage 5 - Legacy mode deprecation and cleanup**
   - Remove or fully isolate obsolete passthrough path after migration evidence is stable.
   - Finalize documentation and operator guidance for replay input contract.

## Validation checkpoints

- **Checkpoint A (before Stage 2):** baseline evidence report includes replay-mode distribution and diagnostics coverage.
- **Checkpoint B (Stage 2):** targeted tests for replay dispatch/adapter behavior pass, including explicit reason-code assertions for non-conformant inputs.
- **Checkpoint C (Stage 3):** replay summary schema v2 validates and remains deterministic across repeated runs on fixed fixtures.
- **Checkpoint D (Stage 4):** replay and invariant tests pass for raw fixtures, migrated legacy fixtures, and mixed-format rejection/conversion scenarios.
- **Checkpoint E (Stage 5):** no remaining production path emits `point_passthrough` semantics without explicit quarantine mode.

## Risks

- **Migration risk:** historical point-like replay artifacts may not contain enough state to reconstruct canonical frames, requiring policy decisions on rejection vs partial conversion.
- **Regression risk:** replay behavior changes can affect downstream diagnostics and historical comparability.
- **Scope risk:** runner replay unification may interact with inter-map-break and session logic if boundaries are not kept strict.
- **Operational risk:** environment reproducibility currently depends on missing runtime deps in this sandbox (`pytest`, `fastapi` unavailable), so approved implementation stages must include dependency-validated execution context.

## Recommended branch plan

- Keep this proposal on: `deep/proposal-20260306-0653-replay-contract-unification`.
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-replay-contract-unification-s1`
  - `deep/stage-YYYYMMDD-HHMM-replay-contract-unification-s2`
  - etc.
- Rebase each stage from `agent-initiative-base` after prior stage promotion decision; do not merge stages autonomously.
- Require a stage report per implementation branch with explicit evidence and stop reason.

## Recommendation

- [x] **Approve planning only** — accept proposal; lock policy and evidence gates before code migration
- [ ] **Approve stage 1** — approve first implementation stage
- [ ] **Defer** — do not approve; revisit later
