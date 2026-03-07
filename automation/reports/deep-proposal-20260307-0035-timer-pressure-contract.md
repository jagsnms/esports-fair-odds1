branch: deep/proposal-20260307-0035-timer-pressure-contract
base_branch: agent-initiative-base
lane: deep
run_type: proposal
status: proposed
recommendation: approve planning only

# Initiative title

Timer Pressure Contract Alignment (Bible Ch.5/6/7)

## Why it outranks other major issues

1. **Direct Bible mismatch in core compute behavior (highest priority)**  
   The Bible requires timer directionality and a post-plant hard boundary:
   - pre-plant clock decay should favor CT,
   - post-plant clock decay should favor T,
   - when `time_remaining < defuse_time`, CT round-win probability must be 0.  
   Current compute paths do not enforce this contract:
   - `engine/compute/q_intra_cs2.py` explicitly documents no bomb direction bias and no timer directional term.
   - `engine/compute/midround_v2_cs2.py` uses time only as symmetric urgency scaling, not side-directional timer pressure with hard boundary semantics.
   - `engine/diagnostics/invariants.py` has no timer-direction or defuse-boundary diagnostic checks.

2. **This is a Bible-facing model-behavior issue, not just observability debt**  
   Recent rail/replay deep stages improved rail-input gating and replay contract signaling, but those do not correct timer direction semantics in live q/PHAT behavior.

3. **It outranks current alternatives under authority order**
   - **PHAT/rails:** recent deep work already landed rail semantic switch + activation backbone; no current evidence of structural rail breakage in canonical reports.
   - **Calibration:** calibration on top of wrong timer direction semantics risks fitting incorrect behavior.
   - **Replay/simulation architecture:** explicitly banked in `automation/BANKED_INITIATIVES.md`; reopen conditions are not the immediate blocker here because the model contract itself is currently mismatched.

4. **Deferred alternative (banked)**  
   Replay + Simulation Validation Architecture remains relevant but deferred per banked policy; keep deferred unless reopen condition is met.

## Bible progression justification

### Direct Bible mismatch addressed

- Bible Ch.5 and Ch.6 define timer directionality and post-plant boundary behavior; Ch.7 lists timer directionality as a behavioral invariant.
- Current codebase does not implement explicit side-directional timer pressure (pre-plant CT push, post-plant T push) or the hard post-plant CT=0 boundary when defuse is impossible.

### Specific next Bible-facing step unlocked

This initiative unlocks **trustworthy replay and calibration evaluation of midround behavior** by making timer semantics Bible-correct first.  
After Stage 1/2 alignment, the next Bible-facing step is a replay validation pass that measures timer-direction invariant rates on raw-contract replay samples and contract diagnostics.

### Why this outranks PHAT / rails / calibration / replay right now

- **PHAT movement:** movement diagnostics exist; timer semantics feeding q remain structurally under-specified/misaligned.
- **Rails:** rail carryover stages already progressed; timer pressure is now the largest remaining direct behavior mismatch in round-resolution logic.
- **Calibration:** tuning before timer contract alignment would optimize around incorrect timer behavior.
- **Replay architecture:** important, but validating a model with known timer contract mismatch produces untrustworthy validation signal.

### Why this is not subsystem drift, momentum bias, or safe local cleanup

This proposal intentionally shifts focus away from recent rail/replay momentum and targets the central round-probability behavior contract. It is cross-module and Bible-mandated, not a local continuation of prior rail observability work.

### What remains blocked or unreliable if skipped

- Replay validation cannot claim Bible-aligned timer behavior.
- Calibration metrics can improve while preserving wrong timer direction dynamics.
- Contract diagnostics remain unable to detect key timer-direction failures.
- Post-plant impossible-defuse states risk semantically invalid q/PHAT outputs.

## Why it exceeds bounded-fix scope

This is not a single-file patch. It requires coordinated architecture changes across:
- feature extraction semantics (timer/bomb/side state),
- q computation and movement coupling implications,
- diagnostics/invariant contracts,
- replay validation assertions and scenario tests,
- staged rollout safeguards to avoid broad regressions.

That makes it a multi-stage, cross-module initiative appropriate for the deep lane.

## Affected modules/files

| Module / path | Role in initiative |
|---|---|
| `engine/compute/midround_v2_cs2.py` | Implement side-directional timer pressure and post-plant boundary semantics in active q path. |
| `engine/compute/q_intra_cs2.py` | Align fallback/debug q semantics with timer contract (or explicitly scope/deprecate to avoid conflicting behavior). |
| `engine/compute/resolve.py` | Ensure contract diagnostics and explain payload expose timer-direction contract fields consistently. |
| `engine/diagnostics/invariants.py` | Add timer-direction and post-plant impossible-defuse diagnostic checks (testing-mode behavioral diagnostics). |
| `engine/models.py` | Confirm/extend canonical fields needed for defuse-boundary evaluation (without changing model identity). |
| `tests/unit/test_midround_v2_cs2.py` | Add explicit pre-plant vs post-plant timer-direction tests and boundary cases. |
| `tests/unit/test_q_intra_cs2.py` | Add/adjust timer-direction contract tests for fallback/debug path semantics. |
| `tests/unit/test_invariants_contract_diagnostics.py` | Add timer-direction and impossible-defuse diagnostic assertions. |
| `tools/replay_verification_assess.py` | Add timer-direction contract summary counters for replay evidence packs. |
| `tests/unit/test_replay_verification_assess_stage1.py` (or successor replay contract tests) | Assert deterministic timer contract counters in replay summaries. |

## Proposed stages

1. **Stage 1 — Contract spec + compute gate introduction**
   - Define explicit timer-direction contract (pre-plant CT push, post-plant T push, impossible-defuse boundary).
   - Implement behind bounded policy/gate to preserve controlled rollout.

2. **Stage 2 — Diagnostics + invariant instrumentation**
   - Extend contract diagnostics with timer-direction reason codes and impossible-defuse checks.
   - Emit deterministic counters for replay assessment and CI visibility.

3. **Stage 3 — Scenario and unit contract hardening**
   - Add deterministic unit/scenario tests for:
     - pre-plant timer progression direction,
     - post-plant timer progression direction,
     - hard boundary (`time_remaining < defuse_time` => CT=0 behavior in q path).

4. **Stage 4 — Replay validation checkpoint**
   - Run replay assessment on canonical raw fixtures and capture timer-direction diagnostics.
   - Compare violation-rate baseline vs post-change.

5. **Stage 5 — Calibration-readiness gate**
   - Confirm no structural invariant regressions and acceptable behavioral diagnostics trend.
   - Only then authorize timer-sensitive calibration work.

## Validation checkpoints

- **Checkpoint A (Stage 1):** deterministic unit tests for timer-direction sign behavior pass.
- **Checkpoint B (Stage 2):** contract diagnostics include timer-direction/boundary fields with deterministic reason codes.
- **Checkpoint C (Stage 3):** new test matrix covers pre-plant/post-plant/boundary edge states and remains green.
- **Checkpoint D (Stage 4):** replay assessment artifact includes timer-direction counters and shows expected directional behavior on controlled fixtures.
- **Checkpoint E (Stage 5):** no new structural violations; behavioral diagnostics trend is stable or improved.

## Risks

1. **Over-correction risk:** timer term may dominate combat/loadout if not bounded.
2. **State ambiguity risk:** missing/ambiguous side or plant metadata can produce false directional signals.
3. **Replay sparsity risk:** existing fixtures may underrepresent late-round/post-plant boundary states.
4. **Compatibility risk:** changing timer semantics can shift downstream calibration baselines and historical expectations.
5. **Scope creep risk:** initiative could drift into broad calibration work before contract stabilization.

## Recommended branch plan

- Keep this proposal branch for planning artifact only:  
  `deep/proposal-20260307-0035-timer-pressure-contract`
- If approved, implement one stage per branch:
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s1-contract`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s2-diagnostics`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s3-tests`
  - `deep/stage-YYYYMMDD-HHMM-timer-pressure-s4-replay-validate`
- Rebase each stage branch from `agent-initiative-base`, never from human branches, and never self-merge.

## Recommendation

**approve planning only**
