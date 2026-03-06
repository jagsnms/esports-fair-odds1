
# ENGINE BIBLE

## Table of Contents

### Part I — Foundations
1. Chapter 1 — Core Model Architecture
2. Chapter 2 — Rail Endpoint Philosophy
3. Chapter 3 — Rail Construction

### Part II — Intra‑Round Mechanics
4. Chapter 4 — PHAT Movement Mechanics
5. Chapter 5 — Round Resolution Signals

### Part III — Engine Operation
6. Chapter 6 — Model Calculation Flow
7. Chapter 7 — Engine Invariants

### Part IV — Calibration and Testing
8. Chapter 8 — Calibration Philosophy
9. Chapter 9 — Testing Framework

### Part V — AI & Development Workflow
10. Chapter 10 — AI Implementation Contract

---

# Chapter 1 — Core Model Architecture

Defines the mathematical structure of the engine.

PHAT represents series probability.

q represents round win probability.

Rail endpoints represent the series probability if the round resolves in favor of one team.

Core identity:

p_hat = rail_low + q * (rail_high - rail_low)

Where:
rail_high = series probability if Team A wins the round
rail_low = series probability if Team B wins the round

PHAT therefore moves within the rail envelope based on round probability.

---

# Chapter 2 — Rail Endpoint Philosophy

Rails represent the series probability AFTER the round resolves.

Rails must only depend on signals that persist across the round boundary.

Allowed signals:
- team economy totals
- loss bonus state
- weapons carried into the next round
- persistent equipment (armor / defuse kits)

Not allowed:
- HP remaining
- positioning
- time remaining
- utility used within the round

Rails represent the starting conditions of the next round.

---

# Chapter 3 — Rail Construction

Rail endpoints represent a range of outcomes based on round quality.

Example:
Winning a round with 5 players alive and a large opponent economy disadvantage produces a stronger rail endpoint than winning with 1 player alive against a wealthy opponent.

Rails therefore exist inside a conceptual envelope determined by carryover state such as:
- surviving weapons
- team economy
- opponent economy
- loss bonus progression

---

# Chapter 4 — PHAT Movement Mechanics

PHAT moves continuously during the round based on q.

Movement characteristics:
- Early round: slow movement
- Mid round: moderate adjustment
- Late round: rapid convergence toward rail endpoints

This mirrors the natural flow of CS rounds where information and decisive events increase as the round approaches resolution.

---

# Chapter 5 — Round Resolution Signals

The engine evaluates:

P(round_win | current_round_state)

Round resolution depends on three signal groups.

## Combat Strength
- players_alive
- current_loadout_value
- hp_state

These combine to represent effective fighting capability.

## Objective State
- bomb_planted
- bomb_timer_remaining

Bomb state changes the win condition and round dynamics.

## Timer Pressure

Timer pressure increases the decisiveness of the current state as time approaches zero.

Before bomb plant:
Timer pressure favors Counter‑Terrorists.

After bomb plant:
Timer pressure favors Terrorists.

Timer influence follows a non‑linear curve where the effect is minimal early in the round and accelerates near expiration.

A hard boundary condition exists post‑plant:

If time_remaining < defuse_time
CT probability = 0

Future versions may incorporate deeper contextual signals such as bomb location, site proximity, and map control.



# Chapter 6 — Model Calculation Flow

This chapter defines the canonical pipeline used by the engine to compute PHAT
from live game state. The purpose is to prevent architectural drift and ensure
that all implementations (human or AI) follow the same computational order.

The engine must always follow this sequence.

---

## Step 1 — Round State Input

The engine ingests the normalized current round state.

Signals include:

Combat signals
- players_alive
- loadout_value
- hp_state

Objective signals
- bomb_planted
- bomb_timer_remaining

Round timer
- round_time_remaining

Carryover signals
- team economy
- opponent economy
- saved weapons

These signals form the **current round state vector**.

---

## Step 2 — Compute Combat Capability

From the round state, the engine evaluates combat capability for both teams.

Primary variables:

players_alive  
loadout_value  
hp_state  

These combine to represent **effective fighting capability**.

The output of this step is a **relative combat advantage state**.

---

## Step 3 — Apply Objective State Modifiers

The engine adjusts the combat state using objective conditions.

Relevant signals:

bomb_planted  
bomb_timer_remaining  

Objective state changes the win condition of the round and modifies how combat
strength is interpreted.

---

## Step 4 — Apply Timer Pressure

Timer pressure acts as a **state amplification scaler**.

Inputs:

round_time_remaining  
bomb_planted_state  

Timer influence follows a non‑linear acceleration curve.

Before bomb plant:
Timer pressure favors Counter‑Terrorists.

After bomb plant:
Timer pressure favors Terrorists.

The closer the timer approaches zero, the stronger the pressure.

---

## Step 5 — Compute Round Win Probability (q)

Using combat state, objective state, and timer pressure, the engine computes:

q = P(team A wins the round | current round state)

Constraints:

0 ≤ q ≤ 1

q represents the probability that Team A wins the current round if the round
resolved from the current state.

---

## Step 6 — Compute Rail Endpoints

Rails represent the **series probability if the round resolves immediately**.

rail_high = series probability if Team A wins the round  
rail_low  = series probability if Team B wins the round

### Dynamic Rail Rule

Rails MUST be recomputed every tick.

Reason:
Rails represent conditional post‑round states that depend on the current round
context (the rail envelope concept). As the round evolves, the quality of a
potential round win changes, which changes the resulting series probability.

Therefore rails are **dynamic round-terminal endpoints**.

### Allowed Rail Inputs (Carryover Signals Only)

- team economy
- opponent economy
- loss bonus state
- weapons carried into the next round
- persistent equipment

### Forbidden Rail Inputs

Rails must NEVER depend on transient signals such as:

- HP remaining
- positioning
- flashbangs remaining
- round timer
- bomb site control

These signals do not persist across the round boundary.

---

## Step 7 — Compute Target PHAT

Using q and the rails:

target_p_hat = rail_low + q * (rail_high - rail_low)

This represents the **series probability implied by the current round state**.

---

## Step 8 — Apply Movement / Confidence

PHAT should not jump instantly to the target value.

Instead, PHAT moves toward target_p_hat using a confidence / inertia rule.

Example conceptual form:

p_hat = p_hat + confidence * (target_p_hat - p_hat)

Confidence increases as the round approaches resolution.

Early round → slower movement  
Late round → faster convergence

This mirrors the natural tempo of CS rounds.

---

## Step 9 — Output PHAT

Final output:

p_hat = P(series win | current game state)

Invariant:

rail_low ≤ p_hat ≤ rail_high

PHAT must always remain within the rail envelope.



# Chapter 7 — Engine Invariants

## Purpose

Engine invariants define mathematical truths the engine is expected to satisfy
when correctly implemented and calibrated.

Invariants serve two purposes:

1. Protect the mathematical integrity of the engine.
2. Provide diagnostic signals when the model behaves incorrectly.

During testing and development, violations of *behavioral invariants* should
trigger diagnostics rather than silent correction.

The engine distinguishes between:

Structural invariants — must never be violated.
Behavioral invariants — expected to hold in a correct model but may be violated
during testing to reveal model flaws.

---

# 7.1 Structural Invariants (Hard Rules)

These invariants must never be violated in any mode.

## 7.1.1 Round Probability Bounds

Round probability must always remain a valid probability.

0 ≤ q ≤ 1

Violation indicates a fundamental calculation error.

---

## 7.1.2 Rail Ordering

Rail endpoints must never cross.

rail_low ≤ rail_high

If rails cross, the rail envelope becomes invalid and the engine structure is broken.

---

## 7.1.3 Series Resolution Boundary

If the series outcome is already decided, PHAT must reflect the final outcome.

If Team A has already won the required maps:

p_hat = 1

If Team B has already won the required maps:

p_hat = 0

---

# 7.2 Behavioral Invariants (Diagnostic Rules)

These invariants describe the expected behavior of a correct engine but may be
violated during testing. Violations should be logged and investigated.

---

## 7.2.1 Rail Containment (Theoretical)

In a correctly calibrated model, PHAT should remain inside the rail envelope.

rail_low ≤ p_hat ≤ rail_high

This follows directly from the coupling equation:

p_hat = rail_low + q * (rail_high − rail_low)

### Testing Policy

During testing and calibration, PHAT must be allowed to leave the rail envelope.
This allows the engine to expose model weaknesses and calibration errors.

Violations should trigger diagnostics but must not be silently clamped.

### Production Policy

Optional production safety constraints may restrict displayed PHAT values to the
rail envelope. If enabled, this behavior must:

• be explicitly configurable  
• emit diagnostics when triggered  
• never be used during calibration or testing

---

## 7.2.2 Convergence at Round Resolution

As round probability approaches certainty, PHAT should converge to the
appropriate rail endpoint.

If q → 1, then p_hat → rail_high.

If q → 0, then p_hat → rail_low.

Observed deviations indicate movement logic or calibration errors.

---

## 7.2.3 Timer Directionality

Timer pressure must push round probability in the correct direction.

Before bomb plant:

As round_time_remaining decreases, CT advantage should increase.

After bomb plant:

As bomb_timer_remaining decreases, T advantage should increase.

Violations indicate incorrect timer pressure modeling.

---

## 7.2.4 Monotonic Combat Improvements

Improving the combat state for a team should not decrease its round win probability.

Examples of monotonic improvements:

• gaining a player
• improving loadout strength
• increasing team HP
• planting the bomb

Violations suggest feature weighting or interaction errors.

---

# 7.3 Diagnostics

When behavioral invariants are violated, the engine should emit structured
diagnostic information including:

• current round state
• computed q
• rail_low / rail_high
• p_hat
• timer state
• movement parameters

These diagnostics enable systematic debugging and calibration.



# Chapter 8 — Calibration Philosophy

## 8.1 Purpose

Calibration adjusts numerical parameters so the engine’s predictions better match
observed outcomes. Calibration improves accuracy but must never change the
structural meaning of the engine defined in earlier chapters.

The structure of the engine (PHAT definition, rail mechanics, invariants, and
core pipeline) is fixed. Calibration is only permitted to tune parameters within
that structure.

---

## 8.2 Structural vs Calibrated Components

### Structural Components (Not Calibrated)

The following elements define the identity of the engine and must never be
modified through calibration:

- PHAT definition
- q definition
- rail coupling equation

p_hat = rail_low + q * (rail_high - rail_low)

- rail philosophy (carryover-only signals)
- timer directionality
- invariant rules
- engine calculation flow

Calibration must never alter these.

---

### Calibrated Components (Allowed)

Calibration may adjust parameters inside the model including:

- feature weights
- timer curve parameters
- combat strength scaling
- interaction coefficients
- PHAT movement confidence parameters

These adjustments refine the model but must not change the architecture.

---

## 8.3 Signal Hierarchy Protection

The engine organizes signals into fixed groups:

Combat Strength  
Objective State  
Timer Pressure  

Calibration may tune signals **within these groups** but must not alter the
hierarchical relationship between groups.

Example:

Calibration may adjust:
- players_alive weight
- loadout_value weight
- hp scaling

Calibration must not change the structural relationship such as making timer
pressure dominate combat strength.

This rule preserves the conceptual identity of the engine.

---

## 8.4 Calibration Objectives

Calibration should optimize:

### Probability Accuracy
Predicted probabilities should match observed frequencies over large samples.

### Stability
Small changes in state should not create extreme probability swings unless
justified by major events.

### Interpretability
Model behavior should remain understandable and explainable.

---

## 8.5 Calibration Constraints

Calibration must respect all engine invariants.

Any calibration that produces:

- q outside [0,1]
- rail crossings
- timer direction reversals
- structural invariant violations

must be rejected.

---

## 8.6 Calibration Data Sources

Calibration should rely on:

- historical match data
- replay simulation datasets
- archived telemetry logs

Data should represent real match conditions.

---

## 8.7 Data Collection Autonomy

The calibration system may determine that additional data is required for
improving the model.

In such cases the system is allowed to generate **data collection scripts**
whose purpose is to collect missing or insufficient telemetry.

Examples include:

- logging additional round-state variables
- increasing sampling frequency
- storing intermediate engine outputs
- capturing contextual match metadata

These scripts exist solely for improving calibration datasets and must not
modify engine behavior.

This rule allows the calibration system to request the exact data format and
fields required for proper statistical analysis.

---

## 8.8 Calibration Transparency

Every calibration update must generate a report including:

- parameters modified
- reason for modification
- before/after performance metrics
- invariant validation results
- datasets used

Calibration must always remain auditable and reversible.

---

## 8.9 Calibration Iteration Strategy

Calibration should proceed incrementally:

1. validate invariants
2. adjust parameters
3. test against replay datasets
4. evaluate probability calibration
5. deploy incremental improvements

Large uncontrolled parameter changes should be avoided.



# Chapter 9 — Testing Framework

## 9.1 Purpose

The testing framework ensures engine behavior stays aligned with the Bible.
Testing is not optional “nice-to-have”; it is the mechanism that prevents drift,
regressions, and silent philosophical violations.

Testing serves four goals:

- Detect structural errors
- Validate invariants
- Measure probability calibration quality
- Prevent regressions across real match conditions

---

## 9.2 Test Categories

### 9.2.1 Structural Tests (Hard)

Structural tests verify the engine respects fixed architecture.

Examples:
- 0 ≤ q ≤ 1
- rail_low ≤ rail_high
- series-decided → p_hat ∈ {0,1}
- pipeline order: state → q, rails → target_p_hat → movement → p_hat

Structural failures must fail CI and block merges.

---

### 9.2.2 Invariant Tests (Hard + Diagnostic)

Invariants from Chapter 7 are tested in two modes:

Hard structural invariants:
- must never be violated

Behavioral invariants:
- may be violated during testing
- must emit diagnostics
- must be trackable over time (counts, rates, distribution)

---

### 9.2.3 Replay Validation Tests (Primary)

Replay tests run the engine against recorded telemetry / match logs.

Goals:
- Validate PHAT trajectories under real conditions
- Detect pathological movement (oscillation, jitter, step explosions)
- Verify rail envelope dynamics under changing carryover context
- Ensure timer pressure behaves correctly pre-plant vs post-plant

Replay validation is the main testing source of truth.

---

### 9.2.4 Scenario Stress Tests (Targeted)

Scenario tests validate edge cases and rare-but-important states.

Examples:
- extreme economy mismatches
- 5v1, 1v5, 2v3 clutch states
- bomb timer near-zero boundary cases
- buy-time / pre-buy transitions
- low data / missing telemetry degraded-mode behavior

Scenario tests ensure “weird states” don’t break the engine.

---

## 9.3 Convergence Testing

The engine must demonstrate correct convergence behavior.

As q approaches certainty:
- if q → 1, then p_hat should converge toward rail_high
- if q → 0, then p_hat should converge toward rail_low

This is evaluated statistically over replays (not just hand-picked examples).

---

## 9.4 Rail Integrity Tests

Rails must satisfy:
- carryover-only dependency (no transient microstate leakage)
- dynamic recomputation each tick (rail envelope concept)
- ordering: rail_low ≤ rail_high

Rail integrity testing includes “extreme economy” and “saved guns” cases to
ensure endpoints respond to carryover context as intended.

---

## 9.5 Probability Calibration Tests

Calibration quality is assessed via standard probability validation techniques.

Required tests:
- Binned reliability (calibration) curves
- Brier score (or equivalent)
- Log loss (or equivalent)

Example:
If the model predicts 0.60 in a large bucket of states,
the observed win rate should be ~0.60 within reasonable tolerance.

---

## 9.6 Regression Tests

Regression tests compare “before vs after” engine versions:

- PHAT trajectories
- q outputs
- rail endpoints
- invariant violation rates
- calibration metrics

Unexpected deviations must be flagged and explained.

---

## 9.7 Continuous Testing Triggers

Testing should run automatically on:
- every commit touching engine math, rail logic, or movement
- calibration parameter changes
- telemetry ingestion / normalization changes

Testing must be cheap enough to run frequently, with deeper suites run nightly.

---

## 9.8 Diagnostic Logging Contract

When a behavioral invariant or scenario test fails, the system must record:

- full round state vector (as logged fields)
- q, rail_low, rail_high
- target_p_hat and p_hat
- movement parameters (confidence/inertia)
- timer state (round / bomb)
- reason codes for failure

This diagnostic payload is required for iterative calibration.

---

## 9.9 Simulation Testing (Required Target)

The testing framework must include a simulation “sandbox harness” capable of
generating synthetic match/round trajectories to stress the engine.

This exists to:
- expose probability drift
- test stability under random-but-plausible sequences
- validate end-to-end behavior when replay coverage is incomplete
- enable rapid iteration without waiting on new real data

### Phase 1 — Synthetic State Generator (Immediate)

Implement a generator that produces plausible sequences of round states, e.g.:
- players_alive trajectories (5→4→3→…)
- loadout regimes (eco / half / full buy)
- bomb planted transitions and bomb timer countdown
- timer pressure ramp near zero
- carryover context transitions (saved guns, economy swings)

The goal is not perfect realism; the goal is coverage + stress.

### Phase 2 — Policy-Driven Simulation (Future)

Add “agent policies” to generate more realistic sequences:
- default execute timing
- retake timing
- clutch behavior templates
- eco force patterns

This produces richer distributions while still being synthetic.

### Simulation Contract

Simulations must:
- preserve invariants (structural invariants must never break)
- produce diagnostic artifacts (plots/metrics) comparable to replay tests
- be reproducible (seeded RNG)

---

## 9.10 Autonomous Sandbox Branch (AI-Driven Iteration Target)

A dedicated sandbox branch may be used for automated iteration loops:

1. Implement a candidate change aligned to the Bible
2. Run replay + simulation tests
3. Calibrate parameters if allowed
4. If failures occur: adjust, re-test, and log rationale
5. Commit changes on the sandbox branch only
6. Produce a “promotion report” proposing a PR to main

This loop must never modify main directly.
Promotion requires human review.



# Chapter 10 — AI Implementation Contract (Cursor Automations)

## 10.1 Purpose

This chapter defines the rules for automated agents (Cursor Automations and any
equivalent AI workflow) that modify the repo.

The agent’s job is to improve alignment between:
- the ENGINE BIBLE (intended behavior)
- the repo (actual implementation)

The agent must operate safely, transparently, and in a way that keeps changes
auditable and reviewable by a human.

---

## 10.2 Required Inputs

Before making any change, the agent must read:

1. ENGINE_BIBLE (all chapters)
2. ENGINE_INVARIANTS (if present as a separate reference doc)
3. The current test suite definitions and how to run them
4. Recent diagnostics / calibration outputs (if available)

The Bible is the source of truth. If repo behavior conflicts with the Bible,
the agent must treat this as a bug or missing implementation, not a reason to
reinterpret the Bible.

---

## 10.3 Safety Model

### 10.3.1 Sandbox Branch Only

The agent must never push directly to main.

All work must occur on a sandbox branch such as:

automation/*

Examples:
- automation/sandbox
- automation/sandbox-YYYYMMDD

### 10.3.2 No Self-Merge

The agent must never merge its own work into main.
Promotion to main must happen via human-reviewed PR.

### 10.3.3 Explicit Budget

Each run must enforce at least one budget constraint, such as:
- max wall-clock time per run
- max iterations per run
- max commits per run

When budget is reached, the agent must stop and produce a report.

### 10.3.4 No Silent Clamping During Testing

The agent must not “fix” behavioral invariant violations by adding silent clamps
or cosmetic smoothing when the Bible calls for diagnostic visibility.
Production-only safety constraints must remain explicitly configurable and must
never be enabled during calibration/testing workflows.

---

## 10.4 Allowed Scope of Changes (Sandbox Autonomy)

Inside sandbox branches, the agent is allowed to:

- Modify engine logic to better align with the Bible
- Tune parameters within the allowed calibration scope
- Add tests (replay, scenario, simulation)
- Improve instrumentation and diagnostics
- Generate data collection scripts (data-only; must not alter engine behavior)
- Refactor for clarity if it preserves the Bible-defined structure

The agent must not:
- change the engine’s core structure (Chapter 1 coupling equation)
- change rail philosophy (carryover-only rule)
- change invariant definitions
- bypass the canonical calculation flow (Chapter 6)

---

## 10.5 Improvement Campaign Model

Each automation run is an improvement campaign.

The agent must not be limited to a single narrow fix. Instead, it should:

1. Measure baseline metrics
2. Identify the highest-value current weakness
3. Iterate on that weakness until improvement plateaus
4. Re-rank remaining weaknesses
5. Continue with the next most valuable target
6. Stop when budget is exhausted or no meaningful gains remain

This campaign model repeats across runs.

On the next scheduled run, the agent should:
- read the Bible again
- read its prior PR/report
- continue from where it left off
- select the next highest-value opportunity

---

## 10.6 Target Selection (What to Work On)

The agent should prioritize targets based on measurable value such as:

- structural invariant failures (highest priority)
- behavioral invariant violation rates (PHAT outside rails, timer direction errors)
- replay test regressions
- simulation instability
- calibration metrics (Brier/logloss/reliability)
- missing diagnostics or missing data collection needed for calibration

The agent must always justify target selection in the run report.

---

## 10.7 Iteration Loop (Canonical Agent Loop)

Within a run, the agent follows this loop:

1. Select target based on metrics
2. Implement change(s)
3. Run relevant tests:
   - structural/invariant tests
   - replay validation
   - scenario stress tests
   - simulation sandbox tests (as applicable)
4. Compare metrics to baseline
5. If improved: keep and continue
6. If worse: revert and try alternative
7. Stop when plateau or budget reached

The agent must keep changes scientific:
- one target at a time
- measure before/after
- avoid mixing unrelated changes

---

## 10.8 Push and PR Strategy

Recommended pattern:
- Iterate locally within a run
- Push only when there is a coherent improvement bundle
- Open or update a PR against main

The PR must include a structured “promotion report” (see below).

The agent may create intermediate commits, but should avoid spamming the repo.
If multiple commits are needed, they must be clearly named and organized.

---

## 10.9 Loop Prevention

If the automation is triggered by repo events (push/PR update), the agent must
avoid infinite self-trigger loops.

Acceptable strategies include:
- triggering only on schedules or manual/webhook triggers
- excluding automation/* branches from triggers
- using commit-message guards (e.g., detect [automation] and exit)
- using state guards (record last processed SHA and exit if unchanged)

The selected strategy must be documented in the automation configuration.

---

## 10.10 Required Outputs (Run Report)

Every run must produce a report (PR description or attached artifact) including:

- Run goal(s) selected and why (metrics-based)
- Changes made (files + summary)
- Tests run and results
- Metrics before/after:
  - invariant failure counts
  - behavioral violation rates
  - replay metrics
  - simulation metrics
  - calibration metrics (if applicable)
- Diagnostics emitted (examples)
- What remains broken / next targets
- Whether the agent believes gains have plateaued

The report must reference Bible chapters to explain rationale.

---

## 10.11 “No Further Gains” Condition

The agent may conclude a campaign has little remaining value when:

- structural invariants are consistently satisfied
- behavioral invariant violation rates are low and stable
- replay/simulation metrics show no meaningful improvement over multiple runs
- calibration quality improvements plateau

When this occurs, the agent must state explicitly:
- what it tried
- what evidence indicates plateau
- what future data or features would be required to improve further

---

## 10.12 Human Review and Promotion Rule

Promotion to main requires human approval.

The agent’s PR is a proposal, not an authority.

Humans decide whether to:
- merge as-is
- request modifications
- cherry-pick select commits
- discard the branch

---

---

# Chapter 11 — Cursor Automation Execution Policy

## 11.1 Purpose
This policy governs all autonomous Cursor Automation runs for this repository.

The automation exists to make measurable, bounded, reviewable improvements to the project without changing model identity, violating written invariants, or drifting into speculative refactors.

The automation is not a co-owner of the architecture. It is a constrained maintenance and improvement worker operating under explicit written law.

## 11.2 Operating assumptions
Cursor Automations may run on schedules or external triggers. They may execute in cloud sandboxes, use configured tools and MCPs, and retain memory from prior runs.

Because the automation may run repeatedly and learn from prior runs, it must be governed by repository evidence and written policy rather than by its own habits or prior preferences.

## 11.3 Authority order
When deciding what is true, the automation must follow this order of authority:

1. This Bible and its explicit invariants
2. Canonical repository code and canonical test paths
3. Current test results and replay evidence
4. Current diagnostics and metrics
5. Prior automation reports
6. Automation memory

Memory is advisory only. It is never authoritative.

## 11.4 Branch isolation rules
The automation may never write directly to:
- main
- master
- dev
- release branches
- human-owned feature branches

The automation may only write to its own isolated branch lane.

Recommended structure:
- `agent-base` = approved automation starting point
- `agent/run-YYYYMMDD-issue-slug` = fresh per-run working branch

Each run must begin from the current approved automation base branch and create a fresh run branch for the selected issue.

The automation may commit and push only to its own run branch.

The automation may never merge itself.

Human review is required before any promotion into shared branches.

## 11.5 Scope of allowed work
The automation may perform only one primary issue class per run.

Allowed run types:
- audit-only diagnosis
- localized bug fix
- failing-test repair
- replay mismatch repair
- diagnostic instrumentation
- small non-structural refactor
- documentation synchronization tied directly to current code behavior

Disallowed without explicit human approval:
- architecture rewrites
- broad refactors spanning unrelated systems
- moving canonical modules
- deleting or reorganizing major legacy areas
- model identity changes
- rail identity changes
- PHAT semantic rewrites outside allowed tuning policy
- CI threshold weakening
- broad renames
- speculative “cleanup” not tied to a ranked issue
- bundling multiple unrelated fixes into one run

## 11.6 Required issue ranking policy
Each run must identify the highest-ranked unresolved issue using the following fixed priority ladder:

1. Structural invariant violations
2. Failing canonical tests
3. Confirmed replay mismatches
4. High-frequency diagnostic invariant failures
5. Missing instrumentation that blocks diagnosis
6. Calibration weaknesses
7. Cleanup or documentation

The automation must not choose issues based on novelty, curiosity, file familiarity, or aesthetic preference.

The automation must select exactly one primary issue per run.

## 11.7 Non-reselection rule
The automation must not reselect an issue it previously addressed unless one of the following is true:
- current evidence shows the issue still exists
- a regression has reintroduced the issue
- the prior run ended unresolved
- the prior run only added instrumentation and the issue is now diagnosable

A prior branch, report, or memory entry is not enough to justify rework by itself.

## 11.8 Required run workflow
Every automation run must follow this sequence:

1. Read this Bible
2. Read the latest automation reports
3. Sync to the current approved automation base branch
4. Gather current evidence from canonical tests, replay checks, diagnostics, and reports
5. Build a ranked candidate issue list
6. Select exactly one highest-ranked unresolved issue
7. Create a fresh run branch
8. Establish a baseline for the selected issue
9. Make the smallest viable change set that addresses the issue
10. Validate repeatedly in sandbox
11. Stop when diminishing returns or boundary conditions are hit
12. Commit and push the result to the run branch
13. Produce a promotion report

## 11.9 Baseline requirements
Before changing code, the automation must collect baseline evidence appropriate to the selected issue, such as:
- failing tests
- replay mismatch output
- invariant violation rates
- diagnostic counters
- calibration summaries
- current behavior screenshots or logs when relevant

The automation may not claim improvement without baseline evidence.

## 11.10 Minimal-change rule
The automation must prefer the smallest change set that can plausibly solve the selected issue.

It must not expand scope merely because adjacent code looks imperfect.

It must not “improve while here” unless the additional change is strictly necessary for the selected issue.

## 11.11 Validation rule
The automation must test the selected issue aggressively inside the sandbox.

Validation must focus on the selected issue’s evidence, not generic confidence theater.

Appropriate validation may include:
- targeted unit tests
- replay checks
- scenario tests
- invariant checks
- UI verification where relevant
- repeated runs to check stability

The automation must not confuse “lots of activity” with “proof.”

## 11.12 Diminishing returns stop rule
The automation must stop iterating when any of the following becomes true:
- the primary issue is fixed and stable across repeated validation
- additional edits produce no meaningful measurable improvement
- remaining failures are outside the selected issue scope
- further progress would require a disallowed structural change
- confidence in causality becomes mixed or weak
- the automation begins drifting into secondary or aesthetic work

Stopping cleanly is a success condition.

## 11.13 Idempotence rule
Before making changes, the automation must check whether:
- the issue is already fixed
- an equivalent fix already exists
- an existing run branch already addresses the issue
- the issue is no longer reproducible
- the intended instrumentation already exists

The automation must not duplicate work, duplicate instrumentation, or reopen already-resolved work without new evidence.

## 11.14 Reporting rule
Every run must leave both:
- a machine-readable report
- a human-readable report

Each report must include:
- selected issue
- why it outranked other candidates
- baseline evidence
- files changed
- validation performed
- before/after metrics or evidence
- unresolved risks
- reason the run stopped
- recommendation: promote, hold, or discard

## 11.15 Failure behavior
If the automation cannot confidently identify a bounded, high-priority, evidence-backed issue, it must not guess.

It may produce an audit-only report instead of code changes.

If repository evidence conflicts with automation memory, repository evidence wins.

If the Bible conflicts with automation memory, the Bible wins.

If the automation cannot stay within the permitted scope, it must stop and report rather than escalate itself.

## 11.16 Success definition
A successful run is not “many edits.”
A successful run is:
- one correctly ranked issue,
- one bounded branch,
- measurable evidence,
- reviewable changes,
- and a clear stop point.

# Chapter 12 — Cursor Automation Rule Block

You are an autonomous repository improvement agent operating under strict scope, branch, and evidence rules.

## 12.1 Core mission
On each trigger, identify exactly one highest-priority unresolved issue using canonical repository evidence, address only that issue within a fresh isolated run branch, validate until gains flatten, then stop and report.

## 12.2 Hard constraints
- Never write to main, master, dev, release, or human-owned branches.
- Only write to a fresh automation run branch created from the approved automation base branch.
- Never merge your own work.
- Never perform more than one primary issue class in a single run.
- Never change model identity, rail identity, PHAT semantics, or written invariants unless explicitly authorized by a human.
- Never weaken CI or tests to make a change appear successful.
- Never treat memory as authoritative over current repository evidence or written policy.

## 12.3 Authority order
When deciding what is true, use this order:
1. Bible and written invariants
2. Canonical code paths and canonical tests
3. Current test and replay evidence
4. Current diagnostics and metrics
5. Prior automation reports
6. Memory

## 12.4 Required issue ranking ladder
Always rank candidate issues in this order:
1. Structural invariant violations
2. Failing canonical tests
3. Confirmed replay mismatches
4. High-frequency diagnostic invariant failures
5. Missing instrumentation blocking diagnosis
6. Calibration weaknesses
7. Cleanup/documentation

Select exactly one highest-ranked unresolved issue.

Do not choose based on novelty, aesthetics, or file familiarity.

## 12.5 Non-reselection rule
Do not reselect a previously addressed issue unless current evidence shows:
- it still exists,
- it regressed,
- the prior attempt ended unresolved,
- or the prior attempt only added instrumentation and the issue is now diagnosable.

Past work alone is not enough reason to revisit the same issue.

## 12.6 Per-run workflow
1. Read the Bible.
2. Read recent automation reports.
3. Sync to the approved automation base branch.
4. Gather current evidence from canonical tests, replay checks, diagnostics, and reports.
5. Build a ranked issue list using the required ladder.
6. Select exactly one highest-ranked unresolved issue.
7. Create a fresh run branch named for the selected issue.
8. Establish baseline evidence for that issue before editing code.
9. Apply the smallest viable change set that addresses the issue.
10. Validate repeatedly in sandbox using issue-specific evidence.
11. Stop when the issue is resolved, gains flatten, or scope expansion would be required.
12. Commit and push only to the run branch.
13. Produce a promotion report.

## 12.7 Baseline rule
Before changing code, collect baseline evidence tied to the selected issue.
Examples:
- failing tests
- replay mismatches
- invariant violation counts
- missing instrumentation confirmation
- calibration summaries
- behavior logs or screenshots when relevant

Do not claim improvement without baseline evidence.

## 12.8 Minimal-change rule
Prefer the smallest change set that can plausibly solve the selected issue.
Do not make “while I am here” improvements.
Do not expand to adjacent cleanup unless strictly required for the selected issue.

## 12.9 Validation rule
Validate the selected issue aggressively.
Use targeted evidence, not generic confidence.
Possible validation:
- targeted tests
- replay checks
- scenario tests
- invariant checks
- repeated runs for stability
- UI verification where relevant

## 12.10 Diminishing returns stop rule
Stop immediately when any of the following becomes true:
- the selected issue is fixed and stable across repeated validation
- additional edits produce no meaningful measurable improvement
- remaining failures are outside the selected issue scope
- further progress requires disallowed structural work
- causal confidence becomes mixed
- work begins drifting into secondary issues

Do not continue optimizing after the selected issue reaches diminishing returns.

## 12.11 Idempotence rule
Before editing, check whether:
- the issue is already fixed
- an equivalent fix already exists
- an existing automation branch already covers the issue
- the issue is not currently reproducible
- the intended instrumentation already exists

Never duplicate instrumentation, duplicate fixes, or reopen resolved work without new evidence.

## 12.12 Failure handling
If you cannot identify a bounded, evidence-backed issue with high confidence, do not guess.
Produce an audit-only report instead of code changes.

If evidence conflicts:
- Bible beats memory
- current repo evidence beats memory
- canonical tests beat assumptions

## 12.13 Report format
Every run must produce a promotion report containing:
- selected issue
- why it outranked alternatives
- baseline evidence
- files changed
- validation performed
- before/after evidence
- unresolved risks
- explicit stop reason
- recommendation: promote, hold, or discard

## 12.14 Definition of success
Success is not maximum code churn.
Success is:
- one correctly ranked issue
- one bounded run branch
- measurable evidence
- reviewable changes
- and a clean stop point
