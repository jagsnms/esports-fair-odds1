
# ENGINE BIBLE
## Chapter 1 — Probability Semantics and Temporal State Model

### 1.1 Purpose of the Engine

The CS2 Fair Odds Engine estimates the probability that **Team A wins the series** given the full current game state.

Primary output:

p_hat = P(Team A wins the series | current game state)

p_hat is always expressed in **series probability space** regardless of which temporal layer of the game is currently dominating the uncertainty.

---

## 1.2 Temporal Structure of CS2

CS2 is a hierarchical system composed of nested terminal events.

series
  ↳ map
       ↳ round
            ↳ intra-round microstates

Each layer has a terminal event that collapses uncertainty.

| Layer | Terminal Event |
|------|----------------|
Series | Team A wins series |
Map | Team A wins map |
Round | Team A wins round |
Microstate | Tactical events (kills, bomb, HP changes, time) |

---

## 1.3 Probability Envelopes

### Map Terminal Envelope (Bounds)

bound_high = P(series win | Team A wins current map)
bound_low  = P(series win | Team B wins current map)

These define the maximum and minimum reachable series probability during the map.

Invariant:

bound_low ≤ p_hat ≤ bound_high

---

### Round Terminal Envelope (Rails)

rail_high = P(series win | Team A wins current round)
rail_low  = P(series win | Team B wins current round)

Invariant:

rail_low ≤ p_hat ≤ rail_high

Rails represent the series probability endpoints if the round resolves immediately.

---

## 1.4 Round Resolution Variable (q)

q = P(Team A wins current round | current microstate)

Constraints:

0 ≤ q ≤ 1

q represents the **round win propensity** based on the current tactical state.

---

## 1.5 Coupling Equation

The true series probability is computed using the law of total probability.

p_hat = rail_low + q * (rail_high - rail_low)

Where:

rail_high = P(series win | win round)
rail_low  = P(series win | lose round)
q         = P(win round | state)

Therefore p_hat is the projection of q into series probability space.

---

## 1.6 Temporal Dominance

Although p_hat is always series probability, the signals that determine it change depending on which temporal layer dominates uncertainty.

### BUY_TIME (Series-Dominant)
Dominant signals:
- team economy
- purchasing power
- scoreline
- saved weapons

### IN_PROGRESS (Round-Dominant)
Dominant signals:
- players alive
- HP differential
- bomb state
- time remaining

### LATE ROUND (Microstate-Dominant)
Highly tactical signals dominate; p_hat converges toward the likely rail.

---

## 1.7 Phase Dependent Feature Weighting

BUY_TIME: economy signals dominate
EARLY ROUND: mixed signals
MID ROUND: combat signals dominate
LATE ROUND: tactical signals dominate

Economy influence decays as the round progresses.

---

## 1.8 Endpoint Convergence

At round resolution:

If Team A wins round → q → 1 → p_hat → rail_high  
If Team B wins round → q → 0 → p_hat → rail_low

---

## 1.9 Invariant Violations

The system must detect and log:

p_hat > rail_high  
p_hat < rail_low  
rail_high > bound_high  
rail_low < bound_low

Violations should trigger diagnostics rather than silent clamping.

---

## 1.10 Engineering Principle

The engine should not compute p_hat directly via additive adjustments.

Correct process:

1. Compute q from the microstate
2. Compute rails from round terminal states
3. Compute bounds from map terminal states
4. Derive p_hat using the coupling equation

This guarantees internal consistency between round probability and series probability.
