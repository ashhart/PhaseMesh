# Rung B — Conditional composition: separation for iterated scans (DreamScan)

Status date: 2026-07-01. Status: **pre-registered, not yet run.** Written before
the run; results appended below, pre-registration unedited.

## Claim under test

Rung A's scan has a provable-flavored ceiling: input-dependent linear RNN
transitions cannot depend on state. Conditional composition puts the state in
the transition: generator token g acts as g when φ(state)=1 (state fixes
element 0 — non-homomorphic, not computable from token counts) and as g⁻¹
otherwise. The transition required at position t is a function of the true
prefix state, so a single scan cannot express it, and a fixed stack of k scans
has only k conditioning rounds.

**DreamScan** iterates ONE weight-tied non-abelian scan block S times: sweep s
recomputes all transitions from sweep s−1's output — each sweep is one global
round of state-conditioning at O(log L) parallel depth, and S is a free dial
at eval. This is the DREAM two-timescale thesis with the oscillator replaced
by the validated non-abelian scan: settling's adaptive serial depth, in units
of parallel sweeps.

## Pre-registered bars (all on 3 seeds, noisy conditional-S₅, train L=32,
protocol as Rung A; eval lengths {32,48,64,96,128}; DreamScan S_train=4,
sweep grid {1,2,4,8,16} at eval)

**GO** requires all four:

1. **Class boundary:** scan S=1 (2-layer Rung A model) ≤ 0.30 at 2× (L=64).
2. **Iteration buys the capability:** DreamScan(S_eval=4) ≥ 0.90 at train
   length, and mean accuracy at L ∈ {64,96,128} rises with S_eval up to
   saturation (monotone up to its max, tolerance one inversion ≤ 0.02).
3. **Iteration ≠ just depth:** DreamScan at its best pre-declared grid sweep
   beats the untied 4-layer scan (matched conditioning rounds at train) at
   L=96 and L=128 with non-overlapping ±1σ.
4. **Positive control:** GRU ≥ 0.80 at 2× (task is a plain automaton for a
   serial recurrence — if the GRU fails, the task itself is broken and the
   rung is void).

**PARTIAL** = 1∧2∧4 but not 3 (iteration matches depth without exceeding it:
still a class separation over S=1, weaker novelty — reported as such).
**NO-GO** = DreamScan cannot learn the task (train-L < 0.90) or the sweep grid
is flat at extrapolation lengths.

Decision rule fixed in advance: `beta_init` for all scan-family arms is the
winner of the parity ablation (`split` if its P2 confirms, else `identity`),
decided by that run's pre-registered criteria before this run launches.
No other knobs move between arms.

## Why this is the novel object

Rung A validated exact non-abelian composition at parallel cost (known theory,
validated locally). Settling validated input-adaptive serial depth. DreamScan
is their composition: state-conditional expressivity climbing with iterated
parallel sweeps — capability strictly above the linear-scan class, at
O(S log L) depth instead of O(L). A green Rung B is, to our knowledge, a new
result, and the direct setup for sweep-halting (adaptive S) as Rung C.

---

# DESIGN REFUTED BEFORE THE FULL RUN (2026-07-01, same day)

The CPU/MPS smoke run (300 steps, quick mode) falsified bar 1 outright: the
"impossible" arm — scan S=1, 2 layers — hit **1.000 at train length and
0.977 at 4×**, in 300 steps. The full run was cancelled; the flaw is in the
task's theory, and the proof of the flaw is short:

**Telescoping theorem (why every task of this shape is regular).** For any
state-conditional rule `s_t = s_{t−1} ∘ f(x_t, φ(s_{t−1}))` over a finite
state set, define `F_x(s) = s ∘ f(x, φ(s))`. `F_x` is a *fixed map on the
state set indexed by the input alone* — the "state-dependence" closes into an
input-only automaton over transformation maps. Any such conditional task is
regular, and a linear RNN with input-indexed matrix transitions represents
its transformation monoid directly. There is no class separation here; my
"provable-flavored" claim in the pre-registration above was mathematically
false. The smoke test caught it before GPU spend.

Two genuine findings inside the failure, kept on the record:

1. **The Householder family exploits its non-invertible range.** The task's
   `F_C` map (5-cycle branch) is provably non-injective on the 120 states, so
   solving it requires projection-regime factors (β ≈ 1, the "erasure
   valley"). The 2-layer scan solved it anyway — evidence the transition
   family functions as a full transformation-monoid representation, not just
   the orthogonal group. The β=1 valley is a *feature* for aperiodic/erasing
   automata, exactly the parity story's other half.
2. **Naive sweep-extrapolation is unstable.** DreamScan evaluated at S=16
   after training at S=4 *degrades* (0.19–0.43) — iterating an undamped block
   past its training depth diverges. Any v2 must use damped/equilibrium
   updates (z ← z + α(f(z) − z), as in `equilibrium.py`) for sweeps to be a
   safe eval-time dial.

## Corrected target — Rung B′: nested inversion (regular vs context-free)

The real boundary of the (finite-width) scan class is not "state-dependent
rules" — it is **unbounded recursion**. Corrected task: group-valued balanced
brackets, where a closing bracket *inverts the enclosed segment's product*
((w)⁻¹ = reversed inverses — an order reversal, i.e., a stack, not an
automaton). Nesting depth grows ~log L with sequence length. Then:

- **Fixed-width one-pass models** (scan S=1 at any layer count, GRU — now a
  NEGATIVE control) need width ∝ nesting depth: fail as depth grows. The task
  is context-free, not regular — the class claim is real this time.
- **Iterated fixed-width scans**: sweep k can resolve nesting level k
  (innermost-out), so S ≈ depth ≈ O(log L) sweeps of the same block —
  iteration trades width for sweeps. Predicted sweep-scaling S ~ log L is the
  crisp, falsifiable signature, and Rung C's halting becomes "learned
  recursion budget."
- Habitat-relevant: nested scopes, expression evaluation, call stacks — the
  computer-world's native structure.

Full pre-registration for B′ (bars, width-matched controls, depth-dose curve)
to be written as a fresh section before any B′ run, after the parity-ablation
verdict fixes `beta_init`.
