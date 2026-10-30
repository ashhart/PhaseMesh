# Rung B′ — Nested inversion: iterated damped scans on a context-free task

Status date: 2026-07-01. Status: **pre-registered, not yet run.** Written before
the run; results appended below, pre-registration unedited.

## Why this rung exists (the lesson of Rung B)

Rung B's conditional-composition task was refuted at smoke (commit 6255299). The
rule `s' = s ∘ f(x, φ(s))` is a map indexed by **input alone** — the "state-
dependence" closes into an input-only automaton, so every task of that shape is
regular (the **telescoping theorem**). My pre-registered bar 1 was unfalsifiable
in the wrong direction, and a 300-step smoke caught it before any GPU spend.

B′ corrects the target. The boundary of the finite-width scan class is **not**
state-dependent rules — it is **unbounded recursion**. The new task puts the
state in a *stack*: a close-bracket combines the popped outer with the
**inverse** of the enclosed product, and inversion reverses composition order —
a stack, not an automaton.

## The task

Group-valued balanced brackets over S₅. Token stream of generators
(`{id, (01), 5-cycle}`, ids in `[0,120)`) interleaved with `OPEN` (120), `CLOSE`
(121), and distractors (`[122, 122+ND)`). The reducer maintains a stack of saved
outer products and a current active product:

```
open   ->  push active;  active = IDENTITY
gen g  ->  active = MUL[active, g]
close  ->  active = MUL[pop, INV[active]]     # INVERT the enclosed product
```

`INV` of a product reverses composition order: `(g₁g₂)⁻¹ = g₂⁻¹ g₁⁻¹`. So a
close-bracket's effect on the state depends on a value **buried in the stack**,
not on the current input token. There is no `F_x` that closes this — the exact
property whose absence killed Rung B.

**Why this escapes the telescoping theorem.** In Rung B the close-rule was
`F_x(s) = s ∘ f(x, φ(s))`: a fixed map on the finite state set, indexed by input
alone, so it folded into an input-only automaton. Here the close update reads
`pop`, a value stored under unbounded nesting. The state that decides the close
is the *stack contents* (unbounded), even though the output (one group element)
is finite. Unbounded-buried-memory is the definition of context-free-but-not-
regular (Dyck structure + order reversal).

## The protagonist: damped DreamScan

Rung B's second salvaged finding was that **undamped** sweep extrapolation
diverges: DreamScan evaluated at S=16 after S=4 training collapsed to 0.19–0.43.
The fix is the damped contraction, the idiom already in `equilibrium.py` /
`settling_lm.py`:

```
z ← z + sigmoid(α) · (block(z) − z),     σ(α) ∈ (0,1) ⇒ contraction ⇒ self-stabilizing
```

`DampedDreamScanLM` iterates one weight-tied `NonAbelianLayer` (Rung A's
validated block) S times with this damped update. Sweep `s` recomputes every
transition from sweep `s−1`'s output — each sweep is one global round of stack
resolution at O(log L) parallel depth, and S is a free dial at eval. This is
DREAM's two-timescale thesis (settling's adaptive serial depth, in units of
parallel sweeps) with the oscillator replaced by the validated non-abelian scan.

**Predicted mechanism signature** (the analogue of Rung A's reflection-count
dose-response): a sweep resolves one nesting level innermost-out, so a nest of
depth `D` needs `≈ D` sweeps. The depth-dose × sweep-dose grid should saturate
along the diagonal `S* ≈ D_max`. That diagonal is the crisp, falsifiable
signature — if accuracy saturates at `S* ≪ D_max`, the model is using a
width-based shortcut (and the class claim is weaker than stated); if it never
saturates, iteration isn't buying the capability.

## Controls

| Arm | Role | Expected |
|---|---|---|
| `dreamscan S=S_train` | protagonist | sweep-dose climbs with D_max; damped, no S=16 collapse |
| `scan 1-sweep 2-layer` | **class boundary** | ≥0.90 at D=1 (flat=regular), ≤0.30 at D=8 (deep=context-free) |
| `scan untied 8-layer` | "is iteration just depth?" | matched conditioning rounds, ~6× the params |
| `GRU` | **negative control** (cf. Rung A/B positive ceiling) | ≤0.30 at D=8 — fixed-width serial model can't track an unbounded stack |
| `invert=False` | **void-check** | brackets are a no-op (associativity) → collapses to flat composition (regular); fixed-width scan MUST solve it |

The GRU is a **negative** control here, where in Rung A/B it was the positive
ceiling. This is the load-bearing difference: the regular/serial tasks of the
earlier rungs are exactly what a GRU solves; a context-free task is exactly what
it cannot. If the GRU *solves* deep nesting, the task is too shallow or secretly
regular, and the rung is void.

The void-check is the analogue of Rung A's diagonal (commutative) control: remove
the load-bearing property (non-commutativity there; inversion here) and the task
must collapse to the easy class. Without inversion, `close = MUL[pop, active]`
makes brackets a pure regrouping — by associativity the reduction equals the flat
left-to-right product of the generators (verified in
`test_non_inverting_equals_flat_composition`). A fixed-width scan must solve it.

## Pre-registered bars (GO requires all five)

Protocol: noisy stream, train L=32, gen_prob=0.5, ND=30, 6000 steps, batch 64,
d=128, H=8, d_h=16, n_h=4, AdamW(1e-3, betas=(0.9,0.98), wd=0.01 / no-decay on
biases+LayerNorm+h0), warmup 400 + cosine to 0.1·lr, clip 1.0; eval n=1024
last-token acc, seeds 0–2; `beta_init=identity` (see decision-rule note);
`damp_init=-2.0` (σ≈0.12). All arms trained at `D_max = max(D_maxs) = 8` and
evaluated across `D_maxs ∈ {1,2,4,8}` and sweep-grid `{1,2,4,8,16}`.

1. **Class boundary (depth-dose).** Fixed-width scan S=1 (2-layer):
   **≥0.90 at D_max=1** (flat = regular) AND **≤0.30 at D_max=8** (deep =
   context-free).
2. **Sweep-dose resolves nesting + damping prevents divergence.** DreamScan
   last-token accuracy is monotone non-decreasing in `S_eval` over
   `{1,2,4,8,16}` at each `D_max` (one inversion ≤0.05 tolerated), reaching
   **≥0.80 at `S_eval = D_max`**; and **no collapse at `S_eval=16`** after
   training at `S_train=4` (last-token ≥0.50, contrasting Rung B's undamped
   0.19–0.43).
3. **Iteration ≠ untied depth.** DreamScan at its saturating `S_eval` beats the
   untied-8-layer scan at `D_max=8`, non-overlapping ±1σ — despite the scan
   having ~6× the block parameters (dreamscan 61k vs untied-8-layer 362k at
   d=128). Weight-tied fixed-point iteration beats fixed feedforward depth at
   the same conditioning-round budget.
4. **Negative control.** GRU **≤0.30 at D_max=8**. If the GRU solves deep
   nesting, the task is too shallow or secretly regular → void.
5. **Void-check (non-inverting collapses).** Fixed-width scan S=1 **≥0.90** on
   the `invert=False` variant at `D_max=8` (brackets are a no-op without
   inversion → regular). If it fails, the generator is broken → void.

## Outcomes

- **GO** = all five bars.
- **PARTIAL** = 1 ∧ 2 ∧ 4 ∧ 5 but **not** 3 (iteration matches untied depth
  without exceeding it: still a class separation over S=1, weaker novelty —
  reported as such, Rung C halting still on the table).
- **NO-GO** = DreamScan can't learn even at `D_max = S_train` (train <0.80 at
  D=4, S=4) OR bar 5 fails (void).
- **REFUTED** (the Rung-B outcome) = bar 1 fails the **wrong way**: scan S=1
  **≥0.80 at D_max=8** → the task is secretly regular again → commit the
  refutation (mirroring 6255299) and stop. The smoke's first job is to test this
  before any GPU spend.

## Decision rule: `beta_init`

Fixed in advance from the parity ablation: `split` if its P2 confirms (3-seed
mean @128 ≥0.80, std <0.15), else `identity`. The parity verdict is not on this
disk at pre-registration time, so the default is **identity**; if the parity
result lands confirming `split` before the full run, every scan-family arm
switches to `split` and the run re-launches. No other knobs move between arms.

## Honest caveat on the class claim

For any **fixed** `D_max`, the task is technically regular: a finite automaton
over `120^(D_max+1)` states (the stack contents bounded by D_max) decides it.
The clean context-free claim is the **unbounded** case. The bars approximate it
at `D_max=8`, where `120^9 ≈ 10^18 ≫` any model's hidden width (`d=128` ⇒ the
recurrence carries <2¹⁰²⁴ distinct states in principle, far fewer reachable
under training). So bar 1 is an **empirical-capacity** separation, theory-
motivated (the unbounded task is genuinely context-free) but not a theorem about
the d=128 model. This is stated rather than overclaimed — the contrast with Rung
B's false "provable-flavored" bar is deliberate.

## Run order (the discipline that caught Rung B)

1. `nested_inversion.py` + tests; `pytest tests/test_nested_inversion.py` — all
   pass before any training. **Done (7/7).** The tests caught two real generator
   bugs (an off-by-one in close-forcing that left sequences unbalanced, and a
   void-check test that originally asserted the wrong invariant) — both fixed.
2. This pre-registration, written before the run. **This file.**
3. **Smoke** (`--quick`: 300 steps, 1 seed, all arms, full depth-dose +
   void-check). The smoke's job is exactly Rung B's: catch a class-boundary
   failure (REFUTED) before GPU spend.
4. Only on smoke GO: launch the full 3-seed run + JSON, append results below,
   commit.

---

# Results (appended after the run; pre-registration above unedited)

Run date: 2026-07-02, NVIDIA GB10 GB10, full pre-registered protocol. Raw:
`rungBp_full.json` (+ `rungBp_smoke.json`; smoke cleared the REFUTED trigger:
scan-S1 at 0.062 on D=8 at 300 steps — every arm at floor there, first sign of
hard credit assignment).

## Depth-dose (train_L=32, 3 seeds)

| D_max | dreamscan S=4 | scan S=1 2-layer | scan untied 8-layer | GRU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.869 ± 0.017 | 0.911 ± 0.010 | 0.939 ± 0.004 | 0.574 ± 0.008 |
| 2 | 0.813 ± 0.003 | 0.848 ± 0.000 | 0.901 ± 0.001 | 0.408 ± 0.025 |
| 4 | 0.760 ± 0.012 | 0.818 ± 0.003 | 0.880 ± 0.017 | 0.380 ± 0.015 |
| 8 | 0.775 ± 0.008 | **0.795 ± 0.008** | 0.880 ± 0.011 | 0.368 ± 0.026 |

Sweep×depth grid at D=8 (trained S=4): S=1→0.077, S=2→0.394, **S=4→0.751**,
S=8→0.599, S=16→0.201. Length-gen at D=8: every arm collapses (untied-8:
0.889 → 0.019 from L=32 → 128; dreamscan 0.739 → 0.014; GRU 0.356 → 0.009).

## Verdict: **NO-GO** (formal clause fired), with bar 1 in the mapping gap

- **Bar 1 FAILED** — scan-S1 at D=8 is 0.795 ± 0.008, nowhere near ≤0.30. It
  is 0.005 below the formal ≥0.80 REFUTED trigger — landing in the 0.30–0.80
  band the outcome mapping left unmapped (a drafting gap acknowledged before
  the tables landed). Substantively this is the same design failure the
  REFUTED clause names, and it is treated as such; the 0.005 is not read as
  meaningful.
- **Bar 2 FAILED** — sweep-dose peaks exactly at the trained S=4 and collapses
  at S=16 (0.201 < 0.50): **damping as parameterized did not deliver eval-time
  sweep extension**. Training at fixed S teaches an S-sweep circuit, not a
  convergent operator.
- **Bar 3 FAILED** — untied depth beats tied iteration everywhere (0.880 vs
  0.751 at D=8).
- **Bar 4 marginal-failed** — GRU 0.368 > 0.30, though it is by far the worst
  arm (directionally as predicted).
- **Bar 5** run separately for record completeness (`rungBp_voidcheck.json`);
  cannot change the verdict.
- **Formal NO-GO clause fired**: dreamscan 0.760 < 0.80 at D=4, S=4.

## Diagnosis (the part worth keeping)

1. **The depth-dose is flat for every arm** (scan-S1 slides 0.911 → 0.795 from
   D=1 → 8; no cliff anywhere). At train_L=32, a depth-8 nest needs 16+
   bracket tokens — deep nests are rare events, so the D_max *parameter* never
   stresses realized depth. The class boundary was never actually probed.
   Pre-diagnosed before the tables landed; confirmed by them.
2. **Nothing length-generalizes — the first task in this program where every
   arm dies together.** On noisy-S₅, scan, settling, and GRU all extrapolated;
   here even the strongest arm (untied-8) falls 0.889 → 0.019. What every
   model learned is bounded-window pattern matching, not stack resolution. The
   context-free wall is plausibly real — but as parameterized it crushes all
   arms equally and separates nothing.
3. **DreamScan never got a fair test of its thesis.** Fixed-S training
   produced a fixed-depth circuit (bar 2's shape). The DEQ literature's known
   medicine — randomized unroll depth, fixed-point/deep supervision — was not
   applied. Any B″ must fix the *training objective for convergence* before
   the sweep-dial claim is testable.

## Bar 5 — void-check result (appended when its run completed)

`rungBp_voidcheck.json` (invert=False, 1 seed): scan-S1 at D_max=8 = **0.996**
(bar: ≥0.90) — **PASS**. Remove inversion and the fixed-width scan solves
every depth essentially perfectly at train length; the generator and protocol
are sound, and the 0.996 → 0.795 gap at D=8 is attributable to inversion
specifically. The task's hardness was real — just not realized deeply enough
to probe the class boundary. Verdict tally: bars {1✗, 2✗, 3✗, 4 marginal-✗,
5✓} → NO-GO stands.

Two incidental observations from the void-check, recorded not claimed:

- Even the non-inverting (regular) variant extrapolates poorly relative to
  noisy-S₅ (scan-2: 0.545 @64 here vs 0.938 there) — bracket-structured
  per-position supervision changes the learning problem even when the
  final-token semantics reduce to flat composition. Worth remembering when
  designing any B″ target metric.
- On the regular variant, **the weight-tied iterated model is the most
  length-robust arm** (dreamscan 0.606/0.245/0.105 at 64/96/128 vs scan-2's
  0.545/0.116/0.031 and untied-8's 0.444/0.092/0.036) — the ordering flips
  relative to invert=True. Weak, single-seed, but the first hint of a
  tying-helps-extrapolation effect in this program.

## Amendments required for any B″ (declared, not run)

realized-depth-binned evaluation; nest sampling that forces deep structure at
train length (adversarial bracket placement) or longer train_L; DEQ-style
training for sweep convergence (randomized S_train, fixed-point penalty);
only then re-pose the five bars. Three misses on the B-line (B refuted at
smoke, B′ NO-GO, sweep-extension twice failing) are themselves data: the
separation, if reachable at this scale, will not come cheap.
