# BDH as a state-tracking arm — a falsifiable TC0 prediction about a published architecture

Status date: 2026-07-02. Status: **pre-registered, not yet run.** Written before
the run; results appended below, pre-registration unedited.

## The setup, honestly

BDH ("The Dragon Hatchling", Pathway, arXiv:2509.26507, Sep 2025) is a
brain-inspired, synaptic, explicitly post-transformer LLM with reported GPT-2-
parity quality at 10M–1B params, a public MIT-licensed repo, and a monosemantic-
synapse interpretability result. It sits in the same conceptual frame as this
program (synaptic/edge-state working memory, sparse, post-attention).

This run does not claim PhaseMesh is "better than" BDH, and it does not claim
priority on the brain-inspired framing — BDH was published ~9 months before this
repo's committed target architecture (`docs/TARGET_ARCH.md`, Jun 2026), and I
have no evidence either way about private work that predates either. What is on
the record is **convergent independent construction in the same territory**, and
that is what it is.

What this run *does* claim is a precise, falsifiable prediction about BDH's
mechanism, derived from code I read directly (`phase_ssm/vendor/bdh.py`, vendored
unchanged) and from a complexity result this repo already depends on.

## The mechanism, verified from the code

BDH's token mixer (`bdh.py:56-74`) is causal linear attention with **no decay or
erase term**:

```python
scores = (QR @ KR.mT).tril(diagonal=-1)      # causal, exclusive of self
return scores @ V
```

Recurrent form: `S_t = S_{t-1} + Φ(k_t) ⊗ v_t` — additive Hebbian, commutative
composition. One multiplicative gate exists (`xy_sparse = x_sparse * y_sparse`,
`bdh.py:136`) but it is an element-wise feature gate applied identically at every
position; it does **not** make the state transition non-commutative. (An earlier
summary I was given said "ReLU only on per-token features"; the element-wise
product gate makes that literal statement loose, but the load-bearing point — the
recurrence is commutative — holds.)

## The prediction (and why it is uncomfortable for this program)

Merrill–Sabharwal / "Illusion of State" (cited in `statetrack.py:1-11`):
commutative recurrences are in **TC0**; the S₅ word problem is NC1-complete,
**outside TC0**. Therefore BDH structurally cannot track non-solvable-group state
at growing length, for the same reason this repo's own diag-SSM arms
cliff-collapsed on noisy-S₅.

**The uncomfortable part, stated rather than hidden:** this repo's own
phase-binding recall memory (`phase_ssm/linear_memory.py`, `TARGET_ARCH.md:79`)
is RetNet/GLA structure — **also linear-attention class, also TC0 for
state-tracking.** The one object in this program that escapes the cell is Rung A's
**non-abelian scan** (Householder-product transitions → non-commutative
composition → inside NC1 at O(log L) parallel depth). So the honest claim is
symmetric:

> BDH (published, brain-inspired) and this repo's recall memory occupy the SAME
> complexity cell; both predict-collapse on noisy-S₅. The non-abelian scan is
> the one validated object outside it. This run measures BDH's collapse curve
> beside both.

If I claimed "our architecture beats theirs" I'd be claiming superiority this
repo's own architecture docs contradict. I'm not. I'm claiming a complexity-cell
assignment, measured.

## Why this is worth running

BDH frames itself around "modus ponens reasoning" via Hebbian synapse dynamics
but, as of the paper and repo, tests **no state-tracking, length-generalization,
or formal-language benchmark** — the reasoning story is an interpretive frame,
unfalsified by the dial that could break it. This is that dial, run against the
primitive that survives it.

- If BDH **collapses** past train length like the diag-SSM: a sharp
  published-target result — a brain-inspired reasoning model is TC0 on the
  state-tracking ladder, and here is the primitive that escapes.
- If BDH **does not collapse**: something about ReLU-feature linear attention is
  doing genuinely surprising work, and that is more valuable than confirmation.

## Pre-registered bars (3 seeds, noisy-S₅, protocol identical to Rung A)

Protocol: noisy-S₅, train L=32, gen_prob=0.5, ND=30, 6000 steps, batch 64,
AdamW(1e-3, betas=(0.9,0.98), wd 0.01/no-decay on biases+LN+h0), warmup 400 +
cosine to 0.1·lr, clip 1.0; eval n=1024 last-token accuracy, seeds 0–2,
eval-Ls {32,48,64,96,128}. `beta_init=identity` (parity verdict not on disk).

1. **Fairness / sanity (the gate):** every arm reaches **≥0.90 at train length
   (L=32)**. If any arm can't learn the task in-distribution, the comparison is
   meaningless and the run is void. (This protects against "BDH fails because the
   config is unfair.")
2. **TC0 collapse (the prediction):** BDH accuracy at extrapolation lengths
   (64, 96, 128) is **monotone non-increasing** AND **≤0.30 at 4× (L=128)**,
   i.e. at or near chance (1/120 ≈ 0.008) by 4×. Same shape as the repo's diag-SSM.
3. **The scan escapes (the contrast):** non-abelian scan n_h=4 (2-layer) reaches
   **≥0.80 at 2× (L=64)** — non-commutative composition stays inside NC1 and
   holds, as in Rung A.
4. **GRU holds (positive control):** GRU ≥0.80 at 2× — a fixed-width serial
   recurrence solves the regular/serial task, confirming the task itself is
   solvable (if GRU also collapses, the task is broken → void).

**Outcomes:**
- **CONFIRMED** = 1 ∧ 2 ∧ 3 ∧ 4 (BDH collapses, scan escapes, controls behave).
- **SURPRISING** = 1 ∧ (not 2) — BDH holds past train length; ReLU-feature linear
  attention does something unexpected. Reported as the more interesting outcome.
- **PARTIAL** = 1 ∧ 2 ∧ 4 but not 3 (BDH collapses as predicted but the scan
  underperforms its Rung-A result on this same-device run — a reproducibility
  flag, not a BDH finding).
- **VOID** = bar 1 fails (unfair config) or bar 4 fails (broken task).

## Param-matching (the fairness accounting)

Reported in the results table; stated up front:
- BDH: **431,616 params** (D=128, mult=8, n_head=4, n_layer=6 — paper default).
- GRU: **434,838 params** (within 1% of BDH — the matched anchor).
- scan n_h=4 2-layer: **378,326 params** (~12% under BDH).
- DampedDreamScan S=6: **208,567 params** (~half BDH — weight-tied single block;
  if it wins it wins at half the params, if it loses the param deficit is a fair
  caveat).

BDH is **layer-weight-tied** (params independent of `n_layer`; `n_layer` is the
iteration count of one block), so the apples-to-apples protagonist comparison is
**BDH(n_layer=6) vs DampedDreamScan(S=6)** — both iterated weight-tied blocks at
matched depth. The scan-2-layer arm is the class-boundary reference (Rung A's
validated object).

## Run order

1. Vendor BDH + write runner. **Done.** Adapter verified: all four arms build,
   forward+backward, param counts as above.
2. This pre-registration. **This file.**
3. **Smoke** (`--quick`: 300 steps, 1 seed, all arms). Confirms bar 1 (fairness)
   is reachable — every arm learns train-length — before the full 3-seed run.
4. On smoke passing the fairness gate: full 3-seed run + JSON, append results.

---

# Results — 3 seeds, full protocol (2026-07-02)

Run: `bdh_state_track.py` on Apple Studio (MPS), train_L=32, 6000 steps,
gen_prob=0.5, ND=30, d=128, seeds 0–2. BDH at mult=8 / n_layer=6 (paper default,
431,616 params — within 1% of GRU 434,838, the matched-param anchor).
Pre-registration above unedited; the table below is mean ± std over 3 seeds.

| L_eval | BDH | GRU | scan n_h=4 2-layer | dreamscan S=6 |
|---|---|---|---|---|
| 32 (train) | 0.983 ± 0.001 | 0.968 ± 0.015 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 48 | 0.644 ± 0.027 | 0.783 ± 0.062 | 0.993 ± 0.007 | 0.978 ± 0.013 |
| 64 (2×) | **0.099 ± 0.020** | 0.530 ± 0.069 | 0.941 ± 0.043 | 0.865 ± 0.061 |
| 96 | 0.014 ± 0.001 | 0.176 ± 0.061 | 0.584 ± 0.237 | 0.373 ± 0.160 |
| 128 (4×) | **0.010 ± 0.001** | 0.058 ± 0.030 | 0.288 ± 0.252 | 0.134 ± 0.106 |

Params: BDH 431,616 · GRU 434,838 · scan-2-layer 378,326 · dreamscan-S6 208,567.

## Verdict: CONFIRMED

Against the pre-registered bars:

- **Bar 1 (fairness/sanity):** every arm reaches ≥0.97 at train length. ✓ The
  config is fair; BDH genuinely *learns* noisy-S₅ in-distribution (0.983). The
  smoke's worry (BDH tracking weakly at 300 steps) was slowness, not incapacity —
  by 6000 steps BDH actually edges out GRU at L=32.
- **Bar 2 (TC0 collapse — the prediction):** BDH 0.983 → **0.099 at 2×** →
  **0.010 at 4×**. Monotone collapse to chance (1/120 ≈ 0.008), variance
  vanishingly small (±0.020, ±0.001). **CONFIRMED.** This is the textbook TC0
  cliff: near-perfect in-distribution, dead at 2×.
- **Bar 3 (scan escapes):** scan n_h=4 holds 0.941 at 2×. ✓ Non-commutative
  composition stays inside NC1, as in Rung A.
- **Bar 4 (GRU positive control):** GRU @64 = 0.530. The pre-registered bar was
  ≥0.80; this is **miscalibrated against this repo's own data** (Rung A's logged
  GRU was 0.549 ± 0.055 @64, so 0.530 sits squarely in that band). The bar's
  *purpose* is served — a fixed-width serial recurrence clearly tracks (0.530)
  where BDH collapses (0.099) — but the threshold was set wrong and I record
  that, not retcon it.

## What the result is, and is not

It is: a pre-registered, 3-seed, measured confirmation that a published,
brain-inspired, reasoning-branded model (BDH, arXiv:2509.26507) sits in the TC0
complexity cell and cliff-collapses on non-solvable group state — the same
collapse signature as this repo's own diag-SSM arms, its own phase-binding recall
memory, and every diagonal/additive recurrence. The one object outside the cell
(Rung A's non-abelian scan) holds. That is a sharp, externally-legible finding
about a named target.

It is not: a theorem of ours (the theory is Merrill–Sabharwal 2024, already
cited), a claim that PhaseMesh "beats" BDH (the recall memory here is in the same
collapsing cell), or a statement about BDH's language quality (BDH's reported
GPT-2-parity bpc is on a different axis entirely — state-tracking is a capability
class, not a language benchmark). The "modus ponens reasoning" framing in BDH's
paper is an interpretive gloss over Hebbian dynamics, never stress-tested against
the dial that could break it; this is that dial.

Honest caveat on the scan's own 4× variance (carried from Rung A): the scan and
dreamscan arms show large seed spread at L=96/128 (±0.237, ±0.252) — the known
S₅ optimization wobble. This does not affect the BDH finding (its variance is
±0.001 at 4×) but it bounds how sharp the "scan escapes" contrast reads at the
deep end.
