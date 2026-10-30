# Rung A — Scannable non-abelian recurrence vs settling (noisy-S₅ length-gen)

Status date: 2026-07-01. Status: **pre-registered, not yet run.** This section
is written before the run and will not be edited after it; results and verdict
are appended below it.

## Claim under test

The serial-depth result (`results/serial_depth_lengthgen.md`) proved a settling
recurrence length-generalizes on noisy-S₅ where fixed-depth models collapse —
but settling pays **L sequential steps**. The open problem ("the prize") was
recorded as: non-TC0 expressivity at parallelizable training cost.

Theory says the target is log depth, not linear depth: the S₅ word problem is
NC1-complete (Barrington), group composition is associative, and an associative
operation admits a parallel prefix scan. A linear-in-state RNN with
input-dependent **non-commuting** transitions (products of n_h generalized
Householder factors, the DeltaNet/DeltaProduct algebra) computes the running
composition in O(log L) parallel depth. Diagonal transitions commute — that is
exactly why Mamba-class models stay TC0 and cliff-collapse here.

Constructive existence is unit-tested (`tests/test_nonabelian_scan.py`): with
n_h=4, the Householder products express both S₅ generators exactly, and the
scan reproduces the MUL-table running composition through the permutation
representation. The run decides **learnability**, not expressibility.

## Protocol (matches `s5_seeds.py` exactly)

noisy-S₅, gen_prob=0.5, ND=30; train L=32, batch 64, 6000 steps,
AdamW(lr=1e-3, betas=(0.9,0.98), wd=0.01), warmup 400 + cosine to 0.1×, clip
1.0; eval n=1024, last-token accuracy; eval lengths {32, 48, 64, 96, 128};
seeds 0–2. Primary model: `NonAbelianScanLM` d=128, 2 layers, H=8 heads,
d_h=16, n_h=4, trained through the parallel scan path. Device: local MPS.
Deviation from the shared loop, declared up front: no weight decay on biases,
LayerNorm params, and h0 (wd on the raw-beta bias drags transitions toward the
state-erasing β=1; standard hygiene, applied only to the new model). The GRU
anchor is retrained on the same device with the unmodified protocol; its match
to the logged CUDA/CPU numbers (0.549±0.055 @64) validates cross-device
comparison against the logged settling numbers (0.891±0.018 @64).

## Pre-registered verdict (written before the run)

- **GO** — 3-seed mean of nonabelian-scan (n_h=4) at 2× train length (L=64)
  **≥ 0.80**, AND above the GRU at every extrapolation length with
  non-overlapping ±1σ at L=64, AND within **0.05** of settling's logged mean at
  L=64 (i.e. ≥ 0.841) or above it. Meaning: the scannable algebra captures the
  settling capability at parallel training cost — the prize has a concrete
  candidate.
- **PARTIAL** — beats the GRU (non-overlapping ±1σ at L=64) but lands more than
  0.05 below settling at L=64. Meaning: the algebra captures part of the
  capability; settling retains an advantage worth isolating.
- **NO-GO** — fails to beat the GRU at 2×. Meaning: the cheap scan route does
  not capture the capability at this scale; the prize stays open as stated.

Controls (must both behave, else the verdict is void):

1. **Diagonal void-check**: same architecture, same eigenvalue range
   (λ = tanh ∈ (−1,1)) but commutative transitions must **collapse toward
   chance past train length** (like the logged diag-SSM: 0.011 @64). If the
   diagonal control also length-generalizes, the task is not testing
   non-abelian-ness and the whole rung is void.
2. **Dose-response**: extrapolation accuracy should **rise with n_h**
   (1 < 2 < 4, seed 0). A transposition needs 1 reflection, the 5-cycle needs
   4 — if capability does not rise with the algebra's expressive dose, the
   mechanism story is wrong even if the headline number is good.

Secondary (after the verdict, either way): same-device settling anchor
(seed 0) for a direct wall-clock and accuracy comparison; A₅ replication if
GO.

Anything not meeting a bar above is reported as-is; a red result stops this
rung and is committed with the same prominence as a green one.

---

# Results (appended after the run; pre-registration above unedited)

Run date: 2026-07-01. Primary on Mac mini M4 (MPS) as registered; secondary
stages on NVIDIA GB10 GB10 (CUDA) — a declared deviation for wall-clock, with a
same-device CUDA settling/nonabelian pair as a device-robustness bonus.
Commit under test: 7906d16. Raw: `rungA_nonabelian_s5.json`,
`rungA_dose_s5.json`, `rungA_diag_control_s5.json`,
`rungA_settling_anchor_s5.json`.

## Primary — noisy-S₅, 3 seeds (MPS)

| L_eval | nonabelian n_h=4 | GRU (same device) | settling (logged) |
| ---: | ---: | ---: | ---: |
| 32 | 1.000 ± 0.000 | 0.971 ± 0.011 | 0.997 ± 0.001 |
| 48 | 0.992 ± 0.009 | 0.774 ± 0.052 | 0.962 ± 0.012 |
| 64 (2×) | **0.938 ± 0.060** | 0.527 ± 0.052 | 0.891 ± 0.018 |
| 96 | 0.634 ± 0.277 | 0.176 ± 0.052 | 0.664 ± 0.036 |
| 128 (4×) | 0.382 ± 0.387 | 0.057 ± 0.024 | 0.486 ± 0.032 |

Per-seed @64: 0.857 / 0.957 / 1.000. Per-seed @128: 0.049 / 0.172 / 0.924.
The same-device GRU (0.527±0.052 @64) reproduces the logged GRU band
(0.549±0.055), validating comparison against logged settling.

## Verdict against the pre-registered bars: **GO** (all three legs)

1. Mean @2× = 0.938 ≥ 0.80. ✓
2. Above the GRU at every extrapolation length; ±1σ separation at L=64
   (0.878 vs 0.579). ✓
3. Within 0.05 of settling @2× — exceeded: 0.938 vs 0.891 (above, not just
   within). ✓

## Controls

**Dose-response (CUDA, seed 0)** — the mechanism signature, exactly as
predicted by the reflection-count argument (transposition = 1, 5-cycle = 4):

| L_eval | n_h=1 | n_h=2 | n_h=4 |
| ---: | ---: | ---: | ---: |
| 64 | 0.326 | 0.566 | 1.000 |
| 96 | 0.019 | 0.083 | 0.996 |
| 128 | 0.010 | 0.021 | **0.956** |

n_h=1,2 cliff-collapse (cannot express the 5-cycle transition per step);
n_h=4 tracks to 4× nearly perfectly. Params rise 273k→308k→378k with n_h, but
the 45× jump at @128 from n_h=2→4 tracks expressibility, not the 23% param
increase — and the 435k-param GRU fails regardless.

**Diagonal void-check (CUDA, seed 0)** — same architecture, commutative
transitions: 0.991 @32 → 0.470 @64 → 0.051 @96 → 0.015 @128. It does NOT
length-generalize (near-chance by 3×; chance 0.008), so the rung is not void.
Honest caveat: its shoulder is softer than the logged diag-SSM's (0.470 vs
0.011 @64) — input-dependent diagonal gating buys partial tracking longer than
a vanilla diagonal mixer, but collapses all the same. Non-commutativity, not
input-dependence, is what carries extrapolation.

**Same-device settling anchor (CUDA, seed 0):** settling 0.879 @64 / 0.441
@128 (consistent with logged); nonabelian on the same device 1.000 @64 /
0.956 @128 — above settling at every length, ~2.2× at 4×.

## Honest caveats (on the record)

- **High seed variance at 3–4×** (±0.277 @96, ±0.387 @128). When training
  locks into the group algebra the scan model extrapolates nearly perfectly
  (0.924–0.956 @128, double settling's best); when it drifts, it degrades
  below settling (0.049, 0.172). The pre-registered bars were at 2×, where
  variance is modest (±0.060) — the GO is earned there. Stabilizing the
  4× regime (init, curriculum, or orthogonality regularization) is open work.
- **Seed 0 @128 (0.049) fell below its GRU (0.090)** — one seed, worst
  length; reported because it happened.
- **Wall-clock at train L=32 favors settling on both devices** (CUDA: 107s vs
  229s) — at toy lengths everything is launch-overhead-dominated and settling's
  32 iterations are cheap. The cost claim is structural: BPTT depth L vs
  ceil(log2 L) scan rounds. Measured scaling curve to follow
  (`scan_walltime.py`); at L=32, log-depth buys nothing and the numbers say so.

## What this changes

The open problem logged in `results/serial_depth_lengthgen.md` — "non-TC0
expressivity at sub-sequential (parallelizable) training cost is unsolved
here" — now has a validated candidate: input-dependent Householder-product
transitions composed by parallel prefix scan match (at 2×) and on good draws
double (at 4×) the settling capability, with training-graph depth
O(log L) instead of O(L). Settling remains the existence proof and the more
seed-stable extrapolator at 4×; the scan is the efficient construction.

---

# Supplements (pre-registered secondaries, appended 2026-07-02)

## A₅ replication — ceiling, zero variance (GB10, 3 seeds)

| L_eval | nonabelian n_h=4 | GRU | settling (logged, single run) |
| ---: | ---: | ---: | ---: |
| 32–64 | **1.000 ± 0.000** | 0.999→0.909 | 0.997→0.985 |
| 96 | **1.000 ± 0.000** | 0.675 ± 0.170 | 0.921 |
| 128 (4×) | **1.000 ± 0.000** | 0.477 ± 0.209 | 0.807 |

Perfect tracking to 4× on every seed at every length (`rungA_a5.json`). The
S₅-vs-A₅ variance asymmetry motivated the parity ablation — whose verdict is
**REFUTED** (see `rungA_parity_fix.md`): failed S₅ seeds have perfect det-sign
parity but decay-shaped accuracy, relocating the live hypothesis from parity
access to magnitude saturation.

## Measured training-step wall-clock (GB10, uncontended rerun)

fwd+bwd+step median seconds, B=8, d=128 (`rungA_walltime.json`):

| L | scan (log-depth) | settling (iters=L) | ratio |
| ---: | ---: | ---: | ---: |
| 32 | 0.006 | 0.010 | 1.8× |
| 128 | 0.020 | 0.043 | 2.1× |
| 512 | 0.113 | 0.397 | 3.5× |
| 1024 | 0.241 | 1.445 | **6.0×** |

The ratio grows monotonically with L, as the O(L)-vs-O(log L) BPTT depth
predicts. At L=2048 the settling arm was cut after unified-memory thrash: its
BPTT activation footprint is ~120 GB at that length (2048 sequential
iterations, each storing full-sequence activations) versus ~3 GB for the
scan — the sequential-depth **memory** wall, measured by collision rather
than argued. The earlier caveat stands for toy lengths: at L=32 the ratio is
only 1.8× and launch overhead dominates; the structural claim is the trend.
