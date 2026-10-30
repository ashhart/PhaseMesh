# Rung A follow-up — the parity hypothesis for S₅ 4× seed variance

Status date: 2026-07-01. Status: **pre-registered, not yet run.** Written before
the run; results appended below, pre-registration unedited.

## Hypothesis (topological)

Rung A left one caveat: S₅ extrapolation at 4× has huge seed variance
(0.049/0.172/0.924 @128) while A₅ is at ceiling (1.000±0.000, zero variance).
The structural difference: **A₅'s generators are all even** (rotations,
det +1) — smoothly connected to the identity inside SO(d_h) — while **S₅'s
transposition is odd** (det −1), living in the other connected component of
O(d_h). A rank-1 Householder factor `I − βkkᵀ` has det `1−β ∈ [−1,1]`: it can
only switch parity by dragging its det through 0 — the state-erasing β=1
valley. With the current `identity` init (all factors near I, even side), the
optimizer must carry some factors across that valley to represent the
transposition; whether it crosses cleanly is seed luck → variance. A₅ never
needs the crossing → no variance.

## Predictions (pre-registered)

- **P1 — mechanism (init-independent, correlational):** seeds that extrapolate
  (@128 ≥ 0.8) show det-sign parity alignment — some head's balanced parity
  accuracy (det<0 on transposition tokens, det>0 on even tokens) ≥ 0.95 —
  while failed seeds (@128 ≤ 0.3) show weaker best-head alignment than
  successful ones.
- **P2 — causal fix:** populating the odd component at init (`split`: half
  factors near-reflection raw +4, half near-identity raw −4; or `reflect`: all
  +4) lifts S₅ 3-seed mean @128 to ≥ 0.80 with std < 0.15, vs the identity-init
  arm run on the same device.
- **P3 — control:** A₅ with `split` init stays at ceiling (≥ 0.99 at every
  length, 3 seeds) — the init change must not damage the all-even group.

Outcomes: **CONFIRMED** = P1∧P2∧P3 (mechanism + fix; the Rung A caveat becomes
a finding). **MECHANISM-ONLY** = P1∧P3 but not P2 (the account is right; init
is not the fix — try exact-reflection parameterizations next).
**REFUTED** = P1 fails (successful and failed seeds show similar det-parity
alignment; the topological story is wrong regardless of what init does).

## Protocol

Identical to Rung A primary (noisy-S₅, train L=32, 6000 steps, seeds 0–2,
d=128, H=8, d_h=16, n_h=4, 2 layers) with three β-init arms
{identity, split, reflect}; A₅ control via `a5_nonabelian --beta-init split`.
Device: NVIDIA GB10 GB10 (CUDA), declared — all arms same-device, including the fresh
identity baseline. Diagnostic: `det_parity_diag` (balanced det-sign vs token
parity, per layer, best head and head mean), recorded per seed.

---

# Results (appended after the run; pre-registration above unedited)

Run date: 2026-07-01/02, NVIDIA GB10 GB10. Raw: `rungA_parity_s5.json`,
`rungA_parity_a5_control.json`, det-parity lines in the run log.

Per-seed @128 (best-head det-parity alignment in parentheses):

| arm | seed 0 | seed 1 | seed 2 | mean ± std @128 | mean @64 | mean @96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.956 (1.000) | 0.037 (1.000) | 0.854 (1.000) | 0.616 ± 0.376 | 0.936 | 0.746 ± 0.350 |
| split | 0.089 (0.909) | 0.082 (0.898) | 0.211 (0.951) | 0.127 ± 0.060 | 0.762 | 0.327 |
| reflect | 0.580 (0.988) | 0.543 (0.964) | 0.690 (0.861) | 0.604 ± 0.062 | **0.997** | **0.910 ± 0.030** |

## Verdict: **REFUTED** (per the pre-registered mapping — P1 failed)

- **P1 (mechanism) — FAILED, decisively.** Identity seed 1 has *perfect*
  layer-0 det-sign parity alignment (1.000) and still collapses at 4×
  (0.037). Sign-alignment does not discriminate successful from failed seeds
  — the topological "odd-component access" story, as stated, is wrong.
- **P2 (init fix) — NOT confirmed.** `split` made everything worse
  (0.127 ± 0.060 @128; also worst at 2×) — populating both components
  half-half actively hurts. `reflect` met the variance leg (σ = 0.062 < 0.15)
  but not the mean leg (0.604 < 0.80).
- **P3 (A₅ control) — confirmed.** A₅ with split init: 1.000 at every length,
  all seeds. Init changes don't damage the all-even group.

## What the refutation taught (kept, not spun)

1. **The failure mode is decay, not wrong algebra.** The failed identity seed
   degrades progressively (0.808 @64 → 0.251 @96 → 0.037 @128) with det
   *signs* perfect throughout. Live hypothesis v2: **magnitude saturation** —
   `|1−β|` not pushed close enough to 1, so state amplitude decays with
   length; the sign diagnostic measured parity but not magnitude. Next probe,
   if pursued: per-token `|det|` distributions on good vs bad seeds.
2. **`reflect` init is the practical recommendation regardless.** It is the
   best and most consistent config at 1.5–3× (0.997 @64, 0.910 ± 0.030 @96)
   and eliminates the seed lottery at 4× at some cost to peak. Rung B′ keeps
   `identity` per its pre-committed decision rule (P2 did not confirm), but
   future scan-family work should consider `reflect` its default.
3. Incidental: the A₅ GRU shows its own large 4× variance (0.24/0.75/0.44) —
   seed-lottery extrapolation is not unique to the scan family.

---

# Hypothesis v2 — magnitude saturation (pre-registered before the probe run)

Status: written 2026-07-02, before `--mag-probe` runs. Probe: retrain
{identity, reflect} × 3 seeds on noisy-S₅ (Mini/MPS, protocol unchanged); for
each run record the deepest layer's **carrier head** (best det-sign parity
alignment) and its mean **|det| on generator tokens**. Claim: solutions decay
state amplitude like |det|^(#gens) (~64 generator applications at L=128), so
extrapolation success is governed by how close the carrier's |det| sits to 1 —
signs were already perfect in failures, so magnitude is the discriminating
variable.

- **Binding bar (P-v2):** Spearman rank correlation between acc@128 and
  carrier-head |det|gen across the 6 runs ≥ **0.8**. REFUTED-v2 if ρ < 0.5 or
  the sign is negative; between 0.5 and 0.8 = inconclusive, reported as-is.
- **Soft quantitative check (reported, not binding):** successful runs
  (acc@128 ≥ 0.8) should sit at |det|gen ≥ ~0.99 (amplitude retention
  |det|^64 ≥ ~0.5); failed runs measurably lower. The discrimination band is
  razor-thin by construction (0.98 vs 0.99), which is exactly why it does not
  bind — LayerNorm/readout interactions can shift the usable threshold.
- Device note: MPS, so seeds are fresh draws (CUDA parity-arm outcomes do not
  transfer per-seed); whatever spread appears, the correlation is computed on
  the runs as they land. No selection.

## v2 verdict: **INCONCLUSIVE** (ρ = 0.600; binding band was ≥0.8 / <0.5)

Raw: `rungA_magprobe_s5.json`. Six runs, acc@128 vs deepest-layer carrier
|det|gen: (0.100, 0.583), (0.122, 0.462), (0.188, 0.680), (0.239, 0.772),
(0.443, 0.792), (0.554, 0.615). The first four landed in perfect rank order;
the best run of the batch broke it with a middling |det|. Magnitude is a real
positive correlate — not the discriminator. The soft amplitude bands were
indeed useless (|det|^64 ≈ 0 for every run, successful or not): LayerNorm
rescaling swamps raw amplitude arithmetic, as pre-declared possible.

Kept observations:

1. **Division of labor between layers** (consistent across all 6 runs):
   layer 0 runs aggressive low-|det| projection on generator tokens
   (0.004–0.36 — the erasure valley used as a filter), layer 1 composes
   near-orthogonally (distractor |det| ≈ 0.9–1.0). The transformation-monoid
   reading of the family (Rung B's salvage) shows up unprompted in trained
   solutions.
2. **reflect-init's advantage did NOT replicate on MPS** (0.883 vs identity's
   0.961 @64 here; on CUDA it was 0.997 vs 0.936). The earlier "practical
   default" note is downgraded to: device/batch-inconsistent, unsettled.
3. Thread status: two pre-registered probes, one REFUTED (sign access), one
   INCONCLUSIVE (single-carrier magnitude). The 4× variance mechanism is
   plausibly multi-head/geometric (v3 candidate: joint |det| across the head
   ensemble, or key-subspace alignment). **Parked** — diminishing returns for
   the mainline; the capability results (Rung A GO, A₅ ceiling) are unaffected,
   and stabilization matters only when 4× extrapolation becomes load-bearing.
