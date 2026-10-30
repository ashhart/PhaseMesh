# Input-Dependent Serial Depth for Length-Generalizing State Tracking

Research note. Status date: 2026-06-27. Scope: a narrow, honest empirical result on synthetic state-tracking. **Not** a language-model or general-architecture claim — see Claim Boundary and Negative Results.

## Summary

A recurrence with **input-dependent serial depth** — iterate a non-diagonal, nonlinear "settling" operator a data-dependent number of times — length-generalizes on non-solvable-group state tracking (S₅) where fixed-depth models cannot:

- **Transformers and diagonal SSMs collapse to chance** the moment the sequence exceeds the training length; the settling recurrence degrades gracefully and stays far above chance out to 4× train length.
- It **beats a plain RNN (GRU)** with a margin that *widens* with extrapolation distance — so the advantage is the adaptive-depth structure, not just "being recurrent." Confirmed across 3 seeds with non-overlapping error bars.
- Depth can be made **adaptive**: a learned halting head scales the number of serial steps with input length (≈31 steps for length 32, ≈192 for length 128), recovering accuracy while *choosing* its own compute — at a real accuracy-vs-compute tradeoff.
- The separation holds on a **second non-solvable group (A₅)**, so it is not S₅-specific.

It does **not** transfer to byte-level language as a settling-specific advantage (a plain diagonal SSM matches it there). The contribution is specifically about *algorithmic / state-tracking length generalization*.

## Thesis

Fixed-depth models — transformers and diagonal (Mamba-style) SSMs — are in TC0. The word problem of a **non-solvable** group (S₅, A₅) is NC1-complete, hence outside TC0, so no fixed-depth model can solve it for growing length (Merrill et al., "The Illusion of State in SSMs", 2024). A model whose **serial depth grows with the input** has effective depth set by iteration count, not layer count, and can escape TC0. Three capabilities then fall out of one property:

- **Expressivity** — it can track non-solvable state that fixed-depth provably cannot.
- **Length generalization** — a longer input gets more steps of the *same* learned operator, so behavior learned at length L applies at kL.
- **Adaptive compute** — the step count *is* the compute, allocated by input difficulty; a fixed-depth model structurally cannot do this.

The operator must be **nonlinear and non-diagonal**: a linear/diagonal settle has a closed-form fixed point (no depth gained), which is exactly why diagonal SSMs stay in TC0.

## Setup

**Task: noisy-Sₙ.** Running composition in a non-solvable group (S₅ = 120 elements; A₅ = 60), with the state signal *sparse and buried in distractor tokens* (a state-changing generator appears with probability `gen_prob=0.5`; the rest are distractors). Last-token accuracy = full running-state accuracy. Chance ≈ 1/|G|. Train at length 32, evaluate at 32–128.

**Models (matched ~d=128).** settling (iterated non-diagonal nonlinear cell) vs **positive controls**: transformer and diagonal-SSM (fixed-depth — should collapse past training length), GRU (non-diagonal recurrence — should extrapolate).

## Result 1 — Length generalization (S₅)

Single run, 6000 steps:

| eval length | settling | transformer | diag-SSM | GRU |
| ---: | ---: | ---: | ---: | ---: |
| 32 (train) | 0.996 | 0.967 | 0.988 | 0.983 |
| 48 (1.5×) | 0.943 | 0.020 | 0.036 | 0.825 |
| 64 (2×) | 0.866 | 0.013 | 0.011 | 0.581 |
| 96 (3×) | 0.639 | 0.011 | 0.008 | 0.232 |
| 128 (4×) | 0.413 | 0.005 | 0.007 | 0.084 |

3-seed confirmation (settling vs GRU; non-overlapping error bars at every extrapolation length, gap widens with distance):

| eval length | settling (mean±std) | GRU (mean±std) |
| ---: | ---: | ---: |
| 48 | 0.962 ± 0.012 | 0.816 ± 0.044 |
| 64 | 0.891 ± 0.018 | 0.549 ± 0.055 |
| 96 | 0.664 ± 0.036 | 0.192 ± 0.068 |
| 128 | 0.486 ± 0.032 | 0.076 ± 0.022 |

Fixed-depth models cliff-collapse to chance; recurrent depth extrapolates; settling clearly beats the GRU.

## Result 2 — Adaptive compute (learned halting)

A PonderNet-style halt head plus an explicit ponder cost makes depth data-dependent. Without a cost the model never halts; with one (ponder=0.5), **learned depth scales monotonically with input length**:

| eval length | accuracy | mean depth |
| ---: | ---: | ---: |
| 32 | 1.000 | 30.9 |
| 64 | 0.731 | 152.5 |
| 128 | 0.531 | 191.7 |

Short inputs halt early at full accuracy; long inputs use more — genuine learned adaptive computation. Cost: an accuracy-vs-penalty tradeoff that is not yet tuned.

## Result 3 — Generality on a second non-solvable group (A₅)

Same noisy state-tracking on the alternating group A₅ (non-solvable, 60 elements), 6000 steps:

| eval length | settling | transformer | diag-SSM | GRU |
| ---: | ---: | ---: | ---: | ---: |
| 32 (train) | 0.999 | 0.918 | 0.972 | 1.000 |
| 48 (1.5×) | 0.997 | 0.013 | 0.014 | 0.994 |
| 64 (2×) | 0.985 | 0.021 | 0.014 | 0.932 |
| 96 (3×) | 0.921 | 0.021 | 0.017 | 0.657 |
| 128 (4×) | 0.807 | 0.015 | 0.010 | 0.385 |

The S₅ pattern replicates cleanly on a different non-solvable group: transformer and diag-SSM collapse to chance (≈1/60) past the training length; settling extrapolates best (0.807 at 4×, *even stronger* than on S₅), again beating the GRU with a margin that widens with distance. The effect is a property of the non-solvable / serial-depth class, not specific to S₅.

## Result 4 — Why the advantage disappears: a composition-depth dial

To explain the language-transfer negative rather than just report it, we built a controllable dial: a periodic RESET token forces the tracked S₅ state back to identity every `K` positions, capping the composition depth actually required at `K` regardless of total sequence length `L`. `K ≥ L` reduces exactly to the unbounded task (Result 1); small `K` bounds depth the way natural text plausibly does even with long surface dependencies. Train L=32, eval to 128, settling vs diag-SSM vs transformer (`phase_ssm/composition_depth.py`):

| K | settling | diag-SSM | transformer | gap (settle − diag-SSM) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1.000 | 0.939 | 0.369 | +0.061 |
| 16 | 0.999 | 0.092 | 0.114 | +0.907 |
| 10000 (no reset) | 0.435 | 0.013 | 0.004 | +0.422 |

`K=10000` reproduces Result 1 closely (internal consistency check). The gap is **not monotonic in K** — smallest at K=4, largest at K=16. The mechanism this points to is sharper than "shallow depth helps": diag-SSM and transformer are both, in effect, fixed-depth-≈4 models (diag-SSM stacks 4 layers with FFN nonlinearities between the linear temporal mixers). At K=4, required depth fits inside that budget and both fixed-depth models nearly solve it — the gap collapses. At K=16, required depth exceeds that budget — they collapse to near-floor while settling (no such ceiling) stays near-perfect. At K=10000, settling itself degrades from long-range drift while the fixed-depth models are already pinned at floor.

**Refined account of the language-transfer negative:** a fixed-depth model matches a depth-adaptive one whenever a task's true required composition depth fits inside a handful of layers. Byte-level language modeling apparently lives in that regime — which is a more mechanistic, falsifiable claim than "text has bounded depth," and is itself testable (e.g. does the fixed-depth catch-up point shift if `n_layers` is varied?).

Honest wrinkles: the transformer does not fully catch up even at K=4 (0.369 vs diag-SSM's 0.939) — plausibly confounded by evaluating at 4× its training length (RoPE extrapolation) independent of the depth question. Single run per cell; not seed-confirmed like Result 1.

## Negative results (kept on the record)

- **Associative recall is a dead end for this line.** A separate "phase-binding" memory was built to fix MQAR recall, but at a fair config the *additive* memory and a transformer both solve MQAR perfectly (1.000) — the original "additive fails recall" result was an underpowering artifact (the transformer failed there too). An error-correcting (delta-rule) variant only recalls in a conventional DeltaNet-shaped configuration. No novel contribution; abandoned.
- **No language transfer (settling-specific).** On text8 byte-level length-extrapolation, settling shows *no advantage* over a plain diagonal SSM (both extrapolate; the diag-SSM matches at ¼ the params). The S₅ separation does **not** carry to language. The result is algorithmic, not a language architecture.

## Claim boundary

- **Claimed:** on synthetic non-solvable-group state tracking with the state buried in noise, an input-dependent serial-depth recurrence length-generalizes where transformers and diagonal SSMs collapse, beats a plain RNN, and supports learned adaptive depth.
- **Not claimed:** language quality, transfer to natural sequences, a general architecture, or anything about scale. It is an active research area (recurrent-depth / looped models, length generalization, state tracking); this is a clean datapoint, not a first.

## Limitations & open problems

- **Synthetic only.** The honest next test (language) is negative for settling-specificity.
- **The prize / open problem:** the settling operator is *sequential* (slow). The real architectural question — non-TC0 expressivity at **sub-sequential (parallelizable) training cost** — is unsolved here. Without it, this does not scale.
- The adaptive-compute halting tradeoff is untuned.

## Reproduce

```bash
python -m phase_ssm.equilibrium --noisy --train-L 32 --eval-Ls 32 48 64 96 128 --steps 6000   # Result 1
python -m phase_ssm.s5_seeds --seeds 3 --steps 6000                                            # Result 1 (seeds)
python -m phase_ssm.halting_settle --steps 6000 --ponder-cost 0.5                              # Result 2
python -m phase_ssm.a5_lengthgen --train-L 32 --eval-Ls 32 48 64 96 128 --steps 6000           # Result 3
python -m phase_ssm.lang_lengthgen --train-seq 256 --eval-seqs 256 512 1024 --steps 6000       # language negative
```
