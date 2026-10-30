# Input-dependent serial depth: S₅ length generalization (validated)

Status date: 2026-06-27. **Strongest current capability result.** Logged, seed-confirmed, positive-control-backed.

## Claim

A **settling recurrence** (input-dependent serial depth — the per-step operator is iterated, depth ≈ sequence length) **length-generalizes on S₅ state-tracking where fixed-depth models cannot**, and it **beats a plain RNN (GRU)**, with the advantage widening as you extrapolate further.

This is a capability/expressivity result, not an efficiency or quality claim. It does not claim language quality or a transformer replacement.

## Setup

- Task: **noisy-S₅** (`phase_ssm/equilibrium.py:make_noisy_batch`) — running composition in the non-solvable group S₅ with the state signal sparse and buried in distractor tokens (`gen_prob=0.5`, `ND=30`). Last-token accuracy = full running-state accuracy; chance ≈ 1/120 ≈ 0.008.
- Train at length **32**, evaluate at 32–128 (up to 4× train length).
- Models: settling vs **positive controls** — GRU (non-diagonal recurrence, should extrapolate), transformer + diagonal-SSM (fixed-depth = TC0, should collapse past training length). d=128, 6000 steps.

## Results

Single run, 6000 steps, finer grid (`equilibrium.py --noisy`):

| eval length | settling | transformer | diag-SSM | GRU |
| ---: | ---: | ---: | ---: | ---: |
| 32 (train) | 0.996 | 0.967 | 0.988 | 0.983 |
| 48 (1.5×) | 0.943 | 0.020 | 0.036 | 0.825 |
| 64 (2×) | 0.866 | 0.013 | 0.011 | 0.581 |
| 96 (3×) | 0.639 | 0.011 | 0.008 | 0.232 |
| 128 (4×) | 0.413 | 0.005 | 0.007 | 0.084 |

3-seed confirmation of the contested comparison (`phase_ssm/s5_seeds.py --seeds 3`):

| eval length | settling (mean±std) | GRU (mean±std) | gap |
| ---: | ---: | ---: | ---: |
| 32 | 0.997 ± 0.001 | 0.978 ± 0.008 | +0.019 |
| 48 | 0.962 ± 0.012 | 0.816 ± 0.044 | +0.145 |
| 64 (2×) | 0.891 ± 0.018 | 0.549 ± 0.055 | +0.342 |
| 96 | 0.664 ± 0.036 | 0.192 ± 0.068 | +0.472 |
| 128 | 0.486 ± 0.032 | 0.076 ± 0.022 | +0.410 |

## Pre-registered verdict: GO (met)

Bar set before the run: settling **> 0.8 at 2×** AND beats fixed-depth by a wide margin. Result: settling@2× = **0.891 ± 0.018** (all three seeds ≥ 0.869), vs transformer/diag-SSM at **chance**. Both legs met. The settling > GRU gap has **non-overlapping error bars at every extrapolation length** and widens with distance — so this is the adaptive-depth structure adding real value, not a reinvented RNN.

## Interpretation

- Fixed-depth (transformer, diag-SSM) **cliff-collapse to chance the instant they exceed training length** — the expected TC0 / fixed-depth wall on a non-solvable-group task.
- Recurrent-depth extrapolates; **settling extrapolates substantially better than the GRU**, reliably.

## Adaptive compute (learned halting)

A PonderNet-style halt head (`phase_ssm/halting_settle.py`) replaces hand-set depth, plus an explicit ponder cost on expected depth. Without a real cost the model never halts (more settling monotonically helps → depth pins at the cap). With a cost, depth becomes adaptive:

| eval length | ponder=0.1 (acc / depth) | ponder=0.5 (acc / depth) |
| ---: | ---: | ---: |
| 32 | 0.999 / 192 | 1.000 / 30.9 |
| 48 | 0.979 / 192 | 0.671 / 94.9 |
| 64 | 0.949 / 192 | 0.731 / 152.5 |
| 96 | 0.821 / 192 | 0.727 / 187.0 |
| 128 | 0.670 / 192 | 0.531 / 191.7 |

At ponder=0.5 the **learned depth scales monotonically with input length (30.9 → 191.7)** — short inputs halt early at full accuracy (L=32: 1.000 in ~31 steps), long inputs use more. Genuine learned adaptive computation (the honest counterpart to the refuted "energy-proportional" demo, which had flat depth). **Cost:** a real accuracy-vs-penalty tradeoff — strong penalty buys adaptivity but dips mid-length accuracy (a calibration wrinkle at L=48); weak penalty keeps accuracy but gives no adaptivity. Sweet spot is between, not yet tuned.

## Honest caveats / not claimed

- **Synthetic noisy-S₅, not language.** Transfer to real sequence modeling is unproven.
- **The adaptive-compute leg is demonstrated but not free** (see above): it requires a ponder cost strong enough to trade some accuracy, and the halt calibration is not yet tuned. The base length-gen result does not depend on it (that uses hand-set `iters = L`).
- **Active research area** (length generalization, state-tracking, recurrent-depth / looped models). This is a real, clean result — not unprecedented.

## Language transfer: settling-specificity does NOT carry (negative result)

Tested whether the S₅ settling advantage appears on byte-level text8 length-extrapolation (train seq 256, eval bpc at 256/512/1024), settling vs diag-SSM vs transformer (`phase_ssm/lang_lengthgen.py`).

| model | params | bpc@256 | bpc@512 | bpc@1024 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| settling | 3.0M | 1.604 | 1.602 | 1.583 | −0.021 |
| diag-SSM | 11.5M | 1.562 | 1.510 | 1.513 | −0.049 |

Both SSM-class models extrapolate cleanly (bpc holds/improves at 4× context). **But settling shows no advantage over the plain diag-SSM on language** — unlike the dramatic S₅ separation (0.87 vs 0.01). The settling-specific magic **does not transfer to language**; on text it behaves like an ordinary SSM (and a plain SSM matches it at ¼ the params). Transformer baseline run was cut. **Conclusion:** serial-depth/settling is a real result on algorithmic / state-tracking length-gen — but SSM-class, not settling-special, on language. The make-or-break language transfer is negative for the ambitious "new language architecture" claim.

## Reproduce

```bash
python -m phase_ssm.equilibrium --noisy --train-L 32 --eval-Ls 32 48 64 96 128 --steps 6000
python -m phase_ssm.s5_seeds --seeds 3 --steps 6000
```
