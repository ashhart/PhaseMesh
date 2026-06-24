# PhaseSSM Training Backend Bench

Status date: 2026-06-24.

## Question

Could the 131.7M quality run get a 100x speedup just by replacing the SSM training backend?

Short answer: no. The current 131.7M architecture is dominated by projections, gates, FFNs, convolution, logits, and optimizer work. The SSM backend matters, but it is not a 100x bottleneck at this width.

## Backends Tested

| Backend | Meaning |
| --- | --- |
| `fft` | Current fully trainable complex FFT/autograd backend |
| `real_chunked` | Differentiable real-pair chunked scan in PyTorch |
| `fixed_triton` | Recurrent Triton forward plus exact input-gradient backward; oscillator kernel is frozen |
| `skip` | Diagnostic ceiling: bypass SSM mixing and keep the rest of the block |

## Medium Shape

Config:

```bash
--d-model 384
--n-layers 6
--state-dim 64
--expand 2
--d-ff-mult 3
--seq 512
--batch 16
```

| Backend | Params | Mean step | Train tok/s | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `fft` | 14.6M | 0.239s | 34.3k | 3.86 GB |
| `fft --compile` | 14.6M | 0.219s | 37.4k | 3.02 GB |
| `real_chunked` | 14.6M | 6.833s | 1.2k | 43.45 GB |
| `fixed_triton` | 14.6M | 0.163s | 50.2k | 1.82 GB |

Read: naive differentiable chunking is a dead end. `fixed_triton` is useful at medium scale, giving about `1.46x` over `fft` and much lower memory.

## 131.7M Shape

Config:

```bash
--d-model 1088
--n-layers 12
--state-dim 64
--expand 1
--d-ff-mult 2
--seq 1024
--batch 64
```

| Backend | Params | Mean step | Train tok/s | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `fft` | 131.7M | ~4.96s | ~14.3k | ~68.8 GB process memory |
| `fixed_triton` | 131.7M | 4.434s | 14.8k | 46.43 GB |
| `skip` | 131.7M | 3.092s | 21.2k | 49.80 GB |

Read: even deleting SSM mixing entirely only reaches `21.2k tok/s`. So a perfect SSM scan/backward kernel cannot give 100x on this architecture. The remaining bottleneck is the full LM block and optimizer, not just the oscillator scan.

## What This Means

The near-term safe speed path is:

1. Use `--train-log-interval` and `phase_ssm.trainbench` for honest timing.
2. Use `fixed_triton` only for frozen-SSM experiments where that approximation is acceptable.
3. Do not use `real_chunked` for training.
4. For large quality runs, either accept the `fft` training speed or change the architecture/training system.

The real 100x path is larger than one kernel:

- fused trainable SSM forward/backward
- fused MLP/projection/gate kernels
- activation checkpointing or reversible blocks
- optimizer fusion
- multi-GPU data parallel or sequence parallel training
- architecture changes that move parameters away from dense FFNs if the target is throughput

## Commands

Medium benchmark:

```bash
python -m phase_ssm.trainbench \
  --ssm-backend fft \
  --d-model 384 \
  --n-layers 6 \
  --state-dim 64 \
  --expand 2 \
  --d-ff-mult 3 \
  --seq 512 \
  --batch 16
```

131.7M benchmark:

```bash
python -m phase_ssm.trainbench \
  --ssm-backend fixed_triton \
  --d-model 1088 \
  --n-layers 12 \
  --state-dim 64 \
  --expand 1 \
  --d-ff-mult 2 \
  --seq 1024 \
  --batch 64
```
