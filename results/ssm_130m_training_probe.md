# PhaseSSM 131.7M Training Probe

Status date: 2026-06-24.

## Why This Exists

The first 131.7M run printed `~2k tok/s` because the trainer only reported throughput at eval boundaries. On a tiny dry run, that number mixed training, evaluation, checkpointing, and startup time. It was not the real optimization-step speed.

The trainer now supports train-only logging:

```bash
--train-log-interval 100
--skip-initial-eval
```

## Model Config

This is the closest current PhaseSSM config to 130M parameters:

```bash
--d-model 1088
--n-layers 12
--state-dim 64
--expand 1
--d-ff-mult 2
--short-conv 4
```

Parameter count:

```text
131,688,256
```

## Timing Probe

Command shape:

```bash
python -m phase_ssm.train \
  --model phasessm \
  --d-model 1088 \
  --n-layers 12 \
  --state-dim 64 \
  --expand 1 \
  --d-ff-mult 2 \
  --seq 1024 \
  --batch 64 \
  --steps 20 \
  --eval-interval 0 \
  --train-log-interval 5 \
  --skip-initial-eval \
  --warmup 600 \
  --out runs/ssm-130m-timing
```

Observed train-only log:

| Step | Loss | Train tok/s | Step time |
| ---: | ---: | ---: | ---: |
| 5 | 4.8758 | 12.9k | 4.98s |
| 10 | 3.4846 | 14.3k | 4.96s |
| 15 | 2.7552 | 14.3k | 4.97s |
| 20 | 2.5310 | 14.3k | 4.96s |

## Read

The 131.7M train path is not using the recurrent Triton inference kernel. It trains through the differentiable FFT convolution path in `phase_ssm/model.py`, so it is expected to be much slower than the `216k tok/s` inference benchmark.

The fixed estimate is roughly:

```text
30,000 steps * ~4.96s/step = ~41.3 hours
plus eval/checkpoint overhead
```

The current full run uses train-only logging every 100 steps and skips the misleading step-0 eval:

```bash
python -m phase_ssm.train \
  --model phasessm \
  --d-model 1088 \
  --n-layers 12 \
  --state-dim 64 \
  --expand 1 \
  --d-ff-mult 2 \
  --seq 1024 \
  --batch 64 \
  --steps 30000 \
  --eval-interval 500 \
  --eval-iters 20 \
  --train-log-interval 100 \
  --skip-initial-eval \
  --warmup 600 \
  --out runs/ssm-130m
```

Monitor:

```bash
ssh pgx 'tail -f /home/pgx/phase_mesh_scale/repo/runs/ssm-130m/train.log'
```

Stop:

```bash
ssh pgx 'docker stop phasessm-130m'
```
