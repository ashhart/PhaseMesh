# PhaseMesh LM v0.2 Ladder

Status date: 2026-06-24.

This is the next coherent build target. It turns the current pieces into one
measurable PhaseMesh LM track without claiming general LLM parity before the
checkpoint earns it.

## Goal

Build a trainable PhaseSSM language model with:

- an honest bpc/perplexity comparison against a matched transformer,
- a talkable checkpoint CLI that reports generation latency,
- an optional phase-binding memory branch for recall-heavy tasks,
- a measured kernel status table for speed work,
- clear scorecard rows separating quality, recall, and speed.

## Green Gates

| Gate | Command | Green Criteria |
| --- | --- | --- |
| Quality scout | `python -m phase_ssm.train --model phasessm ...` | Final `val_bpc` recorded beside a matched transformer |
| Talkable checkpoint | `python -m phase_ssm.chat --checkpoint runs/ssm/best.pt "prompt" --json` | Loads, samples, reports `tokens_per_s`, `val_bpc`, and limits |
| Phase memory integration | `python -m phase_ssm.train --model phasessm --use-phase-memory ...` | Trains as a first-class LM config, not only an MQAR script |
| MQAR differentiator | `python -m phase_ssm.mqar_sweep ...` | SSM+memory beats SSM-only and is compared to transformer |
| Short-prefill speed | `python -m phase_ssm.shortbench ...` | Records crossover and rejects/accepts FFT threshold from data |
| Kernel rung | `results/kernel_status.md` | States what is correct, what is fused, and what is still missing |

## Current Working Commands

Quality training:

```bash
python -m phase_ssm.train \
  --model phasessm \
  --out runs/ssm-130m \
  --d-model 1088 \
  --n-layers 12 \
  --state-dim 64 \
  --expand 1 \
  --d-ff-mult 2 \
  --seq 1024 \
  --batch 64 \
  --steps 5000 \
  --train-log-interval 100
```

Phase-memory LM variant:

```bash
python -m phase_ssm.train \
  --model phasessm \
  --out runs/ssm-memory-scout \
  --d-model 384 \
  --n-layers 6 \
  --state-dim 64 \
  --use-phase-memory \
  --mem-heads 4 \
  --mem-head-dim 32 \
  --seq 512 \
  --batch 32 \
  --steps 2000
```

Talk to a checkpoint:

```bash
python -m phase_ssm.chat \
  --checkpoint runs/ssm-130m/best.pt \
  --max-tokens 120 \
  --temperature 0.8 \
  --json \
  "PhaseMesh is"
```

Short-context speed boundary:

```bash
python -m phase_ssm.shortbench \
  --backends fft,triton,auto \
  --auto-threshold 0 \
  --ssm-dtype bfloat16 \
  --lengths 1024 2048 4096 8192 16384 32768 65536
```

## Claim Boundary

This ladder can prove a better PhaseMesh LM checkpoint, a useful recall branch,
or a faster mixer kernel. It does not by itself prove general LLM parity,
assistant usefulness, or a replacement for transformer inference.
