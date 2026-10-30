# PhaseMesh / PhaseSSM Current Scorecard

Status date: 2026-06-24.

This file is the public truth source for measured claims. It separates the
long-context PhaseSSM engine, PhaseMesh-native binding probes, and specialist
trace retrieval surfaces from broader research goals. It does not claim general
LLM parity, a Qwen replacement, or ChatGPT-style usefulness.

## Claim Boundary

| Track | What Is Measured | What It Is Not |
| --- | --- | --- |
| PhaseSSM | Fixed-state sequence mixing speed, memory, and decode behavior on matched synthetic mixer stacks | A general assistant, a quality-equivalent LLM, or a full-block fused model |
| Role-binding PhaseMesh | Retrieval under swapped-role lexical decoys, with role/bag/whole ablations and classical baselines | Open-ended reasoning or language generation |
| PhaseChat | Fast retrieval over poured teacher traces with abstention outside coverage | Raw Qwen inference or weight-equivalent model conversion |
| Phase field runtime | Inspectable resonance, verifier control, topology persistence, and q8 artifacts | A pretrained foundation model |

## What Is Green

| Axis | Current Result | Status |
| --- | --- | --- |
| Long-context prefill | Mixer-level recurrent Triton crosses flash attention at 64k, then reaches `5.44x` at 256k | Green |
| 1M context | PhaseSSM holds `216k tok/s` through `1,048,576` tokens, using `8.59 GB` | Green |
| Decode latency | PhaseSSM stays about `0.51 ms/token`; attention grows to `4.78 ms/token` at 64k | Green |
| Decode ratio | `9.35x` faster than the synthetic attention KV path at 64k | Green |
| Mixed-length batches | `1.50x` effective throughput and `0.30 GB` vs `12.52 GB` on batch-32 uniform lengths | Green |
| Trainable LM core | `phase_ssm.train` trains PhaseSSM and matched transformer checkpoints on identical byte data | Green |
| Talkable checkpoint CLI | `python -m phase_ssm.chat --checkpoint runs/ssm-130m-scout "prompt"` loads and samples a completed checkpoint | Green |
| 131.7M quality scout | Completed with best `val_bpc=1.2568` at step 4750 | Green |
| Training backend probe | `fixed_triton` helps medium models but only reaches `14.8k tok/s` at 131.7M; skip ceiling is `21.2k tok/s` | Measured |
| O(L) kernel correctness | Complex chunked, real-pair chunked, real-pair recurrent, recurrent Triton, and chunked Triton are covered by equivalence tests | Green |

## PhaseMesh-Native Binding Rows

| Demo | Current Result | Read |
| --- | --- | --- |
| Hard role binding | Role phase holds `0.979` accuracy at 30% arbitrary corruption on swapped-role decoys | Strongest current PhaseMesh-native demo |
| Role ablations | Bag phase `0.438`, whole phase `0.054` at the same 30% row | The win is attributable to role binding, not exact token overlap |
| Lexical baselines | BM25 `0.492`, vector `0.446`, n-gram `0.487` at 30% | Lexical baselines see the same words in wrong roles |
| Natural-doc retrieval | PhaseMesh is smaller but BM25/vector/n-gram remain stronger on the generated task | Useful fuzzy attractor path, not a universal retrieval win |

## What Is Not Claimed Yet

| Axis | Current State | Needed To Claim |
| --- | --- | --- |
| General LLM quality | Not proven | Train and evaluate a larger PhaseSSM checkpoint against a matched transformer |
| 130M transformer comparison | Not run in this artifact | Matched 130M transformer run with same data/config budget |
| ChatGPT-style usefulness | Not proven | A trained checkpoint with qualitative and held-out task evals |
| Full-block fused speed | Not proven | Fuse recurrence/normalization/projection/gating or build a row-blocked scan that improves practical short-context utilization |
| Fused RMSNorm/projection kernel | Not implemented | New row-blocked fused kernel shape; the current recurrent kernel is per channel and fuses recurrence/readout only |
| 100x training speedup | Not available from the SSM scan alone | Needs full-block fusion, optimizer fusion, and/or multi-GPU |
| Short-context prefill | Measured | Attention wins below 64k; recurrent Triton crosses at 64k; FFT is rejected as the short-prefill path |
| Broad transformer "slap" | Not available | Current evidence is a real architecture with narrow wins and honest gaps, not an across-the-board transformer replacement |

## Current Mixer-Level Efficiency Table

These are matched synthetic mixer-stack rows. They do not include a full
language-model block with fused RMSNorm, projections, gates, FFN, tokenizer, or
sampling loop.

| Metric | PhaseSSM | Attention Baseline | Ratio |
| --- | ---: | ---: | ---: |
| Prefill at 64k | 217k tok/s | 155k tok/s | 1.40x |
| Prefill at 131k | 214k tok/s | 80k tok/s | 2.69x |
| Prefill at 256k | 216k tok/s | 40k tok/s | 5.44x |
| Decode at 64k | 0.511 ms/token | 4.782 ms/token | 9.35x |
| Mixed batch throughput | 216k tok/s | 144k tok/s | 1.50x |
| Mixed batch memory | 0.30 GB | 12.52 GB | 41.7x less |

## Current Quality Scout

| Metric | PhaseSSM 130M Scout |
| --- | ---: |
| Parameters | 131.7M |
| Best validation bpc | 1.2568 |
| Best step | 4750 |
| Final validation bpc | 1.2603 |
| Typical interval throughput | ~12.6k tok/s |

Artifact: `results/ssm_130m_scout.md`.

Read: this is a completed PhaseSSM checkpoint, not a matched transformer
comparison. The next quality gate is the matching transformer run.

## How To Reproduce The Main Rows

```bash
python -m phase_ssm.effbench \
  --backend triton \
  --lengths 8192 32768 65536 131072 262144

python -m phase_ssm.decodebench \
  --lengths 2048 8192 32768 65536 \
  --steps 32 \
  --reps 3

python -m phase_ssm.effbench \
  --backend triton \
  --seq-dist uniform \
  --batch 32 \
  --min-seq 512 \
  --max-seq 32768
```

FFT ablation:

```bash
python -m phase_ssm.effbench \
  --backend fft \
  --lengths 1024 2048 4096 8192 16384 32768
```

Safe current auto route for speed benches:

```bash
python -m phase_ssm.effbench \
  --backend auto \
  --auto-threshold 0 \
  --ssm-dtype bfloat16 \
  --lengths 1024 2048 4096 8192 16384 32768 65536 131072
```

Threshold experiment, for reproducing the rejected FFT-short route:

```bash
python -m phase_ssm.effbench \
  --backend auto \
  --auto-threshold 32768 \
  --ssm-dtype bfloat16 \
  --lengths 1024 2048 4096 8192 16384 32768 65536 131072
```

Read: the NVIDIA GB10 sweep rejected FFT as the below-64k speed path. Recurrent Triton
is the best current PhaseSSM backend from 1k to 64k, but attention still wins
below 64k and PhaseSSM crosses at 64k. The remaining constant-factor work is a
fused block kernel and a row-blocked parallel scan; those need CUDA kernels, not
another Python routing tweak.

Real-pair note: the PyTorch real-pair paths are bit-equivalent to the complex
oracle and make the kernel fusable in normal real lanes. The current recurrent
Triton state math remains fp32 for stability, so real-pair correctness is
unblocked but full bf16 fused speed is still a kernel-engineering task.

For the measured below-64k routing decision, use:

```bash
python -m phase_ssm.shortbench \
  --backends fft,triton,auto \
  --auto-threshold 32768 \
  --ssm-dtype bfloat16 \
  --lengths 1024 2048 4096 8192 16384 32768 65536 \
  --out results/short_prefill_sweep.json \
  --markdown results/short_prefill_sweep.md
```

`shortbench` writes the measured row table and a recommended auto threshold.
The current NVIDIA GB10 artifact is `results/short_prefill_sweep.md`; it reports
`recommended_auto_threshold=None` and `first_best_ssm_win_vs_attention=65536`.
The kernel rung summary is `results/kernel_status.md`.

## How To Talk To A Trained PhaseSSM Checkpoint

Train:

```bash
python -m phase_ssm.train \
  --model phasessm \
  --out runs/ssm-small \
  --steps 10000 \
  --seq 512 \
  --batch 32
```

Generate:

```bash
python -m phase_ssm.chat \
  --checkpoint runs/ssm-small \
  --max-tokens 120 \
  --temperature 0.8 \
  "PhaseMesh is"
```

## Read

PhaseSSM now has a real long-context mixer engine and a direct checkpoint
conversation path. The strongest current result is not "better chatbot"; it is
"fixed-state sequence mixing avoids the long-context attention wall in the
mixer benchmark." The next proof run is quality: train a larger checkpoint and
put bpc/perplexity beside the speed and memory table. The next speed run is
kernel engineering: full-block fusion or a row-blocked short-prefill scan.
