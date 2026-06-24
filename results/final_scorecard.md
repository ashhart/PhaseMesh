# PhaseMesh / PhaseSSM Current Scorecard

Status date: 2026-06-24.

## What Is Green

| Axis | Current Result | Status |
| --- | --- | --- |
| Long-context prefill | PhaseSSM crosses flash attention between 32k and 64k, then reaches `5.44x` at 256k | Green |
| 1M context | PhaseSSM holds `216k tok/s` through `1,048,576` tokens, using `8.59 GB` | Green |
| Decode latency | PhaseSSM stays about `0.51 ms/token`; attention grows to `4.78 ms/token` at 64k | Green |
| Decode ratio | `9.35x` faster than the synthetic attention KV path at 64k | Green |
| Mixed-length batches | `1.50x` effective throughput and `0.30 GB` vs `12.52 GB` on batch-32 uniform lengths | Green |
| Trainable LM core | `phase_ssm.train` trains PhaseSSM and matched transformer checkpoints on identical byte data | Green |
| Talkable checkpoint CLI | `python -m phase_ssm.chat --checkpoint runs/ssm/best.pt "prompt"` loads and samples a checkpoint | Green |
| 131.7M quality run | Timing probe shows `~14.3k tok/s` on the fully trainable FFT backend | Measured |
| Training backend probe | `fixed_triton` helps medium models but only reaches `14.8k tok/s` at 131.7M; skip ceiling is `21.2k tok/s` | Measured |

## What Is Not Claimed Yet

| Axis | Current State | Needed To Claim |
| --- | --- | --- |
| General LLM quality | Not proven | Train and evaluate a larger PhaseSSM checkpoint against a matched transformer |
| 130M quality gap | Not run in this artifact | Text8/WikiText mix run with final bpc/perplexity table |
| ChatGPT-style usefulness | Not proven | A trained checkpoint with qualitative and held-out task evals |
| Fused RMSNorm/projection kernel | Not implemented | New row-blocked fused kernel shape; the current recurrent kernel is per channel |
| 100x training speedup | Not available from the SSM scan alone | Needs full-block fusion, optimizer fusion, and/or multi-GPU |

## Current Efficiency Table

| Metric | PhaseSSM | Attention Baseline | Ratio |
| --- | ---: | ---: | ---: |
| Prefill at 64k | 217k tok/s | 155k tok/s | 1.40x |
| Prefill at 131k | 214k tok/s | 80k tok/s | 2.69x |
| Prefill at 256k | 216k tok/s | 40k tok/s | 5.44x |
| Decode at 64k | 0.511 ms/token | 4.782 ms/token | 9.35x |
| Mixed batch throughput | 216k tok/s | 144k tok/s | 1.50x |
| Mixed batch memory | 0.30 GB | 12.52 GB | 41.7x less |

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

PhaseSSM now has a real long-context engine and a direct checkpoint conversation path. The strongest current result is not "better chatbot"; it is "fixed-state sequence mixing avoids the long-context attention wall." The next proof run is quality: train a larger checkpoint and put bpc/perplexity beside the speed and memory table.
