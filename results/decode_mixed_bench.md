# PhaseSSM Decode and Mixed-Length Efficiency Bench

This is the second PhaseSSM efficiency result as of 2026-06-24.

## Claim

On matched synthetic mixer stacks, PhaseSSM decode uses a fixed recurrent state while transformer-style attention decode reads a KV cache that grows with context length. In this benchmark the PhaseSSM step stays flat at about `0.51 ms/token`, while attention decode slows as context grows.

For mixed-length batches, PhaseSSM can process each sequence at its actual length. The attention baseline is padded to the batch maximum length. This measures the practical padding tax in variable-length workloads.

These are sequence-mixer efficiency benchmarks. They do not claim language-model quality parity.

## Environment

- Host: PGX
- GPU: NVIDIA GB10, CUDA capability `sm_121`
- Container: `nvcr.io/nvidia/pytorch:26.01-py3`
- PyTorch: `2.10.0a0+a36e1d39eb.nv26.01.42222806`
- Triton: `3.6.0`

## Decode Configuration

- PhaseSSM stack: `d_model=512`, `layers=8`, `state_dim=64`
- Attention stack: `d_model=512`, `layers=8`, `heads=8`
- Batch: `1`
- Decode steps per measurement: `32`
- Repetitions: `3`
- Command:

```bash
python -m phase_ssm.decodebench \
  --lengths 2048 8192 32768 65536 \
  --steps 32 \
  --reps 3
```

## Decode Result

| Context | PhaseSSM Decode | Attention Decode | SSM Mem | Attn Mem | Speed Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.511 ms/tok | 0.461 ms/tok | 0.04 GB | 0.10 GB | 0.90x |
| 8,192 | 0.510 ms/tok | 0.870 ms/tok | 0.07 GB | 0.20 GB | 1.71x |
| 32,768 | 0.510 ms/tok | 2.512 ms/tok | 0.12 GB | 0.60 GB | 4.93x |
| 65,536 | 0.511 ms/tok | 4.782 ms/tok | 0.18 GB | 1.14 GB | 9.35x |

## Mixed-Length Configuration

- PhaseSSM stack: `d_model=512`, `layers=8`, `state_dim=64`
- Attention stack: `d_model=512`, `layers=8`, `heads=8`
- Batch: `32`
- Sequence distribution: `uniform(512, 32768)`
- Mean sampled length: `15399`
- Max sampled length: `31683`
- Command:

```bash
python -m phase_ssm.effbench \
  --backend triton \
  --seq-dist uniform \
  --batch 32 \
  --min-seq 512 \
  --max-seq 32768
```

## Mixed-Length Result

| Workload | PhaseSSM Triton | Flash Attention | SSM Mem | Attn Mem | Speed Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `batch=32`, `uniform(512,32768)` | 216k tok/s | 144k tok/s | 0.30 GB | 12.52 GB | 1.50x |

## Read

The decode benchmark is the sharper long-context result. PhaseSSM stores fixed recurrent state per layer, so decode latency stays essentially flat from `2k` to `64k`. The attention baseline reads more KV cache as context grows, so its per-token latency increases with length.

The mixed-length benchmark shows a practical batching advantage. The PhaseSSM path processes exact sequence lengths. The attention path pays for the padded max length across the batch, which produces both a throughput loss and a large peak-memory gap.

## Backend Notes

The trainable PhaseSSM block already includes the short-range 4-tap depthwise causal convolution (`short_conv=4`) used to recover local token interactions. The current benchmark path keeps the fast production backend as recurrent Triton because the experimental chunked parallel scan is verified but slower on PGX.

RMSNorm plus projection fusion is not implemented in the current recurrent kernel. The recurrent kernel is launched per `(batch, channel)`, while RMSNorm and linear projection are cross-channel operations. A correct fusion needs a new row-blocked kernel shape rather than a small edit inside the current per-channel scan.
