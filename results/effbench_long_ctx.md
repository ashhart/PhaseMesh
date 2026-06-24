# PhaseSSM Long-Context Efficiency Bench

This is the primary PhaseSSM efficiency result as of 2026-06-24.

## Claim

On a matched synthetic mixer stack, the fused recurrent PhaseSSM Triton backend crosses flash attention between 32k and 64k context. At 64k and beyond it is faster and uses less peak GPU memory.

This measures long-context prefill throughput and peak allocated GPU memory. It does not claim language-model quality parity.

## Environment

- Host: PGX
- GPU: NVIDIA GB10, CUDA capability `sm_121`
- Container: `nvcr.io/nvidia/pytorch:26.01-py3`
- PyTorch: `2.10.0a0+a36e1d39eb.nv26.01.42222806`
- Triton: `3.6.0`
- Command:

```bash
python -m phase_ssm.effbench \
  --backend triton \
  --lengths 8192 32768 65536 131072
```

## Configuration

- PhaseSSM stack: `d_model=512`, `layers=8`, `state_dim=64`
- Attention stack: `d_model=512`, `layers=8`, `heads=8`
- PhaseSSM precision: fp32 recurrent Triton kernel
- Attention precision: bf16 flash attention through `torch.nn.functional.scaled_dot_product_attention`
- Batch: `1`

## Result

| Context | PhaseSSM Triton | Flash Attention | SSM Mem | Attn Mem | Speed Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 212k tok/s | 744k tok/s | 0.11 GB | 0.16 GB | 0.29x |
| 32,768 | 218k tok/s | 282k tok/s | 0.32 GB | 0.47 GB | 0.77x |
| 65,536 | 217k tok/s | 155k tok/s | 0.58 GB | 0.87 GB | 1.40x |
| 131,072 | 214k tok/s | 80k tok/s | 1.12 GB | 1.67 GB | 2.69x |
| 262,144 | 216k tok/s | 40k tok/s | 2.19 GB | 3.28 GB | 5.44x |

## Read

The PhaseSSM recurrent Triton backend stays roughly flat at `~214k-218k tok/s` from 32k to 131k. Flash attention falls from `744k tok/s` at 8k to `80k tok/s` at 131k. The crossover appears between 32k and 64k.

Peak memory also favors PhaseSSM throughout this sweep, with the long-context gap widening at larger lengths.

## 1M Extension Sweep

After the first 131k run, the same recurrent Triton backend was swept to 1M context on PGX. This run measured the PhaseSSM side directly, then ran a bounded 256k side-by-side comparison against flash attention.

Command for the SSM-only extension:

```bash
python - <<'PY'
import torch
from phase_ssm.effbench import SSMStack, measure

d = 512
layers = 8
state = 64
dev = "cuda"
ssm = SSMStack(d, state, layers, backend="triton").to(dev)
for L in [262144, 524288, 1048576]:
    tok_s, mem = measure(ssm, L, dev, d, batch=1, reps=3, bf16=False)
    print(L, tok_s, mem)
PY
```

| Context | PhaseSSM Triton | SSM Mem |
| ---: | ---: | ---: |
| 262,144 | 216k tok/s | 2.15 GB |
| 524,288 | 216k tok/s | 4.30 GB |
| 1,048,576 | 216k tok/s | 8.59 GB |

Command for the bounded 256k side-by-side row:

```bash
timeout 240 python -m phase_ssm.effbench \
  --backend triton \
  --lengths 262144
```

Result:

| Context | PhaseSSM Triton | Flash Attention | SSM Mem | Attn Mem | Speed Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | 216k tok/s | 40k tok/s | 2.19 GB | 3.28 GB | 5.44x |

Read: PhaseSSM stays at `~216k tok/s` from 256k through 1M. The measured side-by-side gap is `5.44x` at 256k. Larger attention rows were not run because the 256k row already pushes the comparison into the steep part of the attention curve, while the SSM million-token extension completed directly.

## Backend Notes

The literal proposed chunked-scan snippet was tested as pasted in an isolated scratch file and failed before Triton compilation:

```text
SyntaxError: positional argument follows keyword argument
tl.reduce(step_A, step_b, axis=1, _ssm_reduce_op)
```

The implemented `triton_chunked` backend is mathematically correct against the recurrent Triton and PyTorch real-pair references, but currently slower:

| Backend | Context | SSM tok/s | Flash tok/s | SSM Mem | Attn Mem | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `triton_chunked` | 8,192 | 59k | 755k | 0.17 GB | 0.16 GB | 0.08x |
| `triton_chunked` | 65,536 | 58k | 156k | 1.11 GB | 0.87 GB | 0.37x |

The current production speed path is therefore `--backend triton`, not `--backend triton_chunked`.
