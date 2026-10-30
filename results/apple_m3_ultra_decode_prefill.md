# PhaseSSM Decode and Prefill on Apple Silicon (MPS) and CPU

Measured 2026-10-30 by Chronara (James Tervit) with `decodebench` and `bench` running off-CUDA, using the
PyTorch real-pair step added in this change. Same configurations as `decode_mixed_bench.md` so the rows can be
read next to the GB10 numbers. Random-init weights: these are speed and memory measurements, not quality.

## Claim

Decode: the PhaseSSM recurrent state stays at 2.1 MB and its step cost stays flat out to 65k context on both MPS
and CPU, while the attention KV cache grows linearly to 1.07 GB. Attention decode is faster below about 32k
(MPS) and about 8k (CPU) and slower beyond.

Prefill with the default FFT backend: on CPU, PhaseSSM is slower than the transformer at every measured length
up to 16k and uses more memory, growing linearly (about 10 GB at 16k), consistent with the `(H,N,L)` kernel. The
fixed-state advantage is a decode result here, not a prefill one.

## Environment

- Host: Apple M3 Ultra, 28 CPU cores, 96 GB unified memory, macOS 27.0.1
- PyTorch: `2.15.0` (no Triton: SSM uses `ssm_step_real` / `ssm_chunked_real` / `ssm_final_state_real`)
- Jobs ran at `nice -n 10` beside other light services; no power measurement.

## Decode

`python -m phase_ssm.decodebench --device {mps,cpu} --lengths 2048 8192 32768 65536`
(d_model 512, 8 layers, state 64, 8 heads, batch 1, 32 steps, 5 reps).

`held` is the exact size of what each model keeps between tokens. `added` is measured: sampled
`torch.mps.current_allocated_memory` on MPS, sampled process RSS on CPU with one process per measurement.

### MPS

| L | PhaseSSM ms/tok | attention ms/tok | PhaseSSM held | KV cache held | attention added | attn/ssm time |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.057 | 0.425 | 2.1 MB | 33.6 MB | 0.050 GB | 0.40x |
| 8,192 | 1.101 | 0.612 | 2.1 MB | 134.2 MB | 0.151 GB | 0.56x |
| 32,768 | 1.049 | 1.143 | 2.1 MB | 536.9 MB | 0.554 GB | 1.09x |
| 65,536 | 1.051 | 1.831 | 2.1 MB | 1,073.7 MB | 1.092 GB | 1.74x |

### CPU

| L | PhaseSSM ms/tok | attention ms/tok | PhaseSSM held | KV cache held | attention added | attn/ssm time |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 4.351 | 2.802 | 2.1 MB | 33.6 MB | 0.055 GB | 0.64x |
| 8,192 | 4.418 | 4.111 | 2.1 MB | 134.2 MB | 0.158 GB | 0.93x |
| 32,768 | 4.509 | 10.237 | 2.1 MB | 536.9 MB | 0.560 GB | 2.27x |
| 65,536 | 4.565 | 18.113 | 2.1 MB | 1,073.7 MB | 1.097 GB | 3.97x |

PhaseSSM measured `added` was 0.000 to 0.003 GB at every length, in line with its 2.1 MB held state.

## Prefill (full LM, default FFT backend)

`python -m phase_ssm.bench --device cpu --batch 1 --lengths 512 1024 2048 4096 8192 16384`
(the `bench.py` models: 11.6M params each, 6 layers, d_model 384, PhaseSSM state 96).

| L | PhaseSSM tok/s | transformer tok/s | PhaseSSM added | transformer added |
|---:|---:|---:|---:|---:|
| 512 | 900 | 6,300 | 0.36 GB | 0.05 GB |
| 1,024 | 1,200 | 5,100 | 0.68 GB | 0.08 GB |
| 2,048 | 1,100 | 4,500 | 1.32 GB | 0.09 GB |
| 4,096 | 1,100 | 3,400 | 2.58 GB | 0.14 GB |
| 8,192 | 1,100 | 1,900 | 5.11 GB | 0.21 GB |
| 16,384 | 900 | 1,000 | 10.19 GB | 0.44 GB |

MPS prefill speed (PhaseSSM 32k to 44k tok/s, transformer 103k to 195k tok/s, PhaseSSM 0.21x to 0.39x of transformer) is in
the JSON, but MPS prefill memory is not reported: the FFT temporaries are allocated and freed as work is
enqueued, faster than the 1 ms sampler, so the sampled values are not reliable for them (see `devmem.py`).

## Limits

- Random-init weights; no quality claim.
- One run per row; no confidence intervals.
- Sampled memory on MPS and CPU can miss spikes shorter than 1 ms.
- Prefill used the default FFT backend; `real_chunked` prefill was not measured here.
