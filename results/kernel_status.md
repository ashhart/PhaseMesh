# PhaseSSM Kernel Status

Status date: 2026-06-24.

This is the current kernel-engineering truth table. It is separate from language
quality and assistant usefulness.

## Current State

| Kernel Rung | Status | Read |
| --- | --- | --- |
| FFT convolution | Correct and trainable | Quality-training path, but materializes the sequence kernel and is not the speed path |
| Complex chunked PyTorch | Correct | Oracle-style O(L)-memory chunked scan, still Python/PyTorch heavy |
| Real-pair chunked PyTorch | Correct | Converts complex math to real lanes; this is the bridge to CUDA/Triton fusion |
| Real-pair recurrent PyTorch | Correct | Decode/reference path in real lanes |
| Recurrent Triton | Correct and fast at long context | Fuses recurrence and readout per channel; crosses attention at 64k in the mixer bench |
| Chunked Triton | Correct but not faster yet | The shape is present, but current launch/atomic/reduction costs lose to recurrent |
| Full-block fused Triton/CUDA | Not implemented | Needed for the next real speed jump below 64k and for full-model throughput |

## Measured Speed Boundary

The current NVIDIA GB10 short-prefill sweep reports:

| Length | Best PhaseSSM Backend | PhaseSSM tok/s | Attention tok/s | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 1k | recurrent Triton | ~215k | ~1121k | 0.19x |
| 8k | recurrent Triton | ~213k | ~746k | 0.29x |
| 16k | recurrent Triton | ~218k | ~472k | 0.46x |
| 32k | recurrent Triton | ~217k | ~282k | 0.77x |
| 64k | recurrent Triton | ~215k | ~155k | 1.39x |

FFT did not win at any measured short-context length, so the measured
`recommended_auto_threshold` is `None`. The short-context gap is not solved by
route selection. It needs a fused full-block kernel or a row-blocked parallel
scan that improves GPU utilization below 64k.

## What This Means

The real-pair formulation is a genuine engineering milestone because it removes
the complex dtype barrier and makes the oscillator update writable as standard
real-lane CUDA/Triton math. It does not by itself make PhaseSSM faster than
attention at practical short contexts.

Honest status:

- O(L) dynamics are verified.
- Real-pair math is the right fusable representation.
- Recurrent Triton proves the long-context scaling curve.
- Full-model speed is not proven.
- The next speed rung is fusion, not another routing tweak.
