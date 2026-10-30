# PhaseSSM Short-Context Backend Sweep

This sweep measures the below-64k gap: FFT short prefill, recurrent Triton, hybrid auto, and flash attention.

- `recommended_auto_threshold`: `None`
- `first_best_ssm_win_vs_attention`: `65536`

| Length | Backend | SSM tok/s | Attention tok/s | SSM GB | Attention GB | Ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | `fft` | 17k | 1121k | 0.56 | 0.07 | 0.02x |
| 1024 | `triton` | 215k | 1121k | 0.02 | 0.07 | 0.19x |
| 1024 | `auto` | 17k | 1121k | 0.56 | 0.07 | 0.02x |
| 2048 | `fft` | 18k | 1104k | 1.11 | 0.09 | 0.02x |
| 2048 | `triton` | 215k | 1104k | 0.03 | 0.09 | 0.19x |
| 2048 | `auto` | 17k | 1104k | 1.11 | 0.09 | 0.02x |
| 4096 | `fft` | 18k | 1013k | 2.20 | 0.11 | 0.02x |
| 4096 | `triton` | 220k | 1013k | 0.05 | 0.11 | 0.22x |
| 4096 | `auto` | 18k | 1013k | 2.20 | 0.11 | 0.02x |
| 8192 | `fft` | 17k | 746k | 4.39 | 0.16 | 0.02x |
| 8192 | `triton` | 213k | 746k | 0.08 | 0.16 | 0.29x |
| 8192 | `auto` | 17k | 746k | 4.39 | 0.16 | 0.02x |
| 16384 | `fft` | 17k | 472k | 8.77 | 0.26 | 0.04x |
| 16384 | `triton` | 218k | 472k | 0.15 | 0.26 | 0.46x |
| 16384 | `auto` | 17k | 472k | 8.77 | 0.26 | 0.04x |
| 32768 | `fft` | 16k | 282k | 17.53 | 0.46 | 0.06x |
| 32768 | `triton` | 217k | 282k | 0.28 | 0.46 | 0.77x |
| 32768 | `auto` | 16k | 282k | 17.53 | 0.46 | 0.06x |
| 65536 | `fft` | 16k | 155k | 35.04 | 0.86 | 0.10x |
| 65536 | `triton` | 215k | 155k | 0.55 | 0.86 | 1.39x |
| 65536 | `auto` | 215k | 155k | 0.55 | 0.86 | 1.39x |

Read: `fft` is the candidate short-prefill path. `triton` is the recurrent speed path. If `recommended_auto_threshold` is `None`, FFT did not win at any measured length and `auto` should stay on Triton.
The result is a speed benchmark only; it does not measure language quality.
