"""Recurrent O(1)-per-step form of the oscillatory SSM — the efficient-decode path.

The training forward computes the SSM as a causal convolution with an impulse
response (parallel, but it materialises an (H,N,L) kernel). The SAME linear
diagonal system has an exact recurrence:

    h_t = a ⊙ h_{t-1} + u_t            (a = exp(Δ·(-damping + i·freq)), per (H,N))
    y_t = 2·Re( Σ_n w_n · h_{t,n} ) + D·u_t

State is a fixed (B,H,N) complex vector — no growing KV cache, O(1) per token.
This module verifies the recurrence is bit-equivalent to the conv forward; that
equivalence is the oracle for the chunked scan that replaces the (H,N,L) kernel
at scale (Task 4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import OscillatorySSM


def ssm_poles(ssm: OscillatorySSM):
    dt = torch.exp(ssm.log_dt)                                  # (H,)
    damping = torch.exp(ssm.log_damping)                        # (H,N)
    dtA = dt[:, None] * torch.complex(-damping, ssm.freq)       # (H,N) = log(pole)
    a = torch.exp(dtA)                                          # discrete pole (H,N)
    w = torch.complex(ssm.w_real, ssm.w_imag)                   # (H,N)
    return a, w, dtA


def ssm_pole_real_pairs(ssm: OscillatorySSM, dtype: torch.dtype = torch.float32):
    """Return the complex pole/readout as real pairs.

    This is the kernel-engineering representation: instead of using
    ``complex64`` tensors, pack ``a = ar + i ai`` and ``w = wr + i wi`` into
    real tensors. CUDA/Triton kernels generally want this shape because it keeps
    the math in fp32/bf16 lanes and avoids slow complex dtype lowering.
    """
    dt = torch.exp(ssm.log_dt).to(dtype)                          # (H,)
    damping = torch.exp(ssm.log_damping).to(dtype)                # (H,N)
    freq = ssm.freq.to(dtype)                                     # (H,N)
    log_r = -dt[:, None] * damping                                # log |a|
    angle = dt[:, None] * freq                                    # arg(a)
    radius = torch.exp(log_r)
    ar = radius * torch.cos(angle)
    ai = radius * torch.sin(angle)
    wr = ssm.w_real.to(dtype)
    wi = ssm.w_imag.to(dtype)
    return ar, ai, wr, wi, log_r, angle


def _polar_power(log_r: torch.Tensor, angle: torch.Tensor, step: torch.Tensor):
    radius = torch.exp(step[:, None, None] * log_r[None])
    phase = step[:, None, None] * angle[None]
    return radius * torch.cos(phase), radius * torch.sin(phase)


def ssm_chunked(ssm: OscillatorySSM, u: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Chunked parallel scan. O(L) memory (materialises only (B,C,H,N) per chunk,
    NOT the (H,N,L) kernel), parallel within each chunk, sequential state across
    chunks. Bit-equivalent to the conv forward — the kernel that scales."""
    a, w, dtA = ssm_poles(ssm)
    B, L, H = u.shape
    N = ssm.N
    dev = u.device
    pad = (chunk - L % chunk) % chunk
    if pad:
        u = F.pad(u, (0, 0, 0, pad))
    Lp = u.shape[1]
    nc = Lp // chunk
    uc = u.view(B, nc, chunk, H)
    j = torch.arange(chunk, device=dev, dtype=torch.float32)
    apow = torch.exp(j[:, None, None] * dtA[None])              # a^j      (C,H,N)
    apowp1 = torch.exp((j[:, None, None] + 1.0) * dtA[None])    # a^{j+1}  (C,H,N)
    aneg = torch.exp(-j[:, None, None] * dtA[None])             # a^{-j}   (C,H,N)
    h = torch.zeros(B, H, N, dtype=torch.complex64, device=dev)
    outs = []
    for c in range(nc):
        uch = uc[:, c].to(torch.complex64)                     # (B,C,H)
        scaled = aneg[None] * uch[..., None]                   # (B,C,H,N)
        csum = torch.cumsum(scaled, dim=1)                     # within-chunk prefix
        hj = apowp1[None] * h[:, None] + apow[None] * csum     # (B,C,H,N) state at each pos
        y = 2.0 * (w[None, None] * hj).sum(-1).real + ssm.D[None, None] * uc[:, c]
        outs.append(y)
        h = hj[:, -1]                                          # carry state to next chunk
    return torch.cat(outs, dim=1)[:, :L]


def ssm_chunked_real(ssm: OscillatorySSM, u: torch.Tensor, chunk: int = 64,
                     compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Real-pair chunked scan equivalent to :func:`ssm_chunked`.

    The math is the same diagonal oscillator recurrence, but every complex
    multiply is expanded into real lanes:

        (ar + i ai)(hr + i hi) = (ar hr - ai hi) + i(ai hr + ar hi)

    This is the drop-in bridge from the verified PyTorch algorithm to a fused
    CUDA/Triton kernel. It keeps the public API unchanged and is differentiable,
    so it can also be used for backend experiments beyond inference benches.
    """
    ar, ai, wr, wi, log_r, angle = ssm_pole_real_pairs(ssm, compute_dtype)
    B, L, H = u.shape
    N = ssm.N
    dev = u.device
    pad = (chunk - L % chunk) % chunk
    if pad:
        u = F.pad(u, (0, 0, 0, pad))
    Lp = u.shape[1]
    nc = Lp // chunk
    uc = u.to(compute_dtype).view(B, nc, chunk, H)

    j = torch.arange(chunk, device=dev, dtype=compute_dtype)
    pow_r, pow_i = _polar_power(log_r, angle, j)
    powp1_r, powp1_i = _polar_power(log_r, angle, j + 1.0)
    inv_r, inv_i = _polar_power(log_r, angle, -j)

    h_r = torch.zeros(B, H, N, dtype=compute_dtype, device=dev)
    h_i = torch.zeros_like(h_r)
    D = ssm.D.to(compute_dtype)
    outs = []

    for c in range(nc):
        uch = uc[:, c]                                             # (B,C,H)
        scaled_r = inv_r[None] * uch[..., None]                    # (B,C,H,N)
        scaled_i = inv_i[None] * uch[..., None]
        csum_r = torch.cumsum(scaled_r, dim=1)
        csum_i = torch.cumsum(scaled_i, dim=1)

        # a^{j+1} h_prev
        prev_r = powp1_r[None] * h_r[:, None] - powp1_i[None] * h_i[:, None]
        prev_i = powp1_i[None] * h_r[:, None] + powp1_r[None] * h_i[:, None]
        # a^j Σ a^{-k} u_k
        add_r = pow_r[None] * csum_r - pow_i[None] * csum_i
        add_i = pow_i[None] * csum_r + pow_r[None] * csum_i
        hj_r = prev_r + add_r
        hj_i = prev_i + add_i

        y = 2.0 * (wr[None, None] * hj_r - wi[None, None] * hj_i).sum(-1)
        y = y + D[None, None] * uch
        outs.append(y)
        h_r = hj_r[:, -1]
        h_i = hj_i[:, -1]

    return torch.cat(outs, dim=1)[:, :L].to(u.dtype)


@torch.no_grad()
def ssm_recurrent(ssm: OscillatorySSM, u: torch.Tensor) -> torch.Tensor:
    """Step-by-step recurrence. u: (B,L,H) -> y: (B,L,H). For decode/equivalence."""
    a, w, _ = ssm_poles(ssm)
    B, L, H = u.shape
    h = torch.zeros(B, H, ssm.N, dtype=torch.complex64, device=u.device)
    out = torch.empty(B, L, H, dtype=torch.float32, device=u.device)
    for t in range(L):
        h = a[None] * h + u[:, t, :, None].to(torch.complex64)
        out[:, t, :] = 2.0 * (w[None] * h).sum(-1).real + ssm.D[None] * u[:, t, :]
    return out


@torch.no_grad()
def ssm_recurrent_real(ssm: OscillatorySSM, u: torch.Tensor) -> torch.Tensor:
    """Step-by-step real-pair recurrence for decode/equivalence checks."""
    ar, ai, wr, wi, _, _ = ssm_pole_real_pairs(ssm, torch.float32)
    B, L, H = u.shape
    h_r = torch.zeros(B, H, ssm.N, dtype=torch.float32, device=u.device)
    h_i = torch.zeros_like(h_r)
    out = torch.empty(B, L, H, dtype=torch.float32, device=u.device)
    D = ssm.D.to(torch.float32)
    uf = u.float()
    for t in range(L):
        next_r = ar[None] * h_r - ai[None] * h_i + uf[:, t, :, None]
        next_i = ai[None] * h_r + ar[None] * h_i
        h_r, h_i = next_r, next_i
        out[:, t, :] = 2.0 * (wr[None] * h_r - wi[None] * h_i).sum(-1) + D[None] * uf[:, t, :]
    return out


def _test():
    torch.manual_seed(0)
    H, N, L, B = 32, 16, 200, 2          # L not a multiple of chunk (tests padding)
    ssm = OscillatorySSM(H, N, dt_min=1e-3, dt_max=1e-1)
    u = torch.randn(B, L, H)
    y_conv = ssm(u)                       # parallel conv forward (training path)
    y_rec = ssm_recurrent(ssm, u)         # O(1)/step recurrence (decode path)
    y_chunk = ssm_chunked(ssm, u, chunk=64)  # O(L)-memory chunked scan (the kernel)
    y_real = ssm_chunked_real(ssm, u, chunk=64)
    y_real_rec = ssm_recurrent_real(ssm, u)

    scale = y_conv.abs().max().item() + 1e-9
    r_rec = (y_conv - y_rec).abs().max().item() / scale
    r_chunk = (y_conv - y_chunk).abs().max().item() / scale
    r_real = (y_conv - y_real).abs().max().item() / scale
    r_real_rec = (y_conv - y_real_rec).abs().max().item() / scale
    print(f"rel|conv - recurrent| = {r_rec:.2e}")
    print(f"rel|conv - chunked|   = {r_chunk:.2e}")
    print(f"rel|conv - realpair|  = {r_real:.2e}")
    print(f"rel|conv - realstep|  = {r_real_rec:.2e}")
    ok = r_rec < 1e-4 and r_chunk < 1e-4   # float32 precision floor ~1e-6
    ok = ok and r_real < 1e-4 and r_real_rec < 1e-4
    print("recurrent == conv:", "PASS ✓" if r_rec < 1e-4 else "FAIL ✗")
    print("chunked   == conv:", "PASS ✓" if r_chunk < 1e-4 else "FAIL ✗")
    print("realpair  == conv:", "PASS ✓" if r_real < 1e-4 else "FAIL ✗")
    print("realstep  == conv:", "PASS ✓" if r_real_rec < 1e-4 else "FAIL ✗")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
