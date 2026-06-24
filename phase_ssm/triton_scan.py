"""Experimental fused real-pair Triton scans for PhaseSSM inference.

The verified PyTorch chunk scan in :mod:`phase_ssm.recurrent` is the oracle. This
module is the first CUDA-facing implementation step: it keeps the oscillator
state as real pairs, fuses recurrence + readout projection where possible, and
uses compact chunk state instead of full sequence kernels. It intentionally
falls back loudly when Triton/CUDA is unavailable so local CPU/MPS development
stays deterministic.

The first kernel path is a fused recurrent scan. The second path is a chunked
parallel scan: reduce chunk transitions, scan chunk initials, then emit in-chunk
states in parallel.
"""
from __future__ import annotations

import torch
import os

from .model import OscillatorySSM
from .recurrent import ssm_pole_real_pairs

try:  # pragma: no cover - exercised only on CUDA+Triton hosts.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - local Mac path.
    triton = None
    tl = None


def triton_status() -> tuple[bool, str]:
    if triton is None:
        return False, "triton package is not installed"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    triton_version = getattr(triton, "__version__", "0.0")
    tv = tuple(int(part) for part in triton_version.split(".")[:2])
    if major >= 12 and tv < (3, 3) and os.getenv("PHASE_SSM_ALLOW_UNSUPPORTED_TRITON") != "1":
        return (
            False,
            f"CUDA capability sm_{major}{minor} requires Triton >= 3.3 for this "
            f"kernel path; found Triton {triton_version}. Set "
            "PHASE_SSM_ALLOW_UNSUPPORTED_TRITON=1 only with a toolchain known to support it.",
        )
    return True, "ok"


def triton_available() -> bool:
    ok, _ = triton_status()
    return ok


if triton is not None:  # pragma: no cover - CUDA-only JIT body.

    @triton.jit
    def _ssm_real_recurrent_kernel(
        u_ptr,
        ar_ptr,
        ai_ptr,
        wr_ptr,
        wi_ptr,
        d_ptr,
        y_ptr,
        state_r_ptr,
        state_i_ptr,
        B: tl.constexpr,
        L: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        nmask = offs_n < N

        hn = pid_h * N + offs_n
        ar = tl.load(ar_ptr + hn, mask=nmask, other=0.0)
        ai = tl.load(ai_ptr + hn, mask=nmask, other=0.0)
        wr = tl.load(wr_ptr + hn, mask=nmask, other=0.0)
        wi = tl.load(wi_ptr + hn, mask=nmask, other=0.0)
        d = tl.load(d_ptr + pid_h)

        hr = tl.zeros((BLOCK_N,), dtype=tl.float32)
        hi = tl.zeros((BLOCK_N,), dtype=tl.float32)

        t = 0
        while t < L:
            u_val = tl.load(u_ptr + pid_b * L * H + t * H + pid_h).to(tl.float32)
            next_r = ar * hr - ai * hi + u_val
            next_i = ai * hr + ar * hi
            hr = tl.where(nmask, next_r, 0.0)
            hi = tl.where(nmask, next_i, 0.0)
            prod = tl.where(nmask, wr * hr - wi * hi, 0.0)
            y_val = 2.0 * tl.sum(prod, axis=0) + d * u_val
            tl.store(y_ptr + pid_b * L * H + t * H + pid_h, y_val)
            t += 1

        state_base = (pid_b * H + pid_h) * N + offs_n
        tl.store(state_r_ptr + state_base, hr, mask=nmask)
        tl.store(state_i_ptr + state_base, hi, mask=nmask)


    @triton.jit
    def _ssm_chunk_reduce_kernel(
        u_ptr,
        log_r_ptr,
        angle_ptr,
        chunk_br_ptr,
        chunk_bi_ptr,
        B: tl.constexpr,
        L: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        N_BLOCKS: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_z = tl.program_id(2)
        pid_nb = pid_z % N_BLOCKS
        pid_c = pid_z // N_BLOCKS

        offs_l = tl.arange(0, BLOCK_L)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        l_abs = pid_c * BLOCK_L + offs_l
        lmask = l_abs < L

        hn = pid_h * N + offs_n
        log_r = tl.load(log_r_ptr + hn, mask=nmask, other=0.0)
        angle = tl.load(angle_ptr + hn, mask=nmask, other=0.0)

        # b_chunk = a^{C-1} Σ_k a^{-k} u_k, where C is BLOCK_L for full chunks.
        inv_step = -offs_l.to(tl.float32)
        inv_radius = tl.exp(inv_step[:, None] * log_r[None, :])
        inv_phase = inv_step[:, None] * angle[None, :]
        inv_r = inv_radius * tl.cos(inv_phase)
        inv_i = inv_radius * tl.sin(inv_phase)

        u = tl.load(
            u_ptr + pid_b * L * H + l_abs[:, None] * H + pid_h,
            mask=lmask[:, None] & nmask[None, :],
            other=0.0,
        )
        sum_r = tl.sum(inv_r * u, axis=0)
        sum_i = tl.sum(inv_i * u, axis=0)

        final_step = BLOCK_L - 1.0
        final_radius = tl.exp(final_step * log_r)
        final_phase = final_step * angle
        final_r = final_radius * tl.cos(final_phase)
        final_i = final_radius * tl.sin(final_phase)
        br = final_r * sum_r - final_i * sum_i
        bi = final_i * sum_r + final_r * sum_i

        base = ((pid_b * NUM_CHUNKS + pid_c) * H + pid_h) * N + offs_n
        tl.store(chunk_br_ptr + base, br, mask=nmask)
        tl.store(chunk_bi_ptr + base, bi, mask=nmask)


    @triton.jit
    def _ssm_chunk_state_kernel(
        log_r_ptr,
        angle_ptr,
        chunk_br_ptr,
        chunk_bi_ptr,
        init_r_ptr,
        init_i_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_nb = tl.program_id(2)

        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        hn = pid_h * N + offs_n
        log_r = tl.load(log_r_ptr + hn, mask=nmask, other=0.0)
        angle = tl.load(angle_ptr + hn, mask=nmask, other=0.0)

        step = BLOCK_L
        radius = tl.exp(step * log_r)
        phase = step * angle
        pow_r = radius * tl.cos(phase)
        pow_i = radius * tl.sin(phase)

        hr = tl.zeros((BLOCK_N,), dtype=tl.float32)
        hi = tl.zeros((BLOCK_N,), dtype=tl.float32)

        c = 0
        while c < NUM_CHUNKS:
            base = ((pid_b * NUM_CHUNKS + c) * H + pid_h) * N + offs_n
            tl.store(init_r_ptr + base, hr, mask=nmask)
            tl.store(init_i_ptr + base, hi, mask=nmask)

            br = tl.load(chunk_br_ptr + base, mask=nmask, other=0.0)
            bi = tl.load(chunk_bi_ptr + base, mask=nmask, other=0.0)
            next_r = pow_r * hr - pow_i * hi + br
            next_i = pow_i * hr + pow_r * hi + bi
            hr = tl.where(nmask, next_r, 0.0)
            hi = tl.where(nmask, next_i, 0.0)
            c += 1


    @triton.jit
    def _ssm_chunk_emit_kernel(
        u_ptr,
        log_r_ptr,
        angle_ptr,
        wr_ptr,
        wi_ptr,
        init_r_ptr,
        init_i_ptr,
        y_ptr,
        B: tl.constexpr,
        L: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        N_BLOCKS: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_z = tl.program_id(2)
        pid_nb = pid_z % N_BLOCKS
        pid_c = pid_z // N_BLOCKS

        offs_l = tl.arange(0, BLOCK_L)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        l_abs = pid_c * BLOCK_L + offs_l
        lmask = l_abs < L

        hn = pid_h * N + offs_n
        log_r = tl.load(log_r_ptr + hn, mask=nmask, other=0.0)
        angle = tl.load(angle_ptr + hn, mask=nmask, other=0.0)
        wr = tl.load(wr_ptr + hn, mask=nmask, other=0.0)
        wi = tl.load(wi_ptr + hn, mask=nmask, other=0.0)

        init_base = ((pid_b * NUM_CHUNKS + pid_c) * H + pid_h) * N + offs_n
        h0r = tl.load(init_r_ptr + init_base, mask=nmask, other=0.0)
        h0i = tl.load(init_i_ptr + init_base, mask=nmask, other=0.0)

        j = offs_l.to(tl.float32)
        radius = tl.exp(j[:, None] * log_r[None, :])
        phase = j[:, None] * angle[None, :]
        pow_r = radius * tl.cos(phase)
        pow_i = radius * tl.sin(phase)
        radius_p1 = tl.exp((j[:, None] + 1.0) * log_r[None, :])
        phase_p1 = (j[:, None] + 1.0) * angle[None, :]
        powp1_r = radius_p1 * tl.cos(phase_p1)
        powp1_i = radius_p1 * tl.sin(phase_p1)
        inv_radius = tl.exp(-j[:, None] * log_r[None, :])
        inv_phase = -j[:, None] * angle[None, :]
        inv_r = inv_radius * tl.cos(inv_phase)
        inv_i = inv_radius * tl.sin(inv_phase)

        u = tl.load(
            u_ptr + pid_b * L * H + l_abs[:, None] * H + pid_h,
            mask=lmask[:, None] & nmask[None, :],
            other=0.0,
        )
        scaled_r = inv_r * u
        scaled_i = inv_i * u
        csum_r = tl.cumsum(scaled_r, axis=0)
        csum_i = tl.cumsum(scaled_i, axis=0)

        prev_r = powp1_r * h0r[None, :] - powp1_i * h0i[None, :]
        prev_i = powp1_i * h0r[None, :] + powp1_r * h0i[None, :]
        add_r = pow_r * csum_r - pow_i * csum_i
        add_i = pow_i * csum_r + pow_r * csum_i
        hr = prev_r + add_r
        hi = prev_i + add_i

        partial = 2.0 * tl.sum(tl.where(nmask[None, :], wr[None, :] * hr - wi[None, :] * hi, 0.0), axis=1)
        tl.atomic_add(
            y_ptr + pid_b * L * H + l_abs * H + pid_h,
            partial,
            sem="relaxed",
            mask=lmask,
        )


def ssm_recurrent_triton(ssm: OscillatorySSM, u: torch.Tensor, block_n: int | None = None) -> torch.Tensor:
    """Fused CUDA recurrent scan.

    Raises:
        RuntimeError: when Triton or CUDA is not available.
        ValueError: when the input shape does not match the SSM.
    """
    ok, reason = triton_status()
    if not ok:
        raise RuntimeError(f"Triton PhaseSSM scan is unavailable: {reason}.")
    if u.ndim != 3:
        raise ValueError(f"expected u as (B,L,H), got shape {tuple(u.shape)}")
    B, L, H = u.shape
    if H != ssm.H:
        raise ValueError(f"input has H={H}, but SSM has H={ssm.H}")

    ar, ai, wr, wi, _, _ = ssm_pole_real_pairs(ssm, torch.float32)
    ar = ar.contiguous()
    ai = ai.contiguous()
    wr = wr.contiguous()
    wi = wi.contiguous()
    d = ssm.D.to(torch.float32).contiguous()
    uf = u.to(torch.float32).contiguous()
    y = torch.empty_like(uf)
    state_r = torch.empty(B, H, ssm.N, device=u.device, dtype=torch.float32)
    state_i = torch.empty_like(state_r)

    if block_n is None:
        block_n = triton.next_power_of_2(ssm.N)
    if block_n < ssm.N:
        raise ValueError(f"block_n={block_n} must be >= state_dim={ssm.N}")

    _ssm_real_recurrent_kernel[(B, H)](
        uf, ar, ai, wr, wi, d, y, state_r, state_i,
        B, L, H, ssm.N, block_n,
        num_warps=4,
    )
    return y.to(u.dtype)


def ssm_chunked_scan_triton(
    ssm: OscillatorySSM,
    u: torch.Tensor,
    chunk: int = 128,
    block_n: int = 16,
) -> torch.Tensor:
    """Chunked parallel-prefix Triton scan.

    This path reduces each chunk to a transition, scans chunk-initial states, and
    emits in-chunk states with ``tl.cumsum``. It is exact for full chunks and for
    the emitted valid positions of the final partial chunk; the final state of a
    partial last chunk is not consumed.
    """
    ok, reason = triton_status()
    if not ok:
        raise RuntimeError(f"Triton PhaseSSM chunked scan is unavailable: {reason}.")
    if u.ndim != 3:
        raise ValueError(f"expected u as (B,L,H), got shape {tuple(u.shape)}")
    B, L, H = u.shape
    if H != ssm.H:
        raise ValueError(f"input has H={H}, but SSM has H={ssm.H}")
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    _, _, wr, wi, log_r, angle = ssm_pole_real_pairs(ssm, torch.float32)
    wr = wr.contiguous()
    wi = wi.contiguous()
    log_r = log_r.contiguous()
    angle = angle.contiguous()
    uf = u.to(torch.float32).contiguous()
    y = torch.zeros_like(uf)
    num_chunks = triton.cdiv(L, chunk)
    block_l = triton.next_power_of_2(chunk)
    block_n = triton.next_power_of_2(block_n)
    if block_n > triton.next_power_of_2(ssm.N):
        block_n = triton.next_power_of_2(ssm.N)
    if block_n <= 0 or block_n > ssm.N and ssm.N > 0:
        block_n = triton.next_power_of_2(ssm.N)
    nblocks = triton.cdiv(ssm.N, block_n)

    chunk_br = torch.empty(B, num_chunks, H, ssm.N, device=u.device, dtype=torch.float32)
    chunk_bi = torch.empty_like(chunk_br)
    init_r = torch.empty_like(chunk_br)
    init_i = torch.empty_like(chunk_br)

    grid_chunks = (B, H, nblocks * num_chunks)
    _ssm_chunk_reduce_kernel[grid_chunks](
        uf, log_r, angle, chunk_br, chunk_bi,
        B, L, H, ssm.N, num_chunks, nblocks, block_l, block_n,
        num_warps=4,
    )
    _ssm_chunk_state_kernel[(B, H, nblocks)](
        log_r, angle, chunk_br, chunk_bi, init_r, init_i,
        B, H, ssm.N, num_chunks, block_l, block_n,
        num_warps=1,
    )
    _ssm_chunk_emit_kernel[grid_chunks](
        uf, log_r, angle, wr, wi, init_r, init_i, y,
        B, L, H, ssm.N, num_chunks, nblocks, block_l, block_n,
        num_warps=4,
    )
    y = y + ssm.D.to(torch.float32)[None, None, :] * uf
    return y.to(u.dtype)
