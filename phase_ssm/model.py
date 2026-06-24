"""PhaseSSM — the PhaseMesh substrate made trainable.

Each channel is a damped oscillator with its own frequency and damping (the
``field.py`` damped-wave intuition), but the dynamics, input map, and readout are
optimised end-to-end by gradient descent. Mathematically this is a diagonal
*complex* state-space model (the S4D / LinOSS family) whose poles are
parameterised explicitly as ``exp(Δ·(-damping + i·freq))``:

    h_t = a ⊙ h_{t-1} + u_t ,     a = exp(Δ·(-damping + i·freq))
    y_t = 2·Re( Σ_n w_n · h_{t,n} ) + D·u_t

* ``|a| = exp(-Δ·damping) < 1`` → stable, finite memory (the damping).
* ``arg(a) = Δ·freq``          → oscillation (the phase).

Because the recurrence is linear and diagonal, the whole sequence is computed in
parallel as a causal convolution with the SSM impulse response (kernel), so it
trains as fast as a conv net while remaining a genuine evolving-field model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhaseSSMConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    state_dim: int = 64          # N: oscillators per channel
    expand: int = 2              # d_inner = expand * d_model
    d_ff_mult: int = 2           # FFN hidden = d_ff_mult * d_model
    short_conv: int = 4          # depthwise causal conv kernel before the SSM (0 = off)
    ssm_backend: str = "fft"     # "fft", "auto", "real_chunked", "fixed_triton", or "skip"
    ssm_chunk: int = 64          # chunk length for real-pair scan experiments
    ssm_auto_threshold: int = 32768  # "auto": FFT at/below threshold, recurrent above
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_phase_memory: bool = False  # add the phase-binding associative-memory branch
    mem_heads: int = 4
    mem_head_dim: int = 32
    use_mixer: bool = True
    use_ffn: bool = True
    use_gate: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class OscillatorySSM(nn.Module):
    """Diagonal complex SSM with damped-oscillator poles, computed in conv mode.

    Input/output: ``(B, L, H)`` real. Each of the ``H`` channels carries ``N``
    independent complex oscillators.
    """

    def __init__(
        self,
        channels: int,
        state_dim: int,
        dt_min: float,
        dt_max: float,
        *,
        backend: str = "fft",
        chunk: int = 64,
        auto_threshold: int = 32768,
    ):
        super().__init__()
        self.H = channels
        self.N = state_dim
        self.backend = backend
        self.chunk = chunk
        self.auto_threshold = int(auto_threshold)

        # Timestep Δ per channel (log-uniform in [dt_min, dt_max]).
        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Continuous-time pole λ = -damping + i·freq, damping = exp(log_damping) > 0.
        # damping ≈ 0.5 at init (poles near, but inside, the unit circle).
        self.log_damping = nn.Parameter(torch.full((channels, state_dim), math.log(0.5)))
        # Spread oscillation frequencies (S4D-lin style: distinct rates per oscillator).
        freq = math.pi * torch.arange(state_dim, dtype=torch.float32).repeat(channels, 1)
        self.freq = nn.Parameter(freq)

        # Readout coefficient w = C·B per oscillator (complex), small init.
        scale = 1.0 / math.sqrt(state_dim)
        self.w_real = nn.Parameter(torch.randn(channels, state_dim) * scale)
        self.w_imag = nn.Parameter(torch.randn(channels, state_dim) * scale)

        # Direct skip path.
        self.D = nn.Parameter(torch.ones(channels))

    def kernel(self, L: int, device, dtype) -> torch.Tensor:
        dt = torch.exp(self.log_dt).to(dtype)                       # (H,)
        damping = torch.exp(self.log_damping).to(dtype)            # (H, N)
        freq = self.freq.to(dtype)                                 # (H, N)
        # log of the discrete pole: dtλ = Δ·(-damping + i·freq)
        dtA = dt[:, None] * torch.complex(-damping, freq)          # (H, N) complex
        w = torch.complex(self.w_real, self.w_imag).to(dtA.dtype)  # (H, N) complex

        t = torch.arange(L, device=device, dtype=dtype)            # (L,)
        # a^t = exp(t · dtλ)  →  (H, N, L)
        powers = torch.exp(dtA[:, :, None] * t[None, None, :])     # (H, N, L) complex
        kc = 2.0 * (w[:, :, None] * powers).sum(dim=1).real        # (H, L)
        return kc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        backend = self.backend_for_length(u.shape[1])
        if backend == "skip":
            return u * self.D[None, None, :]
        if backend == "fixed_triton":
            return _FixedKernelSSM.apply(u, self)
        if backend == "real_chunked":
            from .recurrent import ssm_chunked_real

            return ssm_chunked_real(self, u, chunk=self.chunk)
        if backend != "fft":
            raise ValueError(f"unsupported SSM backend: {self.backend!r}")

        # u: (B, L, H) -> (B, H, L)
        B, L, H = u.shape
        x = u.transpose(1, 2)                                      # (B, H, L)
        K = self.kernel(L, u.device, torch.float32)               # (H, L)

        n = 2 * L
        Uf = torch.fft.rfft(x.float(), n=n, dim=-1)
        Kf = torch.fft.rfft(K, n=n, dim=-1)
        y = torch.fft.irfft(Uf * Kf[None], n=n, dim=-1)[..., :L]   # (B, H, L) causal conv
        y = y + x * self.D[None, :, None]
        return y.transpose(1, 2).to(u.dtype)                      # (B, L, H)

    def backend_for_length(self, length: int) -> str:
        if self.backend != "auto":
            return self.backend
        return "fft" if int(length) <= self.auto_threshold else "fixed_triton"


class _FixedKernelSSM(torch.autograd.Function):
    """Fast/frozen SSM kernel.

    Forward uses the recurrent Triton inference kernel when available. Backward
    returns an exact gradient for the input ``u`` with the current SSM kernel,
    but intentionally returns no gradients for SSM parameters. This is a speed
    rung: train projections/gates/FFN around a fixed oscillator bank while the
    full trainable scan backward is built.
    """

    @staticmethod
    def forward(ctx, u: torch.Tensor, ssm: OscillatorySSM) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            kernel = ssm.kernel(u.shape[1], u.device, torch.float32).detach()
            d = ssm.D.detach().to(torch.float32)
            try:
                if u.is_cuda:
                    from .triton_scan import ssm_recurrent_triton, triton_available

                    if triton_available():
                        y = ssm_recurrent_triton(ssm, u)
                    else:
                        raise RuntimeError("Triton unavailable")
                else:
                    raise RuntimeError("CPU fallback")
            except RuntimeError:
                from .recurrent import ssm_chunked_real

                y = ssm_chunked_real(ssm, u, chunk=ssm.chunk)
        ctx.save_for_backward(kernel, d)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        kernel, d = ctx.saved_tensors
        B, L, H = grad_y.shape
        gy = grad_y.to(torch.float32).transpose(1, 2)                    # (B,H,L)
        rev = torch.flip(gy, dims=[-1])
        n = 2 * L
        yf = torch.fft.rfft(rev, n=n, dim=-1)
        kf = torch.fft.rfft(kernel, n=n, dim=-1)
        conv = torch.fft.irfft(yf * kf[None], n=n, dim=-1)[..., :L]
        grad_u = torch.flip(conv, dims=[-1]).transpose(1, 2)
        grad_u = grad_u + grad_y.to(torch.float32) * d[None, None, :]
        return grad_u.to(grad_y.dtype), None


class PhaseTimeMixer(nn.Module):
    """Sequence-mixing sublayer: in-proj → short causal conv → oscillatory SSM → gated out-proj."""

    def __init__(self, cfg: PhaseSSMConfig):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.use_gate = cfg.use_gate
        self.in_proj = nn.Linear(cfg.d_model, d_inner)
        self.gate_proj = nn.Linear(cfg.d_model, d_inner) if cfg.use_gate else None
        self.short_conv = (
            nn.Conv1d(d_inner, d_inner, cfg.short_conv, groups=d_inner, padding=cfg.short_conv - 1)
            if cfg.short_conv > 0 else None
        )
        self.ssm = OscillatorySSM(
            d_inner,
            cfg.state_dim,
            cfg.dt_min,
            cfg.dt_max,
            backend=cfg.ssm_backend,
            chunk=cfg.ssm_chunk,
            auto_threshold=cfg.ssm_auto_threshold,
        )
        self.out_proj = nn.Linear(d_inner, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)                                       # (B, L, d_inner)
        if self.short_conv is not None:
            L = u.shape[1]
            c = self.short_conv(u.transpose(1, 2))[..., :L]       # causal (drop right pad)
            u = c.transpose(1, 2)
        u = F.silu(u)
        y = self.ssm(u)
        if self.gate_proj is not None:
            y = y * F.silu(self.gate_proj(x))                    # input-dependent gate
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * hidden)
        self.w_out = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class PhaseBlock(nn.Module):
    def __init__(self, cfg: PhaseSSMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model) if cfg.use_mixer else None
        self.mixer = PhaseTimeMixer(cfg) if cfg.use_mixer else None
        self.memory = None
        if cfg.use_phase_memory and cfg.use_mixer:
            from .memory import PhaseBindingMemory
            self.norm_m = RMSNorm(cfg.d_model)
            self.memory = PhaseBindingMemory(cfg.d_model, cfg.mem_heads, cfg.mem_head_dim)
        self.norm2 = RMSNorm(cfg.d_model) if cfg.use_ffn else None
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff_mult * cfg.d_model) if cfg.use_ffn else None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer is not None and self.norm1 is not None:
            x = x + self.drop(self.mixer(self.norm1(x)))      # oscillatory SSM (temporal)
        if self.memory is not None:
            x = x + self.drop(self.memory(self.norm_m(x)))    # phase-binding recall (associative)
        if self.ffn is not None and self.norm2 is not None:
            x = x + self.drop(self.ffn(self.norm2(x)))        # channel mixing
        return x


class PhaseSSMLM(nn.Module):
    def __init__(self, cfg: PhaseSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(PhaseBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self) -> int:
        # Tied head shares the embedding, so count unique parameters.
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(ids)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1) if temperature > 0 else logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
