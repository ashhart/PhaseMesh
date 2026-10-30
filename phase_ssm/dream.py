"""DREAM — Deep Recurrent Equilibrium Associative Memory.

Design: docs/superpowers/specs/2026-06-26-dream-architecture-design.md
Rung 0 scope: the delta-rule memory (inner settle, T steps) and a single-pass
block (S=1), wired as an LM with the (ids, targets) -> (logits, loss) interface
so it drops into phase_ssm.tasks.train_on_mqar. The outer settling loop (S>1)
and adaptive halting come in Rung 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import OscillatorySSM, RMSNorm


@dataclass
class DreamConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    # inner settle (memory)
    mem_heads: int = 2
    mem_head_dim: int = 16
    inner_iters: int = 1        # T: delta settle steps (T=1 == standard delta rule)
    beta_init: float = 0.5      # delta-rule write strength (per-head, learnable)
    # outer settle
    outer_iters: int = 1        # S: settling iterations (Rung 0 uses S=1)
    damp_init: float = -2.0     # sigmoid(-2)~0.12 damped relaxation step
    # oscillator propagator
    state_dim: int = 32
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    d_ff_mult: int = 2
    mem_only: bool = False        # block = mem + FFN only (clean recall path, no oscillator/MLP)
    mem_short_conv: bool = False  # depthwise causal short conv before the memory projections
    # control + head
    linearize: bool = False     # remove the nonlinearity (the void-check)
    tie_embeddings: bool = True


class DeltaMemory(nn.Module):
    """Error-correcting associative memory (the inner settle).

    State per head: M in R^{dv x dk}. Causal delta-rule write at each position,
        M <- M + beta * (v - M k) k^T
    iterated `inner_iters` (T) times (T>1 settles M toward the least-squares fix
    point of recent bindings). Read o_t = M q_t. Sequential scan over L — fine at
    Rung-0 scale; a chunked path is deferred to Rung 3.
    """

    def __init__(self, cfg: DreamConfig):
        super().__init__()
        self.h, self.dk, self.dv = cfg.mem_heads, cfg.mem_head_dim, cfg.mem_head_dim
        self.T = cfg.inner_iters
        self.to_q = nn.Linear(cfg.d_model, self.h * self.dk)
        self.to_k = nn.Linear(cfg.d_model, self.h * self.dk)
        self.to_v = nn.Linear(cfg.d_model, self.h * self.dv)
        self.out = nn.Linear(self.h * self.dv, cfg.d_model)
        self.to_beta = nn.Linear(cfg.d_model, self.h)   # input-dependent write gate (DeltaNet-style)
        self.short_conv = (nn.Conv1d(cfg.d_model, cfg.d_model, 4, groups=cfg.d_model, padding=3)
                           if cfg.mem_short_conv else None)  # local mix so k,v co-locate before binding

    @staticmethod
    def delta_write(M, k, v, beta):
        """One delta step. M:(...,dv,dk) k:(...,dk) v:(...,dv) beta broadcastable."""
        Mk = (M * k.unsqueeze(-2)).sum(-1)            # (...,dv) = M k
        return M + beta * (v - Mk).unsqueeze(-1) * k.unsqueeze(-2)

    def forward(self, x):
        B, L, _ = x.shape
        if self.short_conv is not None:
            x = self.short_conv(x.transpose(1, 2))[..., :L].transpose(1, 2)  # causal local mix
        q = self.to_q(x).view(B, L, self.h, self.dk).transpose(1, 2)   # (B,h,L,dk)
        k = self.to_k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.to_v(x).view(B, L, self.h, self.dv).transpose(1, 2)
        k = F.normalize(k, dim=-1)                                      # ||k||=1: stable projection
        beta = torch.sigmoid(self.to_beta(x)).transpose(1, 2)          # (B,h,L) input-dependent write gate
        M = x.new_zeros(B, self.h, self.dv, self.dk)
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]            # (B,h,dk/dv)
            bt = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)            # (B,h,1,1): gate writes per token
            for _ in range(self.T):
                M = self.delta_write(M, kt, vt, bt)
            outs.append(torch.einsum("bhvd,bhd->bhv", M, qt))         # read o_t = M q_t
        o = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, L, self.h * self.dv)
        return self.out(o)


class DreamBlock(nn.Module):
    """One settled block. Oscillator propagator + memory read live INSIDE the
    nonlinearity; `outer_iters` (S) damped relaxation steps toward a fixed point.
    Rung 0 uses S=1."""

    def __init__(self, cfg: DreamConfig):
        super().__init__()
        self.mem_only = cfg.mem_only
        self.S = cfg.outer_iters
        self.norm = RMSNorm(cfg.d_model)
        self.mem = DeltaMemory(cfg)
        if cfg.mem_only:                                     # clean recall path: mem -> FFN, no oscillator
            from .model import SwiGLU
            self.norm2 = RMSNorm(cfg.d_model)
            self.ffn = SwiGLU(cfg.d_model, cfg.d_ff_mult * cfg.d_model)
        else:
            self.osc = OscillatorySSM(cfg.d_model, cfg.state_dim, cfg.dt_min, cfg.dt_max, backend="fft")
            if cfg.linearize:                                # control: no nonlinearity
                self.nl = nn.Linear(cfg.d_model, cfg.d_model)
            else:
                h = 2 * cfg.d_model
                self.nl = nn.Sequential(nn.Linear(cfg.d_model, h), nn.SiLU(), nn.Linear(h, cfg.d_model))
            self.alpha = nn.Parameter(torch.tensor(cfg.damp_init))

    def forward(self, x):
        if self.mem_only:
            x = x + self.mem(self.norm(x))                   # clean residual recall (like the additive memory)
            x = x + self.ffn(self.norm2(x))
            return x
        a = torch.sigmoid(self.alpha)
        for _ in range(self.S):
            zn = self.norm(x)
            mix = self.osc(zn) + self.mem(zn)                # propagate + recall, inside Mix
            x = x + a * (self.nl(mix) - x)                   # damped relaxation toward fixed point
        return x


class DreamLM(nn.Module):
    def __init__(self, cfg: DreamConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(DreamBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self):
        seen, tot = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p)); tot += p.numel()
        return tot
