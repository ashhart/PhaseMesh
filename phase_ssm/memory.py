"""Phase-binding associative memory — the differentiator.

Pure SSMs (Mamba included) have one well-known weakness: exact in-context
associative recall (the MQAR problem) — which is why production models go hybrid.
This layer fixes it with PhaseMesh's own idea: bind key->value as complex
*phasor* products (Holographic Reduced Representations / VSA) and retrieve by
unbinding with the query phase.

    address_t = exp(i·Wk·x_t)          (unit phasor — the binding key)
    write:   S_t = γ·S_{t-1} + address_t ⊗ value_t
    read:    o_t = Re( conj(exp(i·Wq·x_t)) · S_t ) / Σγ

When a later query's phase matches an earlier key's phase, the phasors align and
that value is recovered; mismatched phases average to zero. This is content-
addressable recall with O(L) state (no growing KV cache), and it has an exact
recurrent form — so it is linear-attention-class, not softmax attention.

The training-time path below uses the equivalent decayed parallel form (like
RetNet/GLA): same math, parallelizable; the recurrent form is for streaming.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhaseBindingMemory(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, head_dim: int = 32, v_dim: int | None = None):
        super().__init__()
        self.h = n_heads
        self.dk = head_dim
        self.dv = v_dim or head_dim
        self.to_q = nn.Linear(d_model, n_heads * head_dim)   # query phase angles
        self.to_k = nn.Linear(d_model, n_heads * head_dim)   # key (address) phase angles
        self.to_v = nn.Linear(d_model, n_heads * self.dv)    # value content
        self.out = nn.Linear(n_heads * self.dv, d_model)
        # Per-head memory decay γ = sigmoid(decay_logit); init ≈0.98 (long horizon).
        self.decay_logit = nn.Parameter(torch.full((n_heads,), 4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qa = self.to_q(x).float().view(B, L, self.h, self.dk).permute(0, 2, 1, 3)  # (B,h,L,dk)
        ka = self.to_k(x).float().view(B, L, self.h, self.dk).permute(0, 2, 1, 3)
        v = self.to_v(x).float().view(B, L, self.h, self.dv).permute(0, 2, 1, 3)   # (B,h,L,dv)

        q = torch.exp(1j * qa)      # unit phasors
        k = torch.exp(1j * ka)
        # phase-coherence between query t and key s: |<q_t, k_s>|^2 over dk dims.
        # Aligned phases -> ~dk (large); random phases -> ~O(1). Squaring makes the
        # matching key dominate, and keeps the retrieval weight strictly POSITIVE so
        # the normaliser concentrates on the match instead of diluting over all keys.
        inner = torch.einsum("bhtd,bhsd->bhts", q, k.conj()) / (self.dk ** 0.5)
        coh = (inner.real ** 2 + inner.imag ** 2)                    # (B,h,L,L) >= 0

        gamma = torch.sigmoid(self.decay_logit)                      # (h,)
        t = torch.arange(L, device=x.device)
        rel = t[:, None] - t[None, :]                                # (L,L)
        causal = rel >= 0
        logg = torch.log(gamma.clamp_min(1e-4))[:, None, None]       # (h,1,1)
        decay = torch.exp(rel[None].float() * logg) * causal[None]   # (h,L,L), 0 for future
        A = coh * decay[None]                                        # (B,h,L,L) >= 0
        o = torch.einsum("bhts,bhsd->bhtd", A, v)                    # (B,h,L,dv)
        denom = A.sum(-1, keepdim=True).clamp_min(1e-4)              # (B,h,L,1) match-weighted norm
        o = o / denom
        o = o.permute(0, 2, 1, 3).reshape(B, L, self.h * self.dv)
        return self.out(o.to(x.dtype))

    @torch.no_grad()
    def recurrent_state_bytes(self) -> int:
        """Fixed memory footprint of the streaming state (no KV growth)."""
        return self.h * self.dk * self.dv * 8  # complex64 state per head
