"""BitNet b1.58 ternary linear layer (1.58-bit weights).

Same module as AGI-WorldLLM/ternarylattice/ternary.py — kept self-contained in
PhaseMesh so the ternary-parity gate runs without cross-project imports.

Weights are {-1, 0, +1} at inference via absmean ternarization; an fp 'shadow'
weight is trained via the straight-through estimator (STE). The only state kept
at inference is the int8 ternary tensor + one fp scale per layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternarize(w: torch.Tensor, eps: float = 1e-5):
    """BitNet b1.58 absmean ternarization. Returns (ternary {-1,0,1}, scale)."""
    scale = w.abs().mean().clamp_min(eps)
    wq = torch.round(w / scale).clamp_(-1, 1)
    return wq, scale


class TernaryLinear(nn.Module):
    """Linear whose effective weight is ternary at inference; fp shadow trained via STE."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def effective_weight(self) -> torch.Tensor:
        wq, scale = ternarize(self.weight)
        w_eff = scale * wq
        # straight-through: forward uses w_eff, gradient flows to self.weight
        return self.weight + (w_eff - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def export_ternary(self):
        """Return (int8 ternary tensor, fp scale) — the only inference state."""
        wq, scale = ternarize(self.weight)
        return wq.to(torch.int8), scale.float()
