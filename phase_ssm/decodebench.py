"""Decode benchmark: fixed PhaseSSM state vs growing transformer KV cache.

This is a synthetic mixer-level decode benchmark, paired with ``effbench``. It
measures the step cost after a long context has already been prefetched:

* PhaseSSM stores fixed recurrent state ``(B,H,N)`` per layer.
* Transformer decode reads a KV cache of length ``L`` per layer.

The benchmark intentionally avoids language-model heads so it isolates the
sequence-mixing cost.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import OscillatorySSM
from .triton_scan import ssm_recurrent_triton, ssm_step_triton, triton_status


class DecodeSSMStack(nn.Module):
    def __init__(self, d: int, state_dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([OscillatorySSM(d, state_dim, 1e-3, 1e-1) for _ in range(layers)])

    @torch.no_grad()
    def prefill(self, x: torch.Tensor):
        states = []
        for layer in self.layers:
            y, sr, si = ssm_recurrent_triton(layer, x, return_state=True)
            x = x + y
            states.append((sr, si))
        return x[:, -1], states

    @torch.no_grad()
    def step(self, x: torch.Tensor, states):
        for layer, (sr, si) in zip(self.layers, states):
            y = ssm_step_triton(layer, x, sr, si)
            x = x + y
        return x


class DecodeAttnStack(nn.Module):
    def __init__(self, d: int, heads: int, layers: int):
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.h = heads
        self.hd = d // heads
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(layers)])
        self.proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])

    @torch.no_grad()
    def make_cache(self, batch: int, length: int, device: str):
        return [
            (
                torch.randn(batch, self.h, length, self.hd, device=device, dtype=torch.bfloat16),
                torch.randn(batch, self.h, length, self.hd, device=device, dtype=torch.bfloat16),
            )
            for _ in self.qkv
        ]

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache):
        B = x.shape[0]
        for qkv, proj, (k, v) in zip(self.qkv, self.proj, cache):
            q, _, _ = qkv(x.float()).chunk(3, dim=-1)
            q = q.view(B, self.h, 1, self.hd).to(torch.bfloat16)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            y = y.transpose(1, 2).reshape(B, self.d).float()
            x = x + proj(y)
        return x


@torch.no_grad()
def measure_decode(ssm, attn, length: int, device: str, d: int, batch: int, steps: int, reps: int):
    context = torch.randn(batch, length, d, device=device)
    x0, states = ssm.prefill(context)
    del context

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    x = x0
    for _ in range(steps):
        x = ssm.step(x, states)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        x = x0
        for _ in range(steps):
            x = ssm.step(x, states)
    torch.cuda.synchronize()
    ssm_dt = (time.time() - t0) / (reps * steps)
    ssm_mem = torch.cuda.max_memory_allocated() / 1e9
    del x0, states, x
    torch.cuda.empty_cache()

    cache = attn.make_cache(batch, length, device)
    x_attn = torch.randn(batch, d, device=device)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = x_attn
        for _ in range(steps):
            y = attn.step(y, cache)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(reps):
            y = x_attn
            for _ in range(steps):
                y = attn.step(y, cache)
    torch.cuda.synchronize()
    attn_dt = (time.time() - t0) / (reps * steps)
    attn_mem = torch.cuda.max_memory_allocated() / 1e9
    del cache, x_attn, y
    torch.cuda.empty_cache()

    return {
        "ssm_ms": ssm_dt * 1e3,
        "attn_ms": attn_dt * 1e3,
        "ssm_tok_s": 1.0 / ssm_dt,
        "attn_tok_s": 1.0 / attn_dt,
        "ssm_mem_gb": ssm_mem,
        "attn_mem_gb": attn_mem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.seq is not None:
        args.lengths = [args.seq]

    ok, reason = triton_status()
    if not ok:
        raise SystemExit(f"decodebench requires Triton backend: {reason}")
    dev = "cuda"
    ssm = DecodeSSMStack(args.d_model, args.state_dim, args.layers).to(dev)
    attn = DecodeAttnStack(args.d_model, args.heads, args.layers).to(dev)

    print(
        f"d={args.d_model} layers={args.layers} state={args.state_dim} "
        f"batch={args.batch} steps={args.steps}",
        flush=True,
    )
    print("%8s %11s %11s %9s %9s %8s" % ("L", "ssm_ms/tok", "attn_ms/tok", "ssm_GB", "attn_GB", "ssm/attn"), flush=True)
    for L in args.lengths:
        r = measure_decode(ssm, attn, L, dev, args.d_model, args.batch, args.steps, args.reps)
        print("%8d %11.3f %11.3f %9.2f %9.2f %8.2fx" % (
            L,
            r["ssm_ms"],
            r["attn_ms"],
            r["ssm_mem_gb"],
            r["attn_mem_gb"],
            r["attn_ms"] / r["ssm_ms"],
        ), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
