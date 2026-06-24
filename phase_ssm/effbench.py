"""Efficiency bench — the slap demo. Chunked SSM (O(L)) vs flash-attention (O(L^2)).

Honest setup: a stack of chunked-SSM mixers vs a stack of causal flash-attention
mixers at matched width/depth. We measure prefill throughput and peak memory as
context length grows. The architectural claim: attention is O(L^2) compute, so
the SSM's O(L) chunked scan overtakes it at long context, and avoids the (H,N,L)
blow-up of the FFT kernel (which OOMs).

Precision caveat (stated, not hidden): attention runs bf16 + flash (its best
case); the SSM's complex recurrence runs fp32. So attention gets a constant-factor
head start — the SSM win at long L is therefore a *scaling* win, the honest kind.
"""
from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import OscillatorySSM
from .recurrent import ssm_chunked, ssm_chunked_real
from .triton_scan import ssm_chunked_scan_triton, ssm_recurrent_triton, triton_status


class SSMStack(nn.Module):
    def __init__(self, d, N, M, chunk=128, backend="real", state_block=16):
        super().__init__()
        self.layers = nn.ModuleList([OscillatorySSM(d, N, 1e-3, 1e-1) for _ in range(M)])
        self.chunk = chunk
        self.backend = backend
        self.state_block = state_block

    def forward(self, x):
        for l in self.layers:
            if self.backend == "complex":
                y = ssm_chunked(l, x, chunk=self.chunk)
            elif self.backend == "triton":
                y = ssm_recurrent_triton(l, x)
            elif self.backend == "triton_chunked":
                y = ssm_chunked_scan_triton(l, x, chunk=self.chunk, block_n=self.state_block)
            else:
                y = ssm_chunked_real(l, x, chunk=self.chunk)
            x = x + y
        return x


class AttnStack(nn.Module):
    def __init__(self, d, h, M):
        super().__init__()
        self.h = h
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(M)])
        self.proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(M)])

    def forward(self, x):
        B, L, D = x.shape
        for qkv, proj in zip(self.qkv, self.proj):
            q, k, v = qkv(x).chunk(3, -1)
            q = q.view(B, L, self.h, D // self.h).transpose(1, 2)
            k = k.view(B, L, self.h, D // self.h).transpose(1, 2)
            v = v.view(B, L, self.h, D // self.h).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + proj(a.transpose(1, 2).reshape(B, L, D))
        return x


@torch.no_grad()
def measure(model, L, dev, d, batch=1, reps=3, bf16=False):
    x = torch.randn(batch, L, d, device=dev)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    use_ac = bf16
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        model(x)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        for _ in range(reps):
            model(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return batch * L / dt, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--backend", choices=["real", "complex", "triton", "triton_chunked"], default="real")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--state-block", type=int, default=16)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--seq", type=int, default=None, help="Alias for --lengths SEQ when running a single length")
    args = ap.parse_args()
    if args.seq is not None:
        args.lengths = [args.seq]
    dev = "cuda"
    if args.backend in {"triton", "triton_chunked"}:
        ok, reason = triton_status()
        if not ok:
            raise SystemExit(f"backend=triton unavailable: {reason}")
    ssm = SSMStack(
        args.d_model,
        args.state_dim,
        args.layers,
        chunk=args.chunk,
        backend=args.backend,
        state_block=args.state_block,
    ).to(dev)
    attn = AttnStack(args.d_model, args.heads, args.layers).to(dev)
    print(
        f"d={args.d_model} layers={args.layers} state={args.state_dim} "
        f"| SSM({args.backend},chunk={args.chunk},state_block={args.state_block},fp32) vs Attn(flash,bf16)",
        flush=True,
    )
    print("%8s %11s %11s %9s %9s %8s" % ("L", "ssm_tok/s", "attn_tok/s", "ssm_GB", "attn_GB", "ssm/attn"), flush=True)
    for L in args.lengths:
        try:
            st, sm = measure(ssm, L, dev, args.d_model, bf16=False)
        except RuntimeError:
            st = sm = None
            torch.cuda.empty_cache()
        try:
            at, am = measure(attn, L, dev, args.d_model, bf16=True)
        except RuntimeError:
            at = am = None
            torch.cuda.empty_cache()
        spd = (st / at) if (st and at) else None
        print("%8d %11s %11s %9s %9s %8s" % (
            L,
            f"{st/1e3:.0f}k" if st else "OOM",
            f"{at/1e3:.0f}k" if at else "OOM",
            f"{sm:.2f}" if sm else "OOM",
            f"{am:.2f}" if am else "OOM",
            f"{spd:.2f}x" if spd else "-"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
