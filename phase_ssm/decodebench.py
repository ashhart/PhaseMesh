"""Decode benchmark: fixed PhaseSSM state vs growing transformer KV cache.

This is a synthetic mixer-level decode benchmark, paired with ``effbench``. It
measures the step cost after a long context has already been prefetched:

* PhaseSSM stores fixed recurrent state ``(B,H,N)`` per layer.
* Transformer decode reads a KV cache of length ``L`` per layer.

The benchmark intentionally avoids language-model heads so it isolates the
sequence-mixing cost.

Runs on CUDA with the Triton kernels, and on Apple MPS or CPU with the PyTorch
real-pair step (``ssm_step_real``). Memory is exact on CUDA and sampled on MPS
and CPU (see ``devmem.py``); ``held_gb`` is the exact size of what each model
keeps between tokens: the recurrent state, or the KV cache. On CPU each
measurement runs in its own process so retained allocator pages cannot hide
growth.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .devmem import PeakMemory, default_device, device_type, sync
from .model import OscillatorySSM
from .recurrent import ssm_chunked_real, ssm_final_state_real, ssm_step_real
from .triton_scan import ssm_recurrent_triton, ssm_step_triton, triton_status


class DecodeSSMStack(nn.Module):
    def __init__(self, d: int, state_dim: int, layers: int, use_triton: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([OscillatorySSM(d, state_dim, 1e-3, 1e-1) for _ in range(layers)])
        self.use_triton = use_triton

    @torch.no_grad()
    def prefill(self, x: torch.Tensor):
        states = []
        for layer in self.layers:
            if self.use_triton:
                y, sr, si = ssm_recurrent_triton(layer, x, return_state=True)
            else:
                y = ssm_chunked_real(layer, x)
                sr, si = ssm_final_state_real(layer, x)
            x = x + y
            states.append((sr, si))
        return x[:, -1], states

    @torch.no_grad()
    def step(self, x: torch.Tensor, states):
        step_fn = ssm_step_triton if self.use_triton else ssm_step_real
        for layer, (sr, si) in zip(self.layers, states):
            y = step_fn(layer, x, sr, si)
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


def _held_gb(tensors) -> float:
    return sum(t.numel() * t.element_size() for pair in tensors for t in pair) / 1e9


def _empty_cache(device: str) -> None:
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def _time_steps(step, x0, held, device, steps, reps, autocast_kind=None):
    def run():
        x = x0
        for _ in range(steps):
            x = step(x, held)
        return x

    def maybe_autocast():
        return torch.autocast(autocast_kind, dtype=torch.bfloat16) if autocast_kind else torch.autocast(
            device_type(device), enabled=False)

    with maybe_autocast():
        run()  # warmup
    sync(device)
    t0 = time.time()
    with maybe_autocast():
        for _ in range(reps):
            run()
    sync(device)
    return (time.time() - t0) / (reps * steps)


@torch.no_grad()
def measure_ssm(ssm, length: int, device: str, d: int, batch: int, steps: int, reps: int) -> dict:
    context = torch.randn(batch, length, d, device=device)
    x0, states = ssm.prefill(context)
    del context
    _empty_cache(device)
    with PeakMemory(device) as mem:
        held = [(sr.clone(), si.clone()) for sr, si in states]  # allocated inside, so it is counted
        dt = _time_steps(ssm.step, x0, held, device, steps, reps)
    m = mem.result()
    return {"ms": dt * 1e3, "tok_s": 1.0 / dt, "held_gb": _held_gb(held), "peak_gb": m["peak_gb"],
            "added_gb": m["added_gb"], "mem_method": m["mem_method"]}


@torch.no_grad()
def measure_attn(attn, length: int, device: str, d: int, batch: int, steps: int, reps: int) -> dict:
    x0 = torch.randn(batch, d, device=device)
    _empty_cache(device)
    with PeakMemory(device) as mem:
        cache = attn.make_cache(batch, length, device)
        dt = _time_steps(attn.step, x0, cache, device, steps, reps, autocast_kind=device_type(device))
    m = mem.result()
    return {"ms": dt * 1e3, "tok_s": 1.0 / dt, "held_gb": _held_gb(cache), "peak_gb": m["peak_gb"],
            "added_gb": m["added_gb"], "mem_method": m["mem_method"]}


def _build(kind: str, args, use_triton: bool):
    if kind == "ssm":
        return DecodeSSMStack(args.d_model, args.state_dim, args.layers, use_triton).to(args.device)
    return DecodeAttnStack(args.d_model, args.heads, args.layers).to(args.device)


def _measure(kind: str, length: int, args, use_triton: bool) -> dict:
    fn = measure_ssm if kind == "ssm" else measure_attn
    return fn(_build(kind, args, use_triton), length, args.device, args.d_model, args.batch, args.steps, args.reps)


def _measure_isolated(kind: str, length: int, argv: list[str]) -> dict:
    done = subprocess.run([sys.executable, "-m", "phase_ssm.decodebench", *argv, "--worker", kind, str(length)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"exit {done.returncode}")
    return json.loads(done.stdout.strip().splitlines()[-1])


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
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--backend", choices=("auto", "triton", "torch"), default="auto",
                    help="SSM step kernels: Triton (CUDA only) or the PyTorch real-pair step")
    ap.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=None,
                    help="one process per measurement; default on for CPU, where memory is sampled RSS")
    ap.add_argument("--out", type=str, default=None, help="optional JSON report path")
    ap.add_argument("--worker", nargs=2, metavar=("KIND", "LENGTH"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.seq is not None:
        args.lengths = [args.seq]

    ok, reason = triton_status()
    if args.backend == "triton" and not ok:
        raise SystemExit(f"--backend triton unavailable: {reason}")
    use_triton = ok and device_type(args.device) == "cuda" and args.backend != "torch"
    if args.worker:
        print(json.dumps(_measure(args.worker[0], int(args.worker[1]), args, use_triton)))
        return
    isolate = device_type(args.device) == "cpu" if args.isolate is None else args.isolate
    argv = [a for a in sys.argv[1:] if a not in ("--isolate", "--no-isolate")]
    argv = [a for i, a in enumerate(argv) if not (a == "--out" or (i > 0 and argv[i - 1] == "--out"))]

    print(
        f"device={args.device} backend={'triton' if use_triton else 'torch'} isolated={isolate} "
        f"d={args.d_model} layers={args.layers} state={args.state_dim} batch={args.batch} steps={args.steps}",
        flush=True,
    )
    print("%8s %11s %11s %10s %10s %9s %9s %8s" % (
        "L", "ssm_ms/tok", "attn_ms/tok", "ssm_held", "attn_held", "ssm_GB", "attn_GB", "attn/ssm"), flush=True)
    rows = []
    for L in args.lengths:
        try:
            r = {k: (_measure_isolated(k, L, argv) if isolate else _measure(k, L, args, use_triton))
                 for k in ("ssm", "attn")}
        except RuntimeError as e:
            print(f"{L:8d}  OOM/ERROR: {str(e)[:100]}", flush=True)
            break
        rows.append({"length": L, **r})
        gb = "peak_gb" if device_type(args.device) == "cuda" else "added_gb"
        fmt = lambda v: f"{v:9.3f}" if v is not None else "      n/a"
        print("%8d %11.3f %11.3f %10.4f %10.4f %s %s %8.2fx" % (
            L, r["ssm"]["ms"], r["attn"]["ms"], r["ssm"]["held_gb"], r["attn"]["held_gb"],
            fmt(r["ssm"][gb]), fmt(r["attn"][gb]), r["attn"]["ms"] / r["ssm"]["ms"]), flush=True)
    if args.out:
        report = {"device": args.device, "backend": "triton" if use_triton else "torch", "isolated": isolate,
                  "config": {k: getattr(args, k) for k in ("d_model", "layers", "state_dim", "heads", "batch", "steps", "reps")},
                  "rows": rows}
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
