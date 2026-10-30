"""Inference bench: where the architecture thesis actually lives.

A transformer's attention is O(L^2) compute and its KV cache is O(L) memory that
GROWS with context. An SSM is O(L) compute with FIXED state. So the honest place
to show "faster + flatter memory" is a context-length sweep, not seq-512 training.

Measures, for each model at several context lengths L:
  * prefill throughput (tok/s) for a full L-length forward
  * peak memory for that forward (exact on CUDA, sampled on MPS and CPU: see devmem.py)
  * params and on-disk size (bf16)

    python -m phase_ssm.bench --out runs/bench.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .devmem import PeakMemory, default_device, device_type, sync
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM


def build(name, vocab=256):
    if name == "phasessm":
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=384, n_layers=6,
                                         state_dim=96, expand=1, d_ff_mult=3))
    return TransformerLM(TransformerConfig(vocab_size=vocab, d_model=384, n_layers=6,
                                           n_heads=6, d_ff_mult=3, block_size=65536))


def disk_mb(model) -> float:
    return sum(p.numel() * 2 for p in {id(p): p for p in model.parameters()}.values()) / 1e6  # bf16


@torch.no_grad()
def measure(model, L, device, batch=4, reps=6):
    model.eval()
    x = torch.randint(0, 256, (batch, L), device=device)
    kind = device_type(device)
    with PeakMemory(device) as mem:
        with torch.autocast(kind, dtype=torch.bfloat16):  # warmup
            model(x)
        sync(device)
        t0 = time.time()
        for _ in range(reps):
            with torch.autocast(kind, dtype=torch.bfloat16):
                model(x)
        sync(device)
        dt = (time.time() - t0) / reps
    m = mem.result()
    return {"tok_per_s": batch * L / dt, "latency_ms": dt * 1e3,
            "peak_mem_gb": m["peak_gb"] if m["peak_gb"] is not None else 0.0,
            "added_mem_gb": m["added_gb"], "mem_method": m["mem_method"]}


def _empty_cache(device):
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def _print_row(L, r):
    added = "n/a" if r["added_mem_gb"] is None else f"{r['added_mem_gb']:.2f}GB"
    print(f"  L={L:6d}  {r['tok_per_s']/1e3:7.1f}k tok/s  {r['latency_ms']:8.1f}ms  "
          f"peak {r['peak_mem_gb']:.2f}GB  added {added}  ({r['mem_method']})", flush=True)


def _measure_isolated(name, L, args):
    """One (model, length) in a fresh process, so the CPU allocator's retained pages from an earlier
    measurement cannot hide this one's growth in the RSS high-water mark."""
    cmd = [sys.executable, "-m", "phase_ssm.bench", "--worker", name, str(L),
           "--device", args.device, "--batch", str(args.batch)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else f"exit {done.returncode}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _worker(name, L, args):
    model = build(name).to(args.device)
    print(json.dumps(measure(model, L, args.device, args.batch)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="runs/bench.json")
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=None,
                   help="one process per (model, length); default on for CPU, where memory is sampled RSS")
    p.add_argument("--worker", nargs=2, metavar=("MODEL", "LENGTH"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        _worker(args.worker[0], int(args.worker[1]), args)
        return
    isolate = device_type(args.device) == "cpu" if args.isolate is None else args.isolate

    report = {"device": args.device, "batch": args.batch, "isolated": isolate, "models": {}}
    for name in ("phasessm", "transformer"):
        model = build(name).to(args.device)
        report["models"][name] = {"params_m": model.num_params() / 1e6, "disk_mb_bf16": disk_mb(model), "by_length": {}}
        print(f"\n=== {name}  params={model.num_params()/1e6:.2f}M  disk={disk_mb(model):.1f}MB(bf16) ===")
        if isolate:
            del model
        for L in args.lengths:
            try:
                if isolate:
                    r = _measure_isolated(name, L, args)
                else:
                    _empty_cache(args.device)
                    r = measure(model, L, args.device, args.batch)
                report["models"][name]["by_length"][L] = r
                _print_row(L, r)
            except RuntimeError as e:
                report["models"][name]["by_length"][L] = {"error": str(e)[:120]}
                print(f"  L={L:6d}  OOM/ERROR: {str(e)[:80]}")
                _empty_cache(args.device)
                break
        if not isolate:
            del model
        _empty_cache(args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    # crossover summary
    print("\n=== context-scaling summary (tok/s ratio ssm/xf, mem ratio) ===")
    ssm, xf = report["models"]["phasessm"]["by_length"], report["models"]["transformer"]["by_length"]
    for L in args.lengths:
        if L in ssm and L in xf and "tok_per_s" in ssm[L] and "tok_per_s" in xf[L]:
            sr = ssm[L]["tok_per_s"] / xf[L]["tok_per_s"]
            # Off CUDA the peak includes the whole process, so compare what each forward added.
            key = "peak_mem_gb" if device_type(args.device) == "cuda" else "added_mem_gb"
            mr = (xf[L][key] or 0.0) / max(ssm[L][key] or 0.0, 1e-9)
            print(f"  L={L:6d}  ssm/xf speed {sr:.2f}x   xf/ssm mem {mr:.2f}x")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
