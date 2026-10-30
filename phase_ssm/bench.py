"""Inference bench: where the architecture thesis actually lives.

A transformer's attention is O(L^2) compute and its KV cache is O(L) memory that
GROWS with context. An SSM is O(L) compute with FIXED state. So the honest place
to show "faster + flatter memory" is a context-length sweep, not seq-512 training.

Measures, for each model at several context lengths L:
  * prefill throughput (tok/s) for a full L-length forward
  * peak GPU memory for that forward
  * params and on-disk size (bf16)

    python -m phase_ssm.bench --out runs/bench.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

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
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    # warmup
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
    return {"tok_per_s": batch * L / dt, "latency_ms": dt * 1e3, "peak_mem_gb": peak}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="runs/bench.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    report = {"device": args.device, "batch": args.batch, "models": {}}
    for name in ("phasessm", "transformer"):
        model = build(name).to(args.device)
        report["models"][name] = {"params_m": model.num_params() / 1e6, "disk_mb_bf16": disk_mb(model), "by_length": {}}
        print(f"\n=== {name}  params={model.num_params()/1e6:.2f}M  disk={disk_mb(model):.1f}MB(bf16) ===")
        for L in args.lengths:
            try:
                r = measure(model, L, args.device, args.batch)
                report["models"][name]["by_length"][L] = r
                print(f"  L={L:6d}  {r['tok_per_s']/1e3:7.1f}k tok/s  {r['latency_ms']:8.1f}ms  peak {r['peak_mem_gb']:.2f}GB")
            except RuntimeError as e:
                report["models"][name]["by_length"][L] = {"error": str(e)[:120]}
                print(f"  L={L:6d}  OOM/ERROR: {str(e)[:80]}")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                break
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    # crossover summary
    print("\n=== context-scaling summary (tok/s ratio ssm/xf, mem ratio) ===")
    ssm, xf = report["models"]["phasessm"]["by_length"], report["models"]["transformer"]["by_length"]
    for L in args.lengths:
        if L in ssm and L in xf and "tok_per_s" in ssm[L] and "tok_per_s" in xf[L]:
            sr = ssm[L]["tok_per_s"] / xf[L]["tok_per_s"]
            mr = xf[L]["peak_mem_gb"] / max(ssm[L]["peak_mem_gb"], 1e-9)
            print(f"  L={L:6d}  ssm/xf speed {sr:.2f}x   xf/ssm mem {mr:.2f}x")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
