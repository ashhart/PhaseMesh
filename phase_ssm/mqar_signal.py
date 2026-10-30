"""CPU-feasible MQAR signal run: the three-way differentiator at one difficulty.

This is a reduced-scale version of phase_ssm.mqar_sweep, sized to complete on a
CPU in minutes rather than on NVIDIA GB10. It runs the three configs that define the
go/no-go: SSM-only, SSM + linear phase-memory (OURS), and the transformer — all
at matched parameters, on the same MQAR task, for the same step budget.

The full sweep (multiple pair-counts, multiple seeds, more steps) belongs on
NVIDIA GB10. This script answers the one question that matters first: does OURS separate
from SSM-only and approach the transformer on recall at a difficulty where pure
SSMs are known to struggle?

Pass = OURS > SSM-only clearly AND OURS approaches the transformer.
Fail  = OURS stays near SSM-only -> the memory is not carrying recall.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.tasks import train_on_mqar
from phase_ssm.transformer import TransformerConfig, TransformerLM


def build(kind, vocab, d_model, n_layers, state_dim, mem_linear, mem_layout):
    """Matched-param builders. Transformer d_ff_mult tuned so param counts are close."""
    if kind == "xf":
        return TransformerLM(TransformerConfig(
            vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=max(1, d_model // 32),
            d_ff_mult=2, block_size=512, pos_enc="rope"))
    return PhaseSSMLM(PhaseSSMConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, state_dim=state_dim,
        expand=1, d_ff_mult=2, use_phase_memory=(kind != "ssm_only"),
        mem_heads=2, mem_head_dim=16, mem_linear=mem_linear, mem_layout=mem_layout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--values", type=int, default=128)
    ap.add_argument("--d-model", dest="d_model", type=int, default=64)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=2)
    ap.add_argument("--state-dim", dest="state_dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mem-linear", dest="mem_linear", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mem-layout", dest="mem_layout", choices=["parallel", "sequential"], default="parallel")
    ap.add_argument("--out", default="runs/mqar_signal.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NK, NV = args.keys, args.values
    vocab = 1 + NK + NV
    chance = 1.0 / NV
    specs = [("SSM-only", "ssm_only", 1e-3), ("OURS+mem", "ours", 1e-3),
             ("Transformer", "xf", 3e-4)]

    print(f"MQAR signal run | device={dev} | P={args.pairs} Q={args.queries} "
          f"NK={NK} NV={NV} (chance~{chance:.3f}) | d={args.d_model} L={args.n_layers} "
          f"steps={args.steps} seeds={args.seeds} | mem_linear={args.mem_linear} "
          f"mem_layout={args.mem_layout}", flush=True)

    results = {}
    t_start = time.time()
    for name, kind, lr in specs:
        accs, params = [], None
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = build(kind, vocab, args.d_model, args.n_layers, args.state_dim,
                      args.mem_linear, args.mem_layout)
            params = m.num_params()
            t0 = time.time()
            acc = train_on_mqar(m, steps=args.steps, n_pairs=args.pairs, n_queries=args.queries,
                                n_keys=NK, n_values=NV, batch=args.batch, lr=lr, device=dev, log_every=0)
            accs.append(acc)
            print(f"  {name:<12} seed{s} recall={acc:.3f}  ({time.time()-t0:.0f}s, {params/1e6:.2f}M)", flush=True)
        mean = statistics.mean(accs)
        results[name] = {"mean": mean, "std": statistics.pstdev(accs), "all": accs, "params": params}
        print(f"  {name:<12} MEAN={mean:.3f}  (chance {chance:.3f})", flush=True)

    ssm = results["SSM-only"]["mean"]
    ours = results["OURS+mem"]["mean"]
    xf = results["Transformer"]["mean"]
    print(f"\n=== P={args.pairs} summary (chance~{chance:.3f}) ===", flush=True)
    print(f"  SSM-only    {ssm:.3f}", flush=True)
    print(f"  OURS+mem    {ours:.3f}   (gap over SSM: {ours-ssm:+.3f})", flush=True)
    print(f"  Transformer {xf:.3f}", flush=True)
    if ours - ssm > 0.1 and ours >= 0.5 * (ssm + xf):
        verdict = "PASS: memory separates from SSM-only and tracks the transformer"
    elif ours - ssm > 0.05:
        verdict = "WEAK: memory helps but does not approach the transformer"
    else:
        verdict = "FAIL: memory does not carry recall over SSM-only"
    print(f"  verdict: {verdict}", flush=True)
    print(f"(total {time.time()-t_start:.0f}s)", flush=True)

    payload = {
        "status": "ok", "device": dev, "pairs": args.pairs, "queries": args.queries,
        "keys": NK, "values": NV, "chance": chance, "steps": args.steps, "seeds": args.seeds,
        "d_model": args.d_model, "n_layers": args.n_layers, "mem_linear": args.mem_linear,
        "mem_layout": args.mem_layout,
        "results": results, "verdict": verdict, "total_s": time.time() - t_start,
    }
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
