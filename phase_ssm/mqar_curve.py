"""Focused SSM-only vs OURS+mem MQAR with learning curves.

The first signal run (mqar_signal.py at 800 steps) showed all three configs
clustered near chance — including the transformer — which means 800 steps was
pre-convergence, not a real verdict. This script runs only the differentiator
that matters (SSM-only vs OURS+mem) at higher step counts with periodic eval, so
the learning curves show whether OURS crosses a recall threshold that SSM-only
stays stuck under. A real separation appears as OURS climbing while SSM-only
flattens near chance.

CPU-feasible: one seed, periodic eval, prints a curve we can watch land.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.tasks import make_mqar, recall_accuracy


def train_with_curve(model, *, steps, eval_every, n_pairs, n_queries, n_keys, n_values,
                     batch, lr, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(device)
    warmup = min(1000, steps // 10)
    # fixed eval sets for stable curves
    xv, yv, _ = make_mqar(256, n_pairs, n_queries, n_keys, n_values, device, seed=12345)
    curve = []
    torch.manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        if step < warmup:
            cur_lr = lr * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            cur_lr = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        x, y, _ = make_mqar(batch, n_pairs, n_queries, n_keys, n_values, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            acc = recall_accuracy(model, xv, yv)
            curve.append({"step": step, "recall": acc, "lr": cur_lr})
            print(f"    step {step:5d}  recall {acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    final = recall_accuracy(model, *make_mqar(512, n_pairs, n_queries, n_keys, n_values, device, seed=99)[:2])
    return final, curve


def build(use_mem, vocab, d_model, n_layers, state_dim, mem_linear, mem_layout):
    return PhaseSSMLM(PhaseSSMConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, state_dim=state_dim,
        expand=1, d_ff_mult=2, use_phase_memory=use_mem,
        mem_heads=2, mem_head_dim=16, mem_linear=mem_linear, mem_layout=mem_layout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--values", type=int, default=128)
    ap.add_argument("--d-model", dest="d_model", type=int, default=96)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=3)
    ap.add_argument("--state-dim", dest="state_dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mem-linear", dest="mem_linear", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mem-layout", dest="mem_layout", choices=["parallel", "sequential"], default="parallel")
    ap.add_argument("--out", default="runs/mqar_curve.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NK, NV = args.keys, args.values
    vocab = 1 + NK + NV
    chance = 1.0 / NV
    print(f"MQAR curve | device={dev} | P={args.pairs} (chance~{chance:.3f}) | "
          f"d={args.d_model} L={args.n_layers} steps={args.steps} eval_every={args.eval_every}", flush=True)

    results = {}
    for name, use_mem in [("SSM-only", False), ("OURS+mem", True)]:
        torch.manual_seed(0)
        m = build(use_mem, vocab, args.d_model, args.n_layers, args.state_dim,
                  args.mem_linear, args.mem_layout)
        print(f"  [{name}] params={m.num_params()/1e6:.2f}M", flush=True)
        final, curve = train_with_curve(
            m, steps=args.steps, eval_every=args.eval_every, n_pairs=args.pairs,
            n_queries=args.queries, n_keys=NK, n_values=NV, batch=args.batch,
            lr=1e-3, device=dev, seed=0)
        results[name] = {"final": final, "curve": curve, "params": m.num_params()}
        print(f"  [{name}] FINAL recall={final:.3f}", flush=True)

    ssm = results["SSM-only"]["final"]
    ours = results["OURS+mem"]["final"]
    print(f"\n=== P={args.pairs} final (chance~{chance:.3f}) ===", flush=True)
    print(f"  SSM-only {ssm:.3f}   OURS+mem {ours:.3f}   gap {ours-ssm:+.3f}", flush=True)
    if ours - ssm > 0.15:
        verdict = "PASS: memory separates clearly from SSM-only"
    elif ours - ssm > 0.05:
        verdict = "WEAK: memory helps modestly"
    else:
        verdict = "FAIL: memory does not separate from SSM-only"
    print(f"  verdict: {verdict}", flush=True)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "ok", "device": dev, "pairs": args.pairs, "chance": chance,
        "steps": args.steps, "d_model": args.d_model, "n_layers": args.n_layers,
        "mem_linear": args.mem_linear, "mem_layout": args.mem_layout,
        "results": results, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
