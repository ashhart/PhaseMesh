"""DREAM Rung 0 — the recall go/no-go on MQAR.

Pre-registered: at pairs=16, >=4000 steps, DREAM(S=1,T>=1) must reach recall
>=0.90, beat SSM-only widely, and come within 0.05 of the transformer. Anything
else is NO-GO (the delta memory does not carry recall).

CPU smoke: small pairs/steps to confirm wiring and direction (NOT the verdict).
GPU verdict: --pairs 16 --steps 4000 --seeds 3 on a CUDA box.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from phase_ssm.dream import DreamConfig, DreamLM
from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.tasks import train_on_mqar
from phase_ssm.transformer import TransformerConfig, TransformerLM


def build(kind, vocab, d_model, n_layers, state_dim, T, dream_s=1, dream_damp=-2.0):
    if kind == "xf":
        return TransformerLM(TransformerConfig(
            vocab_size=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=max(1, d_model // 32), d_ff_mult=2, block_size=512, pos_enc="rope"))
    if kind == "dream":
        return DreamLM(DreamConfig(
            vocab_size=vocab, d_model=d_model, n_layers=n_layers, state_dim=state_dim,
            outer_iters=dream_s, inner_iters=T, mem_heads=2, mem_head_dim=16, damp_init=dream_damp))
    # additive phase-memory baseline and SSM-only reuse the existing model
    return PhaseSSMLM(PhaseSSMConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, state_dim=state_dim,
        expand=1, d_ff_mult=2, use_phase_memory=(kind == "additive"),
        mem_heads=2, mem_head_dim=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--values", type=int, default=128)
    ap.add_argument("--d-model", dest="d_model", type=int, default=64)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=2)
    ap.add_argument("--state-dim", dest="state_dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dream-s", dest="dream_s", type=int, default=1, help="DREAM outer settling steps S")
    ap.add_argument("--dream-damp", dest="dream_damp", type=float, default=-2.0, help="DREAM damp_init (sigmoid -> step size)")
    ap.add_argument("--out", default="results/dream_rung0_mqar.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NK, NV = args.keys, args.values
    vocab = 1 + NK + NV
    chance = 1.0 / NV
    # (name, kind, T, lr)
    specs = [("SSM-only", "ssm_only", 1, 1e-3), ("additive-mem", "additive", 1, 1e-3),
             ("DREAM T=1", "dream", 1, 1e-3), ("DREAM T=2", "dream", 2, 1e-3),
             ("Transformer", "xf", 1, 3e-4)]
    print(f"DREAM Rung 0 | device={dev} | P={args.pairs} NK={NK} NV={NV} "
          f"(chance~{chance:.3f}) | steps={args.steps} seeds={args.seeds}", flush=True)

    results = {}
    for name, kind, T, lr in specs:
        accs, params = [], None
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = build(kind, vocab, args.d_model, args.n_layers, args.state_dim, T,
                      dream_s=args.dream_s, dream_damp=args.dream_damp)
            params = m.num_params()
            t0 = time.time()
            acc = train_on_mqar(m, steps=args.steps, n_pairs=args.pairs, n_queries=args.queries,
                                n_keys=NK, n_values=NV, batch=args.batch, lr=lr, device=dev, log_every=0)
            accs.append(acc)
            print(f"  {name:<13} seed{s} recall={acc:.3f} ({time.time()-t0:.0f}s, {params/1e6:.2f}M)", flush=True)
        results[name] = {"mean": statistics.mean(accs), "all": accs, "params": params}
        print(f"  {name:<13} MEAN={results[name]['mean']:.3f}", flush=True)

    ssm = results["SSM-only"]["mean"]
    dream = max(results["DREAM T=1"]["mean"], results["DREAM T=2"]["mean"])
    xf = results["Transformer"]["mean"]
    # pre-registered verdict
    if dream >= 0.90 and dream - ssm > 0.3 and (xf - dream) <= 0.05:
        verdict = "GO: delta memory carries recall (>=0.90, beats SSM-only, tracks transformer)"
    elif dream - ssm > 0.1:
        verdict = "PARTIAL: delta memory helps but misses the pre-registered bar"
    else:
        verdict = "NO-GO: delta memory does not carry recall over SSM-only"
    print(f"\n=== verdict (pre-registered) ===\n  {verdict}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"device": dev, "pairs": args.pairs, "steps": args.steps, "seeds": args.seeds,
         "chance": chance, "results": results, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
