"""Differentiator experiment: does the phase-binding memory close the SSM recall gap?

For each difficulty (pairs P), train SSM-only, OURS (SSM+phase-memory), and a
transformer to convergence, on the same MQAR. The gold pattern we're hunting:
as P grows past the SSM's fixed-state capacity, SSM-only collapses while OURS and
the transformer hold — i.e. the memory gives the SSM the recall it structurally
lacks, at O(L) cost. Per-arch LR (transformer needs ~3e-4 for induction heads;
SSM tolerates 1e-3) — justified by an LR-sensitivity finding, not cherry-picking.
"""
from __future__ import annotations
import argparse, json, statistics
import torch
from phase_ssm.tasks import train_on_mqar
from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.transformer import TransformerConfig, TransformerLM


def build(kind, vocab, state_dim, use_mem):
    if kind == "xf":
        return TransformerLM(TransformerConfig(vocab_size=vocab, d_model=192, n_layers=4,
                                               n_heads=4, d_ff_mult=2, block_size=512))
    return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=192, n_layers=4, state_dim=state_dim,
                                     expand=1, d_ff_mult=2, use_phase_memory=use_mem,
                                     mem_heads=4, mem_head_dim=48))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/mqar_sweep.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NK = NV = 128
    vocab = 1 + NK + NV
    specs = [("SSM-only", "ssm", False, 1e-3), ("OURS+mem", "ssm", True, 1e-3),
             ("Transformer", "xf", False, 3e-4)]
    print(f"MQAR differentiator sweep | state_dim={args.state_dim} | steps={args.steps} | "
          f"queries={args.queries} | keys/vals={NK} | seeds={args.seeds}", flush=True)
    results = {}
    for P in args.pairs:
        results[str(P)] = {}
        for name, kind, mem, lr in specs:
            accs = []
            for s in range(args.seeds):
                torch.manual_seed(s)
                m = build(kind, vocab, args.state_dim, mem)
                acc = train_on_mqar(m, steps=args.steps, n_pairs=P, n_queries=args.queries,
                                    n_keys=NK, n_values=NV, batch=64, lr=lr, device=dev, log_every=0)
                accs.append(acc)
                print(f"  P={P:<3} {name:<12} seed{s} recall={acc:.3f}", flush=True)
            mean = statistics.mean(accs)
            std = statistics.pstdev(accs)
            results[str(P)][name] = {"mean": mean, "std": std, "max": max(accs), "all": accs}
            print(f"  P={P:<3} {name:<12} MEAN={mean:.3f} +/-{std:.3f}  (params {m.num_params()/1e6:.2f}M)", flush=True)
        r = results[str(P)]
        gap = r["OURS+mem"]["mean"] - r["SSM-only"]["mean"]
        print(f"  >>> P={P}: SSM {r['SSM-only']['mean']:.3f} | OURS {r['OURS+mem']['mean']:.3f} | XF "
              f"{r['Transformer']['mean']:.3f}  (OURS-SSM gap {gap:+.3f})", flush=True)
        json.dump(results, open(args.out, "w"), indent=2)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
