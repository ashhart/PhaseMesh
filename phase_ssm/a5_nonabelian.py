"""Rung A secondary (fires on GO): A_5 replication of the non-abelian scan result.

Same noisy state-tracking as a5_lengthgen.py (self-contained A_5 tables there),
nonabelian-scan vs GRU anchor, train short / eval long. A_5's generators are a
3-cycle (= 2 reflections) and a 5-cycle (= 4 reflections), so n_h=4 suffices
constructively, exactly as on S_5.

Run: python -m phase_ssm.a5_nonabelian --seeds 3 --json results/rungA_a5.json
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from .a5_lengthgen import NG, acc_at, make_noisy_a5, train
from .nonabelian_scan import NonAbelianScanLM, pick_device
from .statetrack import GRULM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gen-prob", type=float, default=0.5)
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    vocab = NG + args.ND
    print(f"A5 NONABELIAN | train_L={args.train_L} steps={args.steps} seeds={args.seeds} "
          f"n_h={args.n_h} layers={args.layers} dev={dev} | chance {1/60:.3f}", flush=True)

    arms = {
        f"nonabelian n_h={args.n_h}": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=args.layers, H=8, d_h=args.d // 8, n_h=args.n_h,
            beta_init=args.beta_init),
        "GRU": lambda: GRULM(vocab, args.d, 4),
    }
    res = {name: {Le: [] for Le in args.eval_Ls} for name in arms}
    for s in range(args.seeds):
        for name, ctor in arms.items():
            torch.manual_seed(s)
            m = train(ctor(), args.train_L, dev, steps=args.steps, settling=False,
                      gen_prob=args.gen_prob, ND=args.ND)
            for Le in args.eval_Ls:
                res[name][Le].append(acc_at(m, Le, dev, settling=False,
                                            gen_prob=args.gen_prob, ND=args.ND))
            print(f"  seed {s} {name}: acc@{args.train_L}={res[name][args.train_L][-1]:.3f} "
                  f"acc@{2 * args.train_L}={res[name][2 * args.train_L][-1]:.3f}", flush=True)

    print(f"\n{'L_eval':>7} | " + " | ".join(f"{n:>22}" for n in res), flush=True)
    for Le in args.eval_Ls:
        cells = [f"{np.mean(v):>8.3f} ± {np.std(v):<6.3f}" if len(v) > 1 else f"{v[0]:>8.3f}"
                 for v in (res[n][Le] for n in res)]
        print(f"{Le:>7} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
    print("logged single-run reference (docs/serial_depth_report.md): "
          "settling@64 0.985 @128 0.807 | GRU@64 0.932 @128 0.385", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "steps": args.steps, "seeds": args.seeds,
                       "n_h": args.n_h, "layers": args.layers,
                       "results": {n: {str(Le): v for Le, v in r.items()}
                                   for n, r in res.items()}}, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("A5 NONABELIAN DONE", flush=True)


if __name__ == "__main__":
    main()
