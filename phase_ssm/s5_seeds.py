"""Multi-seed confirmation of the contested claim: settling > GRU on noisy-S5
length generalization (i.e. the adaptive-depth structure beats a plain recurrence,
not just ties it). Fixed-depth (transformer, diag-SSM) cliff-collapse is already
established and seed-robust, so this focuses compute on settling vs GRU.
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch

from phase_ssm.equilibrium import SettlingLM, make_noisy_batch
from phase_ssm.statetrack import GRULM


def train(model, L, dev, *, steps, settling, gen_prob=0.5, ND=30, batch=64, lr=1e-3, warmup=400):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, tgt = make_noisy_batch(batch, L, gen_prob, ND, dev)
        _, loss = model(x, tgt, iters=L) if settling else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_at(model, Le, dev, *, settling, gen_prob=0.5, ND=30, n=1024):
    model.eval()
    x, tgt = make_noisy_batch(n, Le, gen_prob, ND, dev)
    logits = model(x, iters=Le)[0] if settling else model(x)[0]
    return (logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--ND", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 120 + args.ND

    print(f"S5 SEEDS | settling vs GRU | train_L={args.train_L} steps={args.steps} "
          f"seeds={args.seeds} d={args.d} dev={dev}", flush=True)

    res = {"settling": {Le: [] for Le in args.eval_Ls}, "GRU": {Le: [] for Le in args.eval_Ls}}
    for s in range(args.seeds):
        torch.manual_seed(s)
        sett = train(SettlingLM(vocab, d=args.d, iters=args.train_L), args.train_L, dev,
                     steps=args.steps, settling=True, ND=args.ND)
        torch.manual_seed(s)
        gru = train(GRULM(vocab, args.d, 4), args.train_L, dev, steps=args.steps, settling=False, ND=args.ND)
        for Le in args.eval_Ls:
            res["settling"][Le].append(acc_at(sett, Le, dev, settling=True, ND=args.ND))
            res["GRU"][Le].append(acc_at(gru, Le, dev, settling=False, ND=args.ND))
        print(f"  seed {s} done | settling@2x={res['settling'][min(2*args.train_L, max(args.eval_Ls))][-1]:.3f}", flush=True)

    print(f"\n{'L_eval':>7} | {'settling mean±std':>20} | {'GRU mean±std':>18} | {'gap':>6}", flush=True)
    for Le in args.eval_Ls:
        sm, ss = np.mean(res['settling'][Le]), np.std(res['settling'][Le])
        gm, gs = np.mean(res['GRU'][Le]), np.std(res['GRU'][Le])
        print(f"{Le:>7} | {sm:>10.3f} ± {ss:<7.3f} | {gm:>8.3f} ± {gs:<6.3f} | {sm-gm:>+6.3f}", flush=True)
    print("S5 SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
