"""Composition-depth spectrum: WHY does settling's S5 advantage vanish on text?

Hypothesis: real text rarely needs composition depth that scales with
sequence length, even when dependencies are long-range. This isolates that
exact variable. A periodic RESET token forces the tracked state back to
IDENTITY every K positions, capping the sequential composition depth actually
required at K regardless of total sequence length L. K >= L => no resets fire
=> reduces EXACTLY to today's banked noisy-S5 task (internal consistency
check). Small K => depth is bounded the way we hypothesize natural text is.

Prediction: the settling-vs-diag-SSM gap should SHRINK as K shrinks. A
transformer (fixed depth ~n_layers) may start catching up once K drops below
its own layer count, since a fixed-depth model can track state within a
window no deeper than its own depth -- a falsifiable confirmation of the
actual mechanism, not just a correlation.

Self-contained (own group table) so it cannot affect the banked S5 code in
equilibrium.py / statetrack.py.
"""
from __future__ import annotations
import argparse, itertools, math
import torch

from .equilibrium import SettlingLM
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM

PERMS = list(itertools.permutations(range(5)))
IDX = {p: i for i, p in enumerate(PERMS)}
def _compose(a, b):
    return tuple(b[a[i]] for i in range(5))
MUL = torch.tensor([[IDX[_compose(PERMS[i], PERMS[j])] for j in range(120)] for i in range(120)])
IDENTITY = IDX[tuple(range(5))]
GENS = [IDENTITY, IDX[(1, 0, 2, 3, 4)], IDX[(1, 2, 3, 4, 0)]]
NG = 120


def make_periodic_reset_batch(B, L, gen_prob, ND, K, device):
    """Noisy-S5, but state forcibly resets to IDENTITY every K positions
    (signaled by an explicit RESET token in the vocab). K >= L => the reset
    condition never fires => identical in distribution to make_noisy_batch."""
    mul = MUL.to(device)
    gens = torch.tensor(GENS, device=device)
    RESET_ID = NG + ND
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)
    state = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
    for t in range(L):
        if K < L and t > 0 and t % K == 0:
            state = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
            x[:, t] = RESET_ID
        else:
            is_gen = torch.rand(B, device=device) < gen_prob
            gtok = gens[torch.randint(0, len(GENS), (B,), device=device)]
            dtok = NG + torch.randint(0, ND, (B,), device=device)
            x[:, t] = torch.where(is_gen, gtok, dtok)
            state = torch.where(is_gen, mul[state, gtok], state)
        tgt[:, t] = state
    return x, tgt


def train(model, L, K, dev, *, steps, settling, gen_prob, ND, batch=64, lr=1e-3, warmup=400):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, tgt = make_periodic_reset_batch(batch, L, gen_prob, ND, K, dev)
        _, loss = model(x, tgt, iters=L) if settling else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_at(model, Le, K, dev, *, settling, gen_prob, ND, n=1024):
    model.eval()
    x, tgt = make_periodic_reset_batch(n, Le, gen_prob, ND, K, dev)
    logits = model(x, iters=Le)[0] if settling else model(x)[0]
    return (logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 16, 10000])  # 10000 ~= "no reset"
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gen-prob", type=float, default=0.5)
    ap.add_argument("--ND", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = NG + args.ND + 1  # +1 for RESET token
    maxlen = max(args.eval_Ls)
    print(f"COMPOSITION-DEPTH SPECTRUM | train_L={args.train_L} eval_Ls={args.eval_Ls} "
          f"Ks={args.Ks} steps={args.steps} d={args.d} dev={dev}", flush=True)
    print("K>=train_L behaves as 'no reset' (today's unbounded task, the consistency check); "
          "small K caps required composition depth regardless of L.", flush=True)

    rows = []
    for K in args.Ks:
        print(f"\n--- K={K} ---", flush=True)
        torch.manual_seed(0)
        sett = train(SettlingLM(vocab, d=args.d, iters=args.train_L), args.train_L, K, dev,
                     steps=args.steps, settling=True, gen_prob=args.gen_prob, ND=args.ND)
        torch.manual_seed(0)
        ssm = train(PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=args.d, n_layers=4, state_dim=64,
                    expand=1, d_ff_mult=2, use_phase_memory=False)), args.train_L, K, dev,
                    steps=args.steps, settling=False, gen_prob=args.gen_prob, ND=args.ND)
        torch.manual_seed(0)
        xf = train(TransformerLM(TransformerConfig(vocab_size=vocab, d_model=args.d, n_layers=4, n_heads=4,
                   d_ff_mult=2, block_size=maxlen, pos_enc="rope")), args.train_L, K, dev,
                   steps=args.steps, settling=False, gen_prob=args.gen_prob, ND=args.ND)

        for Le in args.eval_Ls:
            s = acc_at(sett, Le, K, dev, settling=True, gen_prob=args.gen_prob, ND=args.ND)
            d = acc_at(ssm, Le, K, dev, settling=False, gen_prob=args.gen_prob, ND=args.ND)
            t = acc_at(xf, Le, K, dev, settling=False, gen_prob=args.gen_prob, ND=args.ND)
            gap = s - d
            print(f"  K={K:<6} L={Le:<4} | settling={s:.3f} diag-SSM={d:.3f} "
                  f"transformer={t:.3f} | gap(settle-ssm)={gap:+.3f}", flush=True)
            rows.append((K, Le, s, d, t, gap))

    print("\n=== SUMMARY: settling-vs-diagSSM gap at max eval length, by K ===", flush=True)
    print(f"{'K':>8} | {'settling':>9} | {'diag-SSM':>9} | {'transformer':>11} | {'gap':>7}", flush=True)
    for K, Le, s, d, t, gap in rows:
        if Le == maxlen:
            print(f"{K:>8} | {s:>9.3f} | {d:>9.3f} | {t:>11.3f} | {gap:>+7.3f}", flush=True)
    print("COMPOSITION-DEPTH DONE", flush=True)


if __name__ == "__main__":
    main()
