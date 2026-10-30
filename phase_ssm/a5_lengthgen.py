"""Generality check: does the serial-depth length-gen result hold on a SECOND
non-solvable group (A_5), not just S_5? Same noisy state-tracking setup, settling
vs transformer vs diag-SSM vs GRU, train short / eval long. Self-contained group
tables so it cannot affect the banked S_5 code.

A_5 (alternating group on 5, the even permutations) is non-solvable (its word
problem is NC1-complete), so fixed-depth models should again collapse past the
training length while recurrent-depth extrapolates.
"""
from __future__ import annotations
import argparse, itertools, math
import torch

from .equilibrium import SettlingLM
from .statetrack import GRULM
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM


def _parity(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j]) % 2


PERMS = [p for p in itertools.permutations(range(5)) if _parity(p) == 0]   # 60 even perms
IDX = {p: i for i, p in enumerate(PERMS)}
def _compose(a, b):
    return tuple(b[a[i]] for i in range(5))                                # a THEN b
MUL = torch.tensor([[IDX[_compose(PERMS[i], PERMS[j])] for j in range(60)] for i in range(60)])
IDENTITY = IDX[tuple(range(5))]
GENS = [IDENTITY, IDX[(1, 2, 0, 3, 4)], IDX[(1, 2, 3, 4, 0)]]              # id, 3-cycle, 5-cycle
NG = 60


def subgroup_size():
    """BFS-close the generators; should be 60 if they generate all of A_5."""
    seen = {IDENTITY}
    frontier = [IDENTITY]
    gens = [g for g in GENS if g != IDENTITY]
    while frontier:
        s = frontier.pop()
        for g in gens:
            t = int(MUL[s, g])
            if t not in seen:
                seen.add(t); frontier.append(t)
    return len(seen)


def make_noisy_a5(B, L, gen_prob, ND, device):
    mul = MUL.to(device)
    gens = torch.tensor(GENS, device=device)
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)
    state = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
    for t in range(L):
        is_gen = torch.rand(B, device=device) < gen_prob
        gtok = gens[torch.randint(0, len(GENS), (B,), device=device)]
        dtok = NG + torch.randint(0, ND, (B,), device=device)
        x[:, t] = torch.where(is_gen, gtok, dtok)
        state = torch.where(is_gen, mul[state, gtok], state)
        tgt[:, t] = state
    return x, tgt


def train(model, L, dev, *, steps, settling, gen_prob, ND, batch=64, lr=1e-3, warmup=400):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, tgt = make_noisy_a5(batch, L, gen_prob, ND, dev)
        _, loss = model(x, tgt, iters=L) if settling else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_at(model, Le, dev, *, settling, gen_prob, ND, n=1024):
    model.eval()
    x, tgt = make_noisy_a5(n, Le, gen_prob, ND, dev)
    logits = model(x, iters=Le)[0] if settling else model(x)[0]
    return (logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gen-prob", type=float, default=0.5)
    ap.add_argument("--ND", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = NG + args.ND
    sg = subgroup_size()
    print(f"A5 LENGTH-GEN | generators close to subgroup of size {sg} (A_5 = 60) | "
          f"train_L={args.train_L} steps={args.steps} d={args.d} gen_prob={args.gen_prob} dev={dev}", flush=True)

    torch.manual_seed(0)
    sett = train(SettlingLM(vocab, d=args.d, iters=args.train_L), args.train_L, dev,
                 steps=args.steps, settling=True, gen_prob=args.gen_prob, ND=args.ND)
    torch.manual_seed(0)
    xf = train(TransformerLM(TransformerConfig(vocab_size=vocab, d_model=args.d, n_layers=4, n_heads=4,
               d_ff_mult=2, block_size=max(args.eval_Ls), pos_enc="rope")), args.train_L, dev,
               steps=args.steps, settling=False, gen_prob=args.gen_prob, ND=args.ND)
    torch.manual_seed(0)
    ssm = train(PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=args.d, n_layers=4, state_dim=64,
                expand=1, d_ff_mult=2, use_phase_memory=False)), args.train_L, dev,
                steps=args.steps, settling=False, gen_prob=args.gen_prob, ND=args.ND)
    torch.manual_seed(0)
    gru = train(GRULM(vocab, args.d, 4), args.train_L, dev,
                steps=args.steps, settling=False, gen_prob=args.gen_prob, ND=args.ND)

    print(f"{'L_eval':>7} | {'settling':>9} | {'transformer':>11} | {'diag-SSM':>9} | {'GRU':>6}", flush=True)
    for Le in args.eval_Ls:
        s = acc_at(sett, Le, dev, settling=True, gen_prob=args.gen_prob, ND=args.ND)
        t = acc_at(xf, Le, dev, settling=False, gen_prob=args.gen_prob, ND=args.ND)
        d = acc_at(ssm, Le, dev, settling=False, gen_prob=args.gen_prob, ND=args.ND)
        g = acc_at(gru, Le, dev, settling=False, gen_prob=args.gen_prob, ND=args.ND)
        print(f"{Le:>7} | {s:>9.3f} | {t:>11.3f} | {d:>9.3f} | {g:>6.3f}", flush=True)
    print("A5 DONE", flush=True)


if __name__ == "__main__":
    main()
