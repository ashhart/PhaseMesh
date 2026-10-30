"""Nonlinear settling (equilibrium) model — the post-transformer capability bet.

The hypothesis (from the post-transformer survey): fixed-depth models (transformers
AND diagonal SSMs) are in TC0 and cannot track S_5 state at growing length. A model
that *settles* — iterates a non-diagonal nonlinear block toward a fixed point, a
VARIABLE number of iterations — has effective depth set by the iteration count, not
the layer count, so it can escape TC0. The signature of a real win: accuracy RISES
with settling iterations on S_5 and matches the GRU, where transformer/diag-SSM
(measured 0.04 / 0.29 at L=64) plateau.

Design: a damped relaxation z_{k+1} = (1-a)·z_k + a·f(z_k, inj), where f mixes each
position with its causal neighbour (look-back-1) through a nonlinear MLP. One
iteration propagates state one position, so K iterations propagate K steps —
iterations ARE depth. Damping + LayerNorm keep the unrolled settle trainable.

Controls baked in: --linearize replaces the nonlinearity (if accuracy survives,
the model is secretly linear attention → result void); iteration sweep at eval.
"""
from __future__ import annotations
import argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .statetrack import make_batch, VOCAB, build as build_baseline, train_eval


def causal_shift(z):
    return F.pad(z, (0, 0, 1, 0))[:, :-1, :]   # look-back by 1 position (causal)


class SettlingCell(nn.Module):
    def __init__(self, d, hidden=None, linearize=False):
        super().__init__()
        h = hidden or 2 * d
        self.norm = nn.LayerNorm(d)
        self.lin = linearize
        if linearize:                                  # control: no nonlinearity
            self.f = nn.Linear(3 * d, d)
        else:
            self.f = nn.Sequential(nn.Linear(3 * d, h), nn.SiLU(), nn.Linear(h, d))
        self.alpha = nn.Parameter(torch.tensor(-0.8))  # sigmoid(-0.8)~0.31 damped step

    def forward(self, z, inj):
        zin = self.norm(z)
        tgt = self.f(torch.cat([zin, causal_shift(zin), inj], dim=-1))
        a = torch.sigmoid(self.alpha)
        return (1 - a) * z + a * tgt                   # damped relaxation toward f's fixed point


class SettlingLM(nn.Module):
    def __init__(self, vocab, d=128, hidden=256, iters=64, linearize=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cell = SettlingCell(d, hidden, linearize=linearize)
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.iters = iters

    def forward(self, ids, targets=None, iters=None):
        inj = self.emb(ids)
        z = torch.zeros_like(inj)
        for _ in range(iters or self.iters):
            z = self.cell(z, inj)
        logits = self.head(self.norm_f(z))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class DEQSettlingLM(nn.Module):
    """Proper equilibrium form: solve z* = cell(z*, inj) to convergence under no_grad
    (adaptive iteration count), then backprop through a short differentiable tail
    from the fixed point — O(1) memory w.r.t. the solve length, not O(K) BPTT.
    Length generalization is automatic: a longer sequence just needs more iterations
    to converge (adaptive compute)."""
    def __init__(self, vocab, d=128, hidden=256, max_iters=300, tol=1e-3, tail=6, linearize=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cell = SettlingCell(d, hidden, linearize=linearize)
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.max_iters, self.tol, self.tail = max_iters, tol, tail

    def forward(self, ids, targets=None, return_iters=False):
        inj = self.emb(ids)
        z = torch.zeros_like(inj)
        used = self.max_iters
        with torch.no_grad():
            for k in range(self.max_iters):
                zn = self.cell(z, inj)
                res = (zn - z).norm() / (z.norm() + 1e-6)
                z = zn
                if res < self.tol:
                    used = k + 1
                    break
        z = z.detach()
        for _ in range(self.tail):                    # differentiable tail from the fixed point
            z = self.cell(z, inj)
        logits = self.head(self.norm_f(z))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        if return_iters:
            return logits, loss, used
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_deq(model, L, device, steps=6000, lr=1e-3, batch=64, warmup=400):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(device)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        ids, tgt = make_batch(batch, L, device)
        _, loss = model(ids, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_deq(model, L, device, n=1024):
    model.eval()
    ids, tgt = make_batch(n, L, device)
    logits, _, used = model(ids, return_iters=True)
    acc = (logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()
    return acc, used


def run_deq(train_L, eval_Ls, dev, d=128, steps=6000, linearize=False):
    tag = "DEQ-LINEARIZED(control)" if linearize else "DEQ settling (implicit, adaptive convergence)"
    print(f"{tag} | train_L={train_L} | reports acc + AUTO converged-iters per length", flush=True)
    torch.manual_seed(0)
    m = DEQSettlingLM(VOCAB, d=d, linearize=linearize)
    print(f"params={m.num_params():,}  (memory cost = tail only, not K)", flush=True)
    train_deq(m, train_L, dev, steps=steps)
    print(f"{'L_eval':>7} | {'DEQ acc':>9} | {'auto-iters':>10}", flush=True)
    for Le in eval_Ls:
        acc, used = eval_deq(m, Le, dev)
        print(f"{Le:>7} | {acc:>9.3f} | {used:>10}", flush=True)
    print("reference (unrolled settling 3-seed): L=64 0.99 | L=96 0.97 | L=128 0.94 ; GRU L=128 0.49", flush=True)
    print("DONE", flush=True)


def train_settling(model, L, device, steps=6000, lr=1e-3, batch=64, warmup=400, train_iters=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(device)
    K = train_iters or L
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        ids, tgt = make_batch(batch, L, device)
        _, loss = model(ids, tgt, iters=K)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_at(model, L, device, iters, n=1024):
    model.eval()
    ids, tgt = make_batch(n, L, device)
    pred = model(ids, iters=iters)[0].argmax(-1)
    return (pred[:, -1] == tgt[:, -1]).float().mean().item()


def length_gen(train_L, eval_Ls, dev, d=128, steps=6000):
    """Train short, test long. Settling settles MORE iters for longer L; GRU just runs.
    The decisive 'does settling beat a plain recurrence' test: adaptive-compute
    length generalization a fixed-depth model structurally cannot do."""
    from .statetrack import GRULM, train_eval
    print(f"LENGTH-GEN | train_L={train_L} | eval at {eval_Ls} | settling settles iters=L_eval", flush=True)
    torch.manual_seed(0)
    sett = SettlingLM(VOCAB, d=d, iters=train_L)
    train_settling(sett, train_L, dev, steps=steps, train_iters=train_L)
    torch.manual_seed(0)
    gru = GRULM(VOCAB, d, 4)
    train_eval(gru, train_L, dev, steps=steps)   # trains in place
    print(f"{'L_eval':>7} | {'settling(iters=L)':>18} | {'GRU':>8}", flush=True)
    for Le in eval_Ls:
        s = eval_at(sett, Le, dev, iters=Le)
        gru.eval()
        with torch.no_grad():
            ids, tgt = make_batch(1024, Le, dev)
            g = (gru(ids)[0].argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()
        flag = "  <-- settling generalizes, GRU doesn't" if (s > 0.8 and g < 0.5) else ""
        print(f"{Le:>7} | {s:>18.3f} | {g:>8.3f}{flag}", flush=True)
    print("DONE", flush=True)


def make_noisy_batch(B, L, gen_prob, ND, device):
    """Noisy S_5: each position is a group generator (prob gen_prob, updates state)
    or a distractor token (state unchanged). State signal is sparse + buried in noise
    — the language-like setting. Distractor ids live in [120, 120+ND)."""
    from .statetrack import MUL, IDENTITY, GENS
    mul = MUL.to(device)
    gens = torch.tensor(GENS, device=device)
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)
    state = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
    for t in range(L):
        is_gen = torch.rand(B, device=device) < gen_prob
        gtok = gens[torch.randint(0, len(GENS), (B,), device=device)]
        dtok = 120 + torch.randint(0, ND, (B,), device=device)
        x[:, t] = torch.where(is_gen, gtok, dtok)
        state = torch.where(is_gen, mul[state, gtok], state)
        tgt[:, t] = state
    return x, tgt


def run_noisy(train_L, eval_Ls, dev, d=128, steps=6000, gen_prob=0.5, ND=30):
    from .statetrack import GRULM
    from .model import PhaseSSMConfig, PhaseSSMLM
    from .transformer import TransformerConfig, TransformerLM
    vocab = 120 + ND
    print(f"NOISY-S5 (capability-in-language) | train_L={train_L} | gen_prob={gen_prob} ND={ND} | "
          f"state signal sparse+buried; settling vs transformer vs diag-SSM vs GRU", flush=True)

    def train(model, L, settling=False):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.01)
        model.to(dev)
        for step in range(steps):
            cur = 1e-3 * (step + 1) / 400 if step < 400 else \
                1e-3 * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 400) / max(1, steps - 400))))
            for g in opt.param_groups:
                g["lr"] = cur
            model.train()
            x, tgt = make_noisy_batch(64, L, gen_prob, ND, dev)
            _, loss = model(x, tgt, iters=L) if settling else model(x, tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        return model

    torch.manual_seed(0); sett = train(SettlingLM(vocab, d=d, iters=train_L), train_L, settling=True)
    torch.manual_seed(0); ssm = train(PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=d, n_layers=4, state_dim=64, expand=1, d_ff_mult=2, use_phase_memory=False)), train_L)
    torch.manual_seed(0); xf = train(TransformerLM(TransformerConfig(vocab_size=vocab, d_model=d, n_layers=4, n_heads=4, d_ff_mult=2, block_size=max(eval_Ls))), train_L)
    torch.manual_seed(0); gru = train(GRULM(vocab, d, 4), train_L)

    print(f"{'L_eval':>7} | {'settling':>9} | {'transformer':>11} | {'diag-SSM':>9} | {'GRU':>6}", flush=True)
    for Le in eval_Ls:
        x, tgt = make_noisy_batch(1024, Le, gen_prob, ND, dev)
        row = []
        for name, m, it in [("s", sett, Le), ("xf", xf, None), ("ssm", ssm, None), ("g", gru, None)]:
            m.eval()
            with torch.no_grad():
                logits = m(x, iters=it)[0] if it is not None else m(x)[0]
            row.append((logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item())
        print(f"{Le:>7} | {row[0]:>9.3f} | {row[1]:>11.3f} | {row[2]:>9.3f} | {row[3]:>6.3f}", flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--train-iters", type=int, default=None, help="settle iterations during training (default L)")
    ap.add_argument("--linearize", action="store_true", help="control: remove nonlinearity")
    ap.add_argument("--lengthgen", action="store_true", help="train short, test long: settling vs GRU")
    ap.add_argument("--deq", action="store_true", help="implicit DEQ settling (adaptive convergence, O(1) mem)")
    ap.add_argument("--noisy", action="store_true", help="noisy-S5 capability-in-language: state buried in distractors")
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.noisy:
        run_noisy(args.train_L, args.eval_Ls, dev, d=args.d, steps=args.steps)
        return
    if args.deq:
        run_deq(args.train_L, args.eval_Ls, dev, d=args.d, steps=args.steps, linearize=args.linearize)
        return
    if args.lengthgen:
        length_gen(args.train_L, args.eval_Ls, dev, d=args.d, steps=args.steps)
        return
    L = args.L
    K_train = args.train_iters or L
    tag = "settling-LINEARIZED(control)" if args.linearize else "settling"
    print(f"S_5 settling @ L={L} | device={dev} | train_iters={K_train} | {tag} | chance {1/120:.3f}", flush=True)

    torch.manual_seed(0)
    m = SettlingLM(VOCAB, d=args.d, iters=K_train, linearize=args.linearize)
    print(f"params={m.num_params():,}", flush=True)
    train_settling(m, L, dev, steps=args.steps, train_iters=K_train)

    print("=== iteration sweep (does accuracy rise with settling iterations?) ===", flush=True)
    for it in [1, 2, 4, 8, 16, 32, L, L + L // 2, 2 * L]:
        print(f"  iters={it:>4}  last-token acc {eval_at(m, L, dev, it):.3f}", flush=True)
    print(f"reference @ L={L}: transformer 0.04 | diag-SSM 0.29 | GRU 1.00", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
