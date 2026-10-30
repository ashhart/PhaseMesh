"""Rung B — conditional composition: the separation task for ITERATED scans.

The capability boundary of any input-dependent linear RNN (Rung A's scan
included): the transition A_t may depend only on x_t, never on state. This task
puts the state inside the transition: each generator token acts as g or g^-1
depending on a non-homomorphic predicate of the CURRENT running state
(phi(s) = [s fixes element 0]). The transition needed at position t is a
function of the true prefix state, so no single scan — and no fixed stack of k
scans beyond its k conditioning rounds — computes it for growing L.

DreamScan: one weight-tied NonAbelianLayer ITERATED S times (Universal
Transformer / DEQ style). Sweep s recomputes every transition from sweep
s-1's output, so each sweep buys one global round of state-conditioning at
O(log L) parallel depth. S is a free dial at eval: train at S_train, evaluate
deeper. This is DREAM's two-timescale thesis with the oscillator replaced by
the non-abelian scan — the settling result and the scan result composed into
one object.

Controls (pre-registered in results/rungB_conditional_composition.md):
  scan S=1 (2-layer)   must fail — the class boundary
  untied 4-layer scan  matched compute, fixed conditioning depth — the
                       "is iteration just depth?" control
  GRU                  positive control / capability ceiling (serial automaton)

Run:  python -m phase_ssm.conditional_composition --seeds 3 --json results/rungB_condcomp.json
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonabelian_scan import NonAbelianLayer, pick_device, train_model
from .statetrack import GRULM, IDENTITY, MUL, PERMS, IDX, GENS

# inverse table and the mode predicate (non-homomorphic: not a function of
# token counts, only of the actual running state)
INV = torch.tensor([IDX[tuple(sorted(range(5), key=lambda i: PERMS[j][i]))] for j in range(120)])
FIX0 = torch.tensor([1 if PERMS[j][0] == 0 else 0 for j in range(120)])


def make_conditional_batch(B, L, gen_prob, ND, device):
    """Noisy conditional-S5: generator token g applies g when phi(state)=1
    (state fixes element 0) and g^-1 otherwise; distractors leave state alone.
    Targets are the running state at every position (chance 1/120)."""
    mul, inv, fix0 = MUL.to(device), INV.to(device), FIX0.to(device)
    gens = torch.tensor(GENS, device=device)
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)
    state = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
    for t in range(L):
        is_gen = torch.rand(B, device=device) < gen_prob
        gtok = gens[torch.randint(0, len(GENS), (B,), device=device)]
        dtok = 120 + torch.randint(0, ND, (B,), device=device)
        x[:, t] = torch.where(is_gen, gtok, dtok)
        eff = torch.where(fix0[state] == 1, gtok, inv[gtok])
        state = torch.where(is_gen, mul[state, eff], state)
        tgt[:, t] = state
    return x, tgt


class DreamScanLM(nn.Module):
    """One weight-tied non-abelian scan block, iterated: sweeps ARE
    conditioning depth, each at O(log L) parallel cost."""

    def __init__(self, vocab, d=128, H=8, d_h=16, n_h=4, S_train=4, beta_init="identity"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.block = NonAbelianLayer(d, H, d_h, n_h, "householder", True, beta_init)
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.S_train = S_train

    def forward(self, ids, targets=None, sweeps=None):
        z = self.emb(ids)
        for _ in range(sweeps or self.S_train):
            z = self.block(z)
        logits = self.head(self.norm_f(z))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_cc(model, L, dev, *, steps, sweeps=None, gen_prob=0.5, ND=30,
             batch=64, lr=1e-3, warmup=400):
    """Rung A protocol on the conditional task (same hygiene param groups)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 or n.endswith("h0") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98))
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, tgt = make_conditional_batch(batch, L, gen_prob, ND, dev)
        _, loss = model(x, tgt, sweeps=sweeps) if sweeps is not None else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_cc(model, Le, dev, *, sweeps=None, gen_prob=0.5, ND=30, n=1024, chunk=256):
    model.eval()
    hits = total = 0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        x, tgt = make_conditional_batch(m, Le, gen_prob, ND, dev)
        logits = model(x, sweeps=sweeps)[0] if sweeps is not None else model(x)[0]
        hits += (logits.argmax(-1)[:, -1] == tgt[:, -1]).sum().item()
        total += m
    return hits / total


def main():
    from .nonabelian_scan import NonAbelianScanLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--sweep-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--S-train", type=int, default=4)
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    vocab = 120 + args.ND
    steps = 300 if args.quick else args.steps
    seeds = 1 if args.quick else args.seeds
    eval_n = 256 if args.quick else 1024
    d_h = args.d // 8

    print(f"RUNG B | conditional composition | train_L={args.train_L} steps={steps} "
          f"seeds={seeds} S_train={args.S_train} beta_init={args.beta_init} dev={dev} "
          f"| chance {1/120:.4f}", flush=True)

    arms = {
        f"dreamscan S={args.S_train}": lambda: DreamScanLM(
            vocab, d=args.d, H=8, d_h=d_h, n_h=args.n_h, S_train=args.S_train,
            beta_init=args.beta_init),
        "scan 1-sweep 2-layer": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=2, H=8, d_h=d_h, n_h=args.n_h, beta_init=args.beta_init),
        "scan untied 4-layer": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=4, H=8, d_h=d_h, n_h=args.n_h, beta_init=args.beta_init),
        "GRU": lambda: GRULM(vocab, args.d, 4),
    }
    res = {name: {Le: [] for Le in args.eval_Ls} for name in arms}
    grid = {Le: {S: [] for S in args.sweep_grid} for Le in args.eval_Ls}
    for s in range(seeds):
        for name, ctor in arms.items():
            torch.manual_seed(s)
            m = ctor()
            dream = isinstance(m, DreamScanLM)
            t0 = time.time()
            train_cc(m, args.train_L, dev, steps=steps,
                     sweeps=args.S_train if dream else None, ND=args.ND)
            dt = time.time() - t0
            for Le in args.eval_Ls:
                res[name][Le].append(acc_cc(m, Le, dev, sweeps=args.S_train if dream else None,
                                            ND=args.ND, n=eval_n))
                if dream:
                    for S in args.sweep_grid:
                        grid[Le][S].append(acc_cc(m, Le, dev, sweeps=S, ND=args.ND, n=eval_n))
            print(f"  seed {s} {name}: params={m.num_params():,} train={dt:.0f}s "
                  f"acc@{args.train_L}={res[name][args.train_L][-1]:.3f} "
                  f"acc@{2*args.train_L}={res[name][2*args.train_L][-1]:.3f}", flush=True)

    print(f"\n{'L_eval':>7} | " + " | ".join(f"{n:>22}" for n in res), flush=True)
    for Le in args.eval_Ls:
        cells = [f"{np.mean(v):>8.3f} ± {np.std(v):<6.3f}" if len(v) > 1 else f"{v[0]:>8.3f}"
                 for v in (res[n][Le] for n in res)]
        print(f"{Le:>7} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
    print(f"\nDreamScan eval-sweep grid (mean over seeds; trained at S={args.S_train}):", flush=True)
    print(f"{'L_eval':>7} | " + " | ".join(f"S={S:>4}" for S in args.sweep_grid), flush=True)
    for Le in args.eval_Ls:
        print(f"{Le:>7} | " + " | ".join(f"{np.mean(grid[Le][S]):>6.3f}" for S in args.sweep_grid),
              flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "steps": steps, "seeds": seeds, "S_train": args.S_train,
                       "n_h": args.n_h, "beta_init": args.beta_init,
                       "results": {n: {str(Le): v for Le, v in r.items()} for n, r in res.items()},
                       "sweep_grid": {str(Le): {str(S): v for S, v in g.items()}
                                      for Le, g in grid.items()}}, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("RUNG B DONE", flush=True)


if __name__ == "__main__":
    main()
