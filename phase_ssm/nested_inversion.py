"""Rung B' — nested inversion: a genuinely context-free task for iterated scans.

Rung B's conditional-composition task was refuted at smoke (commit 6255299): the
rule s' = s . f(x, phi(s)) is a map indexed by INPUT ALONE, so every such task
telescopes into an input-only automaton (regular). The "state-dependence" was an
illusion. This rung corrects the target: the boundary of the finite-width scan
class is not state-dependent rules, it is UNBOUNDED RECURSION.

The task: group-valued balanced brackets with nested inversion. A close-bracket
combines the popped outer with the INVERSE of the enclosed product:

    open   -> push active; active = IDENTITY
    gen g  -> active = MUL[active, g]
    close  -> active = MUL[pop, INV[active]]      # inversion: (w)^-1 reverses order

INV of a product reverses composition order — that is a STACK, not an automaton.
The close update depends on a value BURIED in the stack, not on the current input
token, so no F_x closes it (the exact property whose absence killed Rung B). The
stack is unbounded even though the output (one group element) is finite ->
context-free (Dyck + order reversal), not regular.

Honest caveat: for any FIXED D_max the task is technically regular over
120^(D_max+1) states. The clean claim is the unbounded case, approximated here at
D_max=8 where 120^9 ~ 1e18 >> any model's hidden width, so the bars are
empirical-capacity separations, theory-motivated not theorem-claimed.

Damped DreamScan: Rung B's salvaged finding was that UNDAMPED sweep extrapolation
diverges (S=16 eval after S=4 training collapsed to 0.19-0.43). The fix is the
damped contraction z <- z + sigmoid(alpha)(block(z) - z), sigma in (0,1) =>
contraction => self-stabilizing. This is settling's machinery reborn as the
sweep stabilizer (equilibrium.py / settling_lm.py idiom), with the non-abelian
scan as the per-sweep operator. Sweep s recomputes transitions from sweep s-1's
output; S sweeps buy S rounds of stack resolution at O(S log L) parallel depth.
Predicted signature: S* ~ D_max (resolve one nesting level per sweep).

Controls (pre-registered in results/rungB_prime_nested_inversion.md):
  scan S=1 (2-layer)   the class boundary (fixed width; fails as D_max grows)
  untied 8-layer scan  matched conditioning rounds, 8x params — "is iteration
                       just depth?" control
  GRU                  NEGATIVE control: fixed-width serial model cannot track an
                       unbounded stack. (Contrast Rung A/B where GRU was the
                       positive ceiling — there the task was regular/serial.)
  invert=False         void-check: without inversion, brackets are a no-op
                       (associativity) and the task collapses to flat composition
                       (regular). A fixed-width scan MUST solve this; if not, the
                       generator is broken and the rung is void.

Run:
  python -m phase_ssm.nested_inversion --quick              # smoke (the assay)
  python -m phase_ssm.nested_inversion --seeds 3 --json results/rungB_prime.json
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonabelian_scan import NonAbelianLayer, NonAbelianScanLM, pick_device
from .statetrack import GRULM, IDENTITY, MUL, PERMS, IDX, GENS
from .conditional_composition import DreamScanLM

# Inverse-permutation index table: INV[j] = index of PERMS[j]^{-1}. Same one-liner
# as conditional_composition.py; validated by test_inverse_table (MUL[j,INV[j]]==ID).
INV = torch.tensor([IDX[tuple(sorted(range(5), key=lambda i: PERMS[j][i]))] for j in range(120)])

# Token layout: generators are GENS (3 ids in [0,120)); OPEN=120, CLOSE=121;
# distractors fill [122, 122+ND). Targets live in [0,120) (the active group element).
OPEN, CLOSE = 120, 121


def make_nested_batch(B, L, gen_prob, ND, D_max, invert, device):
    """Balanced group-valued brackets with nested inversion (or its void variant).

    Stream-task convention (matches make_noisy_batch / make_conditional_batch):
    positional args, returns (x, tgt) both (B, L) long, tgt is the active group
    element at every position (chance 1/120). Sequences end at base depth by
    construction, so tgt[:, -1] is the full reduction and requires the whole stack.

    Emission (Markov, per position): with prob gen_prob a generator advances the
    active product; of the rest, an even split among open / close / distractor,
    each constrained by the current depth (open only below D_max; close only
    above 0). A trailing "must close" override forces closes as the sequence end
    approaches so depth returns to 0 exactly at the last position.

    invert=True  -> active = MUL[pop, INV[active]] on close (context-free)
    invert=False -> active = MUL[pop,        active ] on close (brackets are a
                    no-op: pop . active = the outer context . inner product, which
                    equals flat left-to-right composition -> regular, void-check)
    """
    mul, inv = MUL.to(device), INV.to(device)
    gens = torch.tensor(GENS, device=device)
    NG = 120
    active = torch.full((B,), IDENTITY, dtype=torch.long, device=device)
    depth = torch.zeros(B, dtype=torch.long, device=device)
    stack = torch.full((B, D_max + 1), IDENTITY, dtype=torch.long, device=device)
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)

    for t in range(L):
        remaining = L - t                                     # slots left INCLUDING this one
        # Balance guarantee: after this token there are (remaining-1) slots, each
        # able to close at most one level, so depth_after must stay <= remaining-1.
        #   must_close : depth already uses every remaining slot -> this token CLOSEs
        #   forbid_open : opening would make depth+1 > remaining-1 -> no room to close
        must_close = depth >= remaining
        forbid_open = (depth + 1) >= remaining
        r = torch.rand(B, device=device)
        is_gen = (r < gen_prob) & (~must_close)
        # of the non-generator, non-forced positions: open / close / distractor
        r2 = torch.rand(B, device=device)
        branch = (r2 * 3).to(torch.long)                      # 0 open, 1 close, 2 distractor
        is_open = (~is_gen) & (branch == 0) & (depth < D_max) & (~must_close) & (~forbid_open)
        is_close = ((~is_gen) & (branch == 1) & (depth > 0)) | must_close
        is_dist = ~(is_gen | is_open | is_close)

        # --- OPEN: push active, reset active to identity, depth+1 ---
        if is_open.any():
            idx = is_open.nonzero(as_tuple=False).squeeze(-1)
            d_at = depth[idx].unsqueeze(1)
            stack[idx] = stack[idx].scatter(1, d_at, active[idx].unsqueeze(1))
            active[idx] = IDENTITY
            depth[idx] += 1
            x[idx, t] = OPEN

        # --- CLOSE: pop, combine with (inverted) active, depth-1 ---
        if is_close.any():
            idx = is_close.nonzero(as_tuple=False).squeeze(-1)
            d_below = (depth[idx] - 1).clamp(min=0).unsqueeze(1)
            pop = stack[idx].gather(1, d_below).squeeze(1)
            combined = torch.where(
                torch.tensor(invert, device=device),
                mul[pop, inv[active[idx]]],
                mul[pop, active[idx]],
            )
            active[idx] = combined
            depth[idx] = torch.clamp(depth[idx] - 1, min=0)
            x[idx, t] = CLOSE

        # --- GEN: advance active by the drawn generator ---
        if is_gen.any():
            idx = is_gen.nonzero(as_tuple=False).squeeze(-1)
            g = gens[torch.randint(0, len(gens), (idx.numel(),), device=device)]
            active[idx] = mul[active[idx], g]
            x[idx, t] = g

        # --- DISTRACTOR: no state change ---
        if is_dist.any():
            idx = is_dist.nonzero(as_tuple=False).squeeze(-1)
            x[idx, t] = NG + 2 + torch.randint(0, ND, (idx.numel(),), device=device)

        tgt[:, t] = active

    return x, tgt


class DampedDreamScanLM(DreamScanLM):
    """DreamScan with a damped contraction per sweep (the Rung-B divergence fix).

    z <- z + sigmoid(alpha) (block(z) - z),  sigma in (0,1) => contraction.
    Same idiom as settling_lm.py / equilibrium.py SettlingCell. Inherits the
    weight-tied NonAbelianLayer block and the sweeps= eval override from
    DreamScanLM; only the per-sweep update is damped.
    """

    def __init__(self, vocab, d=128, H=8, d_h=16, n_h=4, S_train=4,
                 beta_init="identity", damp_init=-2.0):
        super().__init__(vocab, d=d, H=H, d_h=d_h, n_h=n_h, S_train=S_train,
                         beta_init=beta_init)
        self.damp = nn.Parameter(torch.tensor(float(damp_init)))

    def forward(self, ids, targets=None, sweeps=None):
        z = self.emb(ids)
        a = torch.sigmoid(self.damp)
        for _ in range(sweeps or self.S_train):
            z = z + a * (self.block(z) - z)              # damped contraction
        logits = self.head(self.norm_f(z))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1)) if targets is not None else None
        return logits, loss


def train_ni(model, L, dev, *, steps, sweeps=None, gen_prob=0.5, ND=30,
             D_max=4, invert=True, batch=64, lr=1e-3, warmup=400):
    """Rung A/B protocol on the nested-inversion task (same param-group hygiene)."""
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
        x, tgt = make_nested_batch(batch, L, gen_prob, ND, D_max, invert, dev)
        _, loss = model(x, tgt, sweeps=sweeps) if sweeps is not None else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_ni(model, Le, dev, *, sweeps=None, gen_prob=0.5, ND=30, D_max=4,
           invert=True, n=1024, chunk=256):
    model.eval()
    hits = total = 0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        x, tgt = make_nested_batch(m, Le, gen_prob, ND, D_max, invert, dev)
        logits = model(x, sweeps=sweeps)[0] if sweeps is not None else model(x)[0]
        hits += (logits.argmax(-1)[:, -1] == tgt[:, -1]).sum().item()
        total += m
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--D-maxs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sweep-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--S-train", type=int, default=4)
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--damp-init", type=float, default=-2.0)
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--invert", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    NG = 120
    vocab = NG + 2 + args.ND                          # generators + OPEN + CLOSE + distractors
    steps = 300 if args.quick else args.steps
    seeds = 1 if args.quick else args.seeds
    eval_n = 256 if args.quick else 1024
    d_h = args.d // 8

    print(f"RUNG B' | nested inversion | train_L={args.train_L} steps={steps} "
          f"seeds={seeds} S_train={args.S_train} beta_init={args.beta_init} "
          f"damp_init={args.damp_init} invert={args.invert} dev={dev} "
          f"| chance {1/NG:.4f}", flush=True)

    arms = {
        f"dreamscan S={args.S_train}": lambda: DampedDreamScanLM(
            vocab, d=args.d, H=8, d_h=d_h, n_h=args.n_h, S_train=args.S_train,
            beta_init=args.beta_init, damp_init=args.damp_init),
        "scan 1-sweep 2-layer": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=2, H=8, d_h=d_h, n_h=args.n_h, beta_init=args.beta_init),
        "scan untied 8-layer": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=8, H=8, d_h=d_h, n_h=args.n_h, beta_init=args.beta_init),
        "GRU": lambda: GRULM(vocab, args.d, 4),
    }

    # depth-dose: accuracy vs D_max per arm (the primary table)
    depth_res = {name: {D: [] for D in args.D_maxs} for name in arms}
    # DreamScan eval-sweep grid at each depth (the mechanism signature)
    sweep_grid = {D: {S: [] for S in args.sweep_grid} for D in args.D_maxs}
    # classic length-generalization table (secondary)
    len_res = {name: {Le: [] for Le in args.eval_Ls} for name in arms}

    for s in range(seeds):
        for name, ctor in arms.items():
            torch.manual_seed(s)
            m = ctor()
            dream = isinstance(m, DreamScanLM)
            t0 = time.time()
            # train at the deepest D_max so the model sees the hardest case (a
            # fair depth-dose: all arms trained identically, evaluated across D)
            train_ni(m, args.train_L, dev, steps=steps,
                     sweeps=args.S_train if dream else None, ND=args.ND,
                     D_max=max(args.D_maxs), invert=args.invert)
            dt = time.time() - t0
            for D in args.D_maxs:
                depth_res[name][D].append(acc_ni(
                    m, args.train_L, dev, sweeps=args.S_train if dream else None,
                    ND=args.ND, D_max=D, invert=args.invert, n=eval_n))
                if dream:
                    for Sw in args.sweep_grid:
                        sweep_grid[D][Sw].append(acc_ni(
                            m, args.train_L, dev, sweeps=Sw, ND=args.ND,
                            D_max=D, invert=args.invert, n=eval_n))
            for Le in args.eval_Ls:
                len_res[name][Le].append(acc_ni(
                    m, Le, dev, sweeps=args.S_train if dream else None, ND=args.ND,
                    D_max=max(args.D_maxs), invert=args.invert, n=eval_n))
            print(f"  seed {s} {name}: params={m.num_params():,} train={dt:.0f}s "
                  f"acc@D{max(args.D_maxs)}={depth_res[name][max(args.D_maxs)][-1]:.3f}",
                  flush=True)

    def fmt(v):
        return f"{np.mean(v):>8.3f} ± {np.std(v):<6.3f}" if len(v) > 1 else \
               (f"{v[0]:>8.3f}" if v else "-")

    print(f"\nDepth-dose (acc vs D_max, train_L={args.train_L}, invert={args.invert}):",
          flush=True)
    print(f"{'D_max':>6} | " + " | ".join(f"{n:>22}" for n in depth_res), flush=True)
    for D in args.D_maxs:
        cells = [fmt(depth_res[n][D]) for n in depth_res]
        print(f"{D:>6} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)

    print(f"\nDreamScan sweep x depth grid (mean over seeds; trained at S={args.S_train}):",
          flush=True)
    print(f"{'D_max':>6} | " + " | ".join(f"S={S:>4}" for S in args.sweep_grid), flush=True)
    for D in args.D_maxs:
        print(f"{D:>6} | " + " | ".join(f"{np.mean(sweep_grid[D][S]):>6.3f}" for S in args.sweep_grid),
              flush=True)

    print(f"\nLength-generalization (secondary; D_max={max(args.D_maxs)}, invert={args.invert}):",
          flush=True)
    print(f"{'L_eval':>7} | " + " | ".join(f"{n:>22}" for n in len_res), flush=True)
    for Le in args.eval_Ls:
        cells = [fmt(len_res[n][Le]) for n in len_res]
        print(f"{Le:>7} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "train_L": args.train_L, "steps": steps, "seeds": seeds,
                       "S_train": args.S_train, "n_h": args.n_h, "beta_init": args.beta_init,
                       "damp_init": args.damp_init, "invert": args.invert, "ND": args.ND,
                       "D_maxs": args.D_maxs,
                       "depth": {n: {str(D): v for D, v in r.items()} for n, r in depth_res.items()},
                       "sweep_grid": {str(D): {str(S): v for S, v in g.items()}
                                      for D, g in sweep_grid.items()},
                       "length": {n: {str(Le): v for Le, v in r.items()} for n, r in len_res.items()}},
                      f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("RUNG B' DONE", flush=True)


if __name__ == "__main__":
    main()
