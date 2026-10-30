"""Rung A — scannable non-abelian recurrence vs settling on noisy-S5.

The prize restated: escaping TC0 does not require linear serial depth — the S5
word problem is NC1-complete (Barrington), and NC1 is log-depth PARALLEL. Group
composition is associative, so a linear-in-state RNN whose input-dependent
transitions are structured NON-COMMUTING operators can compute the running
composition with a parallel prefix scan: O(log L) depth, O(L) work. Settling
(equilibrium.py) pays L sequential steps for the same capability — it is the
existence proof; the scan is the efficient construction. The load-bearing
property is non-abelian composition inside the scan, NOT pointwise nonlinearity
and NOT serial depth. Diagonal transitions (Mamba-class) commute, which is
exactly why they stay TC0 and cliff-collapse on S5.

Transition family: products of n_h generalized Householder factors
    A(x_t) = prod_i (I - beta_i k_i k_i^T),  beta in (0,2), ||k_i|| = 1
(the DeltaNet/DeltaProduct algebra). Eigenvalues in (-1,1] => non-expansive
(stable at any eval length); beta -> 2 gives reflections, whose products
generate the orthogonal group — S5's permutation representation lives there.
n_h is the dose knob: a transposition needs 1 reflection, the 5-cycle needs 4,
so capability should RISE with n_h (mechanism signature, pre-registered).

Controls:
  --transition diag   same architecture, same eigenvalue range lambda=tanh(.)
                      in (-1,1), but DIAGONAL (commutative). If this also
                      length-generalizes, the task is not testing
                      non-abelian-ness and the result is void.
  scan == sequential  numerical equivalence (tests/test_nonabelian_scan.py);
                      training runs the log-depth scan path, so "trained at
                      parallel cost" is literal, not hypothetical.

Pre-registered verdict: results/rungA_nonabelian_scan.md (written before the
run). Protocol matches s5_seeds.py exactly: noisy-S5 gen_prob=0.5 ND=30,
train L=32 batch 64, 6000 steps, AdamW(1e-3, betas=(0.9,0.98), wd=0.01),
warmup 400 + cosine, clip 1.0, eval n=1024 last-token acc, seeds 0..2.

Run:
  python -m phase_ssm.nonabelian_scan --seeds 3 --with-gru        # primary
  python -m phase_ssm.nonabelian_scan --dose                      # n_h in {1,2,4}
  python -m phase_ssm.nonabelian_scan --transition diag           # void check
  python -m phase_ssm.nonabelian_scan --with-settling             # same-device anchor
"""
from __future__ import annotations
import argparse, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .equilibrium import make_noisy_batch


def householder_products(keys, betas):
    """A = M_{n_h} @ ... @ M_1 (factor 1 applied first), M_i = I - beta_i k_i k_i^T.

    keys: (..., n_h, d) unit-norm; betas: (..., n_h) in [0, 2].
    Built by rank-1 updates: M @ A = A - beta k (k^T A) — no d^3 per factor.
    """
    n_h, d = keys.shape[-2], keys.shape[-1]
    k0 = keys[..., 0, :]
    eye = torch.eye(d, device=keys.device, dtype=keys.dtype)
    A = eye - betas[..., 0, None, None] * k0.unsqueeze(-1) * k0.unsqueeze(-2)
    for i in range(1, n_h):
        k = keys[..., i, :]
        kA = k.unsqueeze(-2) @ A                                  # (..., 1, d)
        A = A - betas[..., i, None, None] * k.unsqueeze(-1) * kA
    return A


def affine_scan(A, b, h0):
    """All prefix states of h_t = A_t h_{t-1} + b_t via Hillis-Steele scan.

    Affine maps compose associatively: (A2,b2)o(A1,b1) = (A2@A1, A2@b1 + b2),
    so the prefix is computed in ceil(log2 L) rounds of batched matmuls — the
    parallel-training-cost claim, executed literally.

    A: (B,L,H,d,d) full transitions  OR  (B,L,H,d) diagonal (control).
    b: (B,L,H,d) injections. h0: (H,d) learned initial state.
    Returns h: (B,L,H,d).
    """
    diag = A.dim() == 4
    B, L, H, d = b.shape
    P, c = A, b
    s = 1
    while s < L:
        if diag:
            Pid = torch.ones(B, s, H, d, device=A.device, dtype=A.dtype)
        else:
            Pid = torch.eye(d, device=A.device, dtype=A.dtype).expand(B, s, H, d, d)
        Pprev = torch.cat([Pid, P[:, :-s]], dim=1)
        cprev = torch.cat([torch.zeros(B, s, H, d, device=A.device, dtype=A.dtype), c[:, :-s]], dim=1)
        if diag:
            c = P * cprev + c
            P = P * Pprev
        else:
            c = (P @ cprev.unsqueeze(-1)).squeeze(-1) + c
            P = P @ Pprev
        s *= 2
    if diag:
        return P * h0 + c
    return torch.einsum("blhij,hj->blhi", P, h0) + c


def affine_seq(A, b, h0):
    """Sequential reference for the scan (tests + sanity)."""
    diag = A.dim() == 4
    B, L, H, d = b.shape
    h = h0.expand(B, H, d)
    outs = []
    for t in range(L):
        if diag:
            h = A[:, t] * h + b[:, t]
        else:
            h = (A[:, t] @ h.unsqueeze(-1)).squeeze(-1) + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


class NonAbelianLayer(nn.Module):
    def __init__(self, d, H=8, d_h=16, n_h=4, transition="householder", scan=True,
                 beta_init="identity"):
        super().__init__()
        assert H * d_h == d
        self.H, self.d_h, self.n_h, self.transition, self.scan = H, d_h, n_h, transition, scan
        self.record_betas = False
        self._betas = None
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        if transition == "householder":
            self.kproj = nn.Linear(d, H * n_h * d_h)
            self.bproj = nn.Linear(d, H * n_h)
            # O(d_h) has two connected components; a rank-1 factor switches
            # parity only through the state-erasing beta=1 valley. beta_init
            # controls which components are populated at init:
            #   identity: all factors near I (raw -2, beta~0.24) — even side
            #   reflect:  all near reflections (raw +4, beta~1.96) — odd side
            #   split:    half odd side, half even side — both parities cheap
            with torch.no_grad():
                if beta_init == "reflect":
                    nn.init.constant_(self.bproj.bias, 4.0)
                elif beta_init == "split":
                    pat = torch.full((n_h,), -4.0)
                    pat[: (n_h + 1) // 2] = 4.0
                    self.bproj.bias.copy_(pat.repeat(H))
                else:
                    nn.init.constant_(self.bproj.bias, -2.0)
        else:                                          # diag control (commutative)
            self.dproj = nn.Linear(d, H * d_h)
            nn.init.constant_(self.dproj.bias, 2.0)    # lambda≈0.96: persistent start
        self.vproj = nn.Linear(d, d)                   # injection
        self.oproj = nn.Linear(d, d)
        self.h0 = nn.Parameter(torch.randn(H, d_h) / math.sqrt(d_h))
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B, L, d = x.shape
        xn = self.norm1(x)
        if self.transition == "householder":
            keys = F.normalize(self.kproj(xn).view(B, L, self.H, self.n_h, self.d_h), dim=-1)
            betas = 2.0 * torch.sigmoid(self.bproj(xn).view(B, L, self.H, self.n_h))
            if self.record_betas:
                self._betas = betas.detach()
            A = householder_products(keys, betas)
        else:
            A = torch.tanh(self.dproj(xn).view(B, L, self.H, self.d_h))
        binj = self.vproj(xn).view(B, L, self.H, self.d_h)
        h = (affine_scan if self.scan else affine_seq)(A, binj, self.h0)
        x = x + self.oproj(h.reshape(B, L, d))
        x = x + self.ffn(self.norm2(x))
        return x


class NonAbelianScanLM(nn.Module):
    """Linear-in-state RNN, input-dependent Householder-product transitions,
    trained through the log-depth parallel scan. Per-position MLPs only —
    ALL cross-position information flows through the non-abelian scan."""

    def __init__(self, vocab, d=128, layers=2, H=8, d_h=16, n_h=4,
                 transition="householder", scan=True, beta_init="identity"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [NonAbelianLayer(d, H, d_h, n_h, transition, scan, beta_init)
             for _ in range(layers)])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, targets=None):
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm_f(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_model(model, L, dev, *, steps, settling=False, gen_prob=0.5, ND=30,
                batch=64, lr=1e-3, warmup=400):
    """s5_seeds.py protocol verbatim, except: no weight decay on biases,
    LayerNorm params, and h0 (standard hygiene; wd on a raw-beta bias would
    drag transitions toward state-erasure beta=1)."""
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
        x, tgt = make_noisy_batch(batch, L, gen_prob, ND, dev)
        _, loss = model(x, tgt, iters=L) if settling else model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def acc_at(model, Le, dev, *, settling=False, gen_prob=0.5, ND=30, n=1024, chunk=256):
    model.eval()
    hits = total = 0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        x, tgt = make_noisy_batch(m, Le, gen_prob, ND, dev)
        logits = model(x, iters=Le)[0] if settling else model(x)[0]
        hits += (logits.argmax(-1)[:, -1] == tgt[:, -1]).sum().item()
        total += m
    return hits / total


@torch.no_grad()
def mag_probe_diag(model, dev, *, ND=30, L=32, n=256):
    """Hypothesis-v2 probe: MAGNITUDE saturation, not sign access. det(A_t) =
    prod(1-beta_i); a faithful near-orthogonal solution needs |det| ~= 1 on
    generator tokens (any |det| < 1 decays state amplitude like |det|^#gens).
    For each layer: pick the carrier head (best det-sign parity alignment) and
    report its mean |det| on generator vs distractor positions."""
    from .statetrack import S as ODD_TOK
    model.eval()
    x, _ = make_noisy_batch(n, L, 0.5, ND, dev)
    gen = x < 120
    odd = x == ODD_TOK
    for blk in model.blocks:
        blk.record_betas = True
    model(x)
    out = []
    for li, blk in enumerate(model.blocks):
        blk.record_betas = False
        dets = (1.0 - blk._betas).prod(dim=-1)                     # (B,L,H)
        neg = dets < 0
        po = (neg & odd.unsqueeze(-1)).sum(dim=(0, 1)).float() / odd.sum().clamp(min=1)
        pe = ((~neg) & (~odd).unsqueeze(-1)).sum(dim=(0, 1)).float() / (~odd).sum().clamp(min=1)
        match = (po + pe) / 2
        h = int(match.argmax())
        absd = dets.abs()
        out.append({
            "layer": li, "carrier_head": h, "sign_best": match[h].item(),
            "gen_absdet": absd[..., h][gen].mean().item(),
            "dist_absdet": absd[..., h][~gen].mean().item(),
        })
    return out


@torch.no_grad()
def det_parity_diag(model, dev, *, ND=30, L=32, n=256):
    """Mechanism probe: det(A_t) = prod_i (1 - beta_i) per head; sign(det) is the
    transition's reflection parity. The S5 transposition token is ODD (det -1 in
    any faithful orthogonal rep); the 5-cycle, identity, and distractors are
    EVEN. Reports, per layer, how well each head's det sign matches token
    parity — best head and head mean on odd-token positions."""
    from .statetrack import S as ODD_TOK
    model.eval()
    x, _ = make_noisy_batch(n, L, 0.5, ND, dev)
    for blk in model.blocks:
        blk.record_betas = True
    model(x)
    out = []
    odd = (x == ODD_TOK)                                     # (B,L)
    for li, blk in enumerate(model.blocks):
        blk.record_betas = False
        dets = (1.0 - blk._betas).prod(dim=-1)               # (B,L,H)
        neg = dets < 0
        per_head_odd = (neg & odd.unsqueeze(-1)).sum(dim=(0, 1)).float() / odd.sum().clamp(min=1)
        per_head_even = ((~neg) & (~odd).unsqueeze(-1)).sum(dim=(0, 1)).float() / (~odd).sum().clamp(min=1)
        match = (per_head_odd + per_head_even) / 2           # balanced parity accuracy per head
        out.append({"layer": li, "best_head": match.max().item(), "mean_head": match.mean().item()})
    return out


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--transition", choices=["householder", "diag"], default="householder")
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--parity-ablation", action="store_true",
                    help="beta-init arms {identity, split, reflect}, --seeds each, + det diagnostics")
    ap.add_argument("--mag-probe", action="store_true",
                    help="hypothesis-v2: arms {identity, reflect}, carrier-head |det| vs acc@max-L")
    ap.add_argument("--dose", action="store_true", help="n_h in {1,2,4}, seed 0 only")
    ap.add_argument("--with-gru", action="store_true", help="same-device GRU anchor")
    ap.add_argument("--with-settling", action="store_true", help="same-device settling anchor (seed 0)")
    ap.add_argument("--quick", action="store_true", help="smoke: 300 steps, 1 seed, n=256")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None, help="write raw results JSON here")
    args = ap.parse_args()
    dev = pick_device(args.device)
    vocab = 120 + args.ND
    steps = 300 if args.quick else args.steps
    seeds = 1 if args.quick else args.seeds
    eval_n = 256 if args.quick else 1024
    d_h = args.d // args.heads

    def build(n_h, transition, beta_init=None):
        return NonAbelianScanLM(vocab, d=args.d, layers=args.layers, H=args.heads,
                                d_h=d_h, n_h=n_h, transition=transition,
                                beta_init=beta_init or args.beta_init)

    print(f"RUNG A | nonabelian-scan vs anchors | train_L={args.train_L} steps={steps} "
          f"seeds={seeds} d={args.d} H={args.heads} d_h={d_h} n_h={args.n_h} "
          f"layers={args.layers} transition={args.transition} dev={dev}", flush=True)

    arms = {}
    if args.mag_probe:
        arms = {f"init={bi}": (lambda bi=bi: build(args.n_h, "householder", bi), False)
                for bi in ("identity", "reflect")}
    elif args.parity_ablation:
        arms = {f"init={bi}": (lambda bi=bi: build(args.n_h, "householder", bi), False)
                for bi in ("identity", "split", "reflect")}
    elif args.dose:
        arms = {f"nonabelian n_h={nh}": (lambda nh=nh: build(nh, "householder"), False)
                for nh in (1, 2, 4)}
        seeds = 1
    else:
        arms[f"nonabelian n_h={args.n_h}"] = (lambda: build(args.n_h, args.transition), False)
    if args.with_gru:
        from .statetrack import GRULM
        arms["GRU"] = (lambda: GRULM(vocab, args.d, 4), False)
    if args.with_settling:
        from .equilibrium import SettlingLM
        arms["settling"] = (lambda: SettlingLM(vocab, d=args.d, iters=args.train_L), True)

    res = {name: {Le: [] for Le in args.eval_Ls} for name in arms}
    probes = []
    import time
    for s in range(seeds):
        for name, (ctor, settling) in arms.items():
            if name == "settling" and s > 0:
                continue                                   # anchor: seed 0 only
            torch.manual_seed(s)
            m = ctor()
            n_par = sum(p.numel() for p in m.parameters())
            t0 = time.time()
            train_model(m, args.train_L, dev, steps=steps, settling=settling, ND=args.ND)
            dt = time.time() - t0
            for Le in args.eval_Ls:
                res[name][Le].append(acc_at(m, Le, dev, settling=settling, ND=args.ND, n=eval_n))
            print(f"  seed {s} {name}: params={n_par:,} train={dt:.0f}s "
                  f"acc@{args.train_L}={res[name][args.train_L][-1]:.3f} "
                  f"acc@{2*args.train_L}={res[name][2*args.train_L][-1]:.3f}"
                  if 2 * args.train_L in res[name] else "", flush=True)
            if isinstance(m, NonAbelianScanLM) and m.blocks[0].transition == "householder":
                acc128 = res[name][max(args.eval_Ls)][-1]
                if args.mag_probe:
                    diag = mag_probe_diag(m, dev, ND=args.ND, L=args.train_L)
                    probes.append({"arm": name, "seed": s, "acc_maxL": acc128, "diag": diag})
                    print("    mag-probe: " + " | ".join(
                        f"L{d['layer']} h{d['carrier_head']} sign {d['sign_best']:.3f} "
                        f"|det|gen {d['gen_absdet']:.4f} dist {d['dist_absdet']:.4f}"
                        for d in diag) + f"  (acc@{max(args.eval_Ls)}={acc128:.3f})", flush=True)
                else:
                    diag = det_parity_diag(m, dev, ND=args.ND, L=args.train_L)
                    print("    det-parity: " + " | ".join(
                        f"L{d['layer']} best {d['best_head']:.3f} mean {d['mean_head']:.3f}"
                        for d in diag) + f"  (acc@{max(args.eval_Ls)}={acc128:.3f})", flush=True)

    import numpy as np
    header = f"{'L_eval':>7} | " + " | ".join(f"{n:>22}" for n in res)
    print("\n" + header, flush=True)
    for Le in args.eval_Ls:
        cells = []
        for name in res:
            v = res[name][Le]
            cells.append(f"{np.mean(v):>8.3f} ± {np.std(v):<6.3f}" if len(v) > 1
                         else (f"{v[0]:>8.3f}" if v else "-"))
        print(f"{Le:>7} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
    print("logged 3-seed reference (results/serial_depth_lengthgen.md): "
          "settling@64 0.891±0.018 @128 0.486±0.032 | GRU@64 0.549±0.055 @128 0.076±0.022",
          flush=True)

    if args.mag_probe and probes:
        import numpy as np
        maxL = max(args.eval_Ls)
        rows = sorted(probes, key=lambda p: p["acc_maxL"])
        print(f"\nmag-probe summary (per run, deepest-layer carrier head; E[#gens]@L={maxL} "
              f"= {maxL // 2}):", flush=True)
        print(f"{'arm':>14} {'seed':>4} {'acc@'+str(maxL):>9} {'|det|gen':>9} "
              f"{'|det|^'+str(maxL//2):>12}", flush=True)
        for p in rows:
            d = p["diag"][-1]
            print(f"{p['arm']:>14} {p['seed']:>4} {p['acc_maxL']:>9.3f} "
                  f"{d['gen_absdet']:>9.4f} {d['gen_absdet'] ** (maxL // 2):>12.4f}", flush=True)
        accs = [p["acc_maxL"] for p in rows]
        mags = [p["diag"][-1]["gen_absdet"] for p in rows]
        ra = np.argsort(np.argsort(accs)).astype(float)
        rm = np.argsort(np.argsort(mags)).astype(float)
        rho = float(np.corrcoef(ra, rm)[0, 1]) if len(rows) > 2 else float("nan")
        print(f"Spearman(acc, carrier |det|gen) over {len(rows)} runs: {rho:.3f}", flush=True)

    if args.json:
        payload = {"device": dev, "train_L": args.train_L, "steps": steps, "seeds": seeds,
                   "d": args.d, "heads": args.heads, "d_h": d_h, "n_h": args.n_h,
                   "layers": args.layers, "transition": args.transition, "ND": args.ND,
                   "beta_init": args.beta_init, "parity_ablation": args.parity_ablation,
                   "mag_probe": probes if args.mag_probe else None,
                   "results": {n: {str(Le): v for Le, v in r.items()} for n, r in res.items()}}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
