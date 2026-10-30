"""BDH as a state-tracking arm: a falsifiable TC0 prediction against a published architecture.

BDH ("The Dragon Hatchling", Pathway, arXiv:2509.26507) is a brain-inspired,
synaptic, explicitly post-transformer model with GPT-2-parity LM quality at
10M-1B. Its token mixer is causal linear attention with NO decay/erase:

    scores = (QR @ KR.mT).tril(diagonal=-1);  out = scores @ V
    recurrent form:  S_t = S_{t-1} + Phi(k_t) (x) v_t        # additive Hebbian

That is a COMMUTATIVE composition (linear-attention class). The Merrill-Sabharwal
/Illusion-of-State result (already cited in statetrack.py) puts commutative
recurrences in TC0, and the S5 word problem is NC1-complete -> provably outside
TC0. So BDH structurally cannot track non-solvable-group state at growing length,
for the same reason this repo's own diag-SSM arms cliff-collapsed on noisy-S5.

Honest framing (load-bearing): the prediction is NOT "our architecture beats
theirs." This repo's own phase-binding recall memory (linear_memory.py) is
RetNet/GLA structure -- ALSO linear-attention class, ALSO TC0 for
state-tracking. The one object in this program that escapes the cell is Rung A's
NON-ABELIAN scan (Householder-product transitions -> non-commutative composition
-> inside NC1). So the pre-registered claim is symmetric and uncomfortable:

    BDH (published, brain-inspired) and this repo's recall memory are in the SAME
    complexity cell; both predict-collapse on noisy-S5. The non-abelian scan is
    the one validated object outside it. This run measures BDH's collapse curve
    beside both, and is the falsifiable prediction about a published model.

Why this matters: BDH frames itself around "modus ponens reasoning" but tests no
state-tracking / length-generalization / formal-language benchmark. This is the
dial that could break the reasoning story, run against the primitive that
survives it. If BDH collapses -> a sharp published-target result. If it does NOT
-> linear attention with ReLU feature gating is doing something surprising and
that is more valuable than confirmation.

Run:
  python -m phase_ssm.bdh_state_track --quick              # smoke / assay
  python -m phase_ssm.bdh_state_track --seeds 3 --json results/bdh_state_track.json
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonabelian_scan import (NonAbelianLayer, NonAbelianScanLM, pick_device,
                              train_model, acc_at)
from .statetrack import GRULM, VOCAB
from .vendor.bdh import BDH, BDHConfig

# A thin wrapper so BDH matches the (logits, loss) contract with num_params() and
# no `iters=` kwarg -- the same contract as GRULM / NonAbelianScanLM. The vendored
# BDH file is NOT modified.
class BDHArm(nn.Module):
    def __init__(self, vocab, d=128, n_layer=6, mult=8, n_head=4):
        super().__init__()
        cfg = BDHConfig(n_layer=n_layer, n_embd=d, mlp_internal_dim_multiplier=mult,
                        vocab_size=vocab, n_head=n_head, dropout=0.0)
        self.bdh = BDH(cfg)

    def forward(self, ids, targets=None):
        return self.bdh(ids, targets)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Damped iterated scan at matched depth to BDH (both are iterated weight-tied
# blocks; this is the apples-to-apples protagonist comparison). Imported lazily to
# keep the BDH arm standalone-testable.
def _dreamscan(vocab, d, n_h, sweeps, d_h, beta_init="identity", damp_init=-2.0):
    from .conditional_composition import DreamScanLM
    from .nested_inversion import DampedDreamScanLM
    return DampedDreamScanLM(vocab, d=d, H=8, d_h=d_h, n_h=n_h, S_train=sweeps,
                             beta_init=beta_init, damp_init=damp_init)


def main():
    from .equilibrium import make_noisy_batch
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--bdh-mult", type=int, default=8,
                    help="BDH mlp_internal_dim_multiplier; 8 -> ~432k params (matched to GRU)")
    ap.add_argument("--bdh-layers", type=int, default=6)
    ap.add_argument("--dreamscan-S", type=int, default=6,
                    help="DampedDreamScan sweeps, matched to BDH n_layer")
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    vocab = VOCAB + args.ND                      # 120 + 30 = 150, the noisy-S5 vocab
    steps = 300 if args.quick else args.steps
    seeds = 1 if args.quick else args.seeds
    eval_n = 256 if args.quick else 1024
    d_h = args.d // 8

    print(f"BDH STATE-TRACK | noisy-S5 train_L={args.train_L} steps={steps} seeds={seeds} "
          f"dev={dev} | chance {1/VOCAB:.4f}", flush=True)
    print(f"  arms: BDH(mult={args.bdh_mult},L={args.bdh_layers}), GRU, scan(n_h={args.n_h},2lyr), "
          f"DampedDreamScan(S={args.dreamscan_S})", flush=True)

    arms = {
        "BDH": lambda: BDHArm(vocab, d=args.d, n_layer=args.bdh_layers, mult=args.bdh_mult),
        "GRU": lambda: GRULM(vocab, args.d, 4),
        "scan n_h=4 2-layer": lambda: NonAbelianScanLM(
            vocab, d=args.d, layers=2, H=8, d_h=d_h, n_h=args.n_h, beta_init=args.beta_init),
        f"dreamscan S={args.dreamscan_S}": lambda: _dreamscan(
            vocab, args.d, args.n_h, args.dreamscan_S, d_h, beta_init=args.beta_init),
    }
    # which arms are scan-family (need the diag-SSM void check / parity diag treatment)?
    from .conditional_composition import DreamScanLM
    res = {name: {Le: [] for Le in args.eval_Ls} for name in arms}
    for s in range(seeds):
        for name, ctor in arms.items():
            torch.manual_seed(s)
            m = ctor()
            t0 = time.time()
            # all arms are non-settling (no iters= kwarg): standard train_model path
            train_model(m, args.train_L, dev, steps=steps, settling=False, ND=args.ND)
            dt = time.time() - t0
            for Le in args.eval_Ls:
                res[name][Le].append(acc_at(m, Le, dev, settling=False, ND=args.ND, n=eval_n))
            print(f"  seed {s} {name}: params={m.num_params():,} train={dt:.0f}s "
                  f"acc@{args.train_L}={res[name][args.train_L][-1]:.3f} "
                  f"acc@{2*args.train_L}={res[name][2*args.train_L][-1]:.3f}", flush=True)

    def fmt(v):
        return f"{np.mean(v):>8.3f} +/- {np.std(v):<6.3f}" if len(v) > 1 else \
               (f"{v[0]:>8.3f}" if v else "-")

    print(f"\n{'L_eval':>7} | " + " | ".join(f"{n:>22}" for n in res), flush=True)
    for Le in args.eval_Ls:
        cells = [fmt(res[n][Le]) for n in res]
        print(f"{Le:>7} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "train_L": args.train_L, "steps": steps, "seeds": seeds,
                       "d": args.d, "n_h": args.n_h, "ND": args.ND,
                       "bdh_mult": args.bdh_mult, "bdh_layers": args.bdh_layers,
                       "dreamscan_S": args.dreamscan_S, "beta_init": args.beta_init,
                       "params": {n: ctor().num_params() for n, ctor in arms.items()},
                       "results": {n: {str(Le): v for Le, v in r.items()} for n, r in res.items()}},
                      f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("BDH STATE-TRACK DONE", flush=True)


if __name__ == "__main__":
    main()
