"""Step 2: learned halting (PonderNet-style) on the settling recurrence.

Replaces the hand-set `iters = length` with a halt head that decides depth per
input. Two questions (pre-registered):
  (a) recovery  — does learned-depth match the fixed-iters length-gen accuracy?
  (b) adaptivity — does mean halting-depth RISE with eval length, on its own?
(b) is the honest version of "energy-proportional compute" (with a control: it
must scale with task size, not emit a constant).

Memory-safe: training streams the expected-CE accumulation (stores only (B,) per
step); inference commits each sequence's prediction at its own halt step.
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .equilibrium import SettlingCell, make_noisy_batch


class HaltingSettlingLM(nn.Module):
    def __init__(self, vocab, d=128, hidden=256, max_iters=48, ponder_eps=0.05,
                 kl_weight=0.01, prior_halt=0.1, ponder_weight=0.0, linearize=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cell = SettlingCell(d, hidden, linearize=linearize)
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.halt = nn.Linear(d, 1)                 # per-sequence halt logit from pooled state
        self.max_iters = max_iters
        self.ponder_eps = ponder_eps
        self.kl_weight = kl_weight
        self.prior_halt = prior_halt
        self.ponder_weight = ponder_weight          # explicit cost on expected depth (forces halting)

    def forward(self, ids, targets=None, max_iters=None):
        N = max_iters or self.max_iters
        inj = self.emb(ids)
        z = torch.zeros_like(inj)
        B, L, _ = inj.shape
        dev = ids.device

        if targets is not None:
            tgt = targets.reshape(-1)
            reach = torch.ones(B, device=dev)        # prob of reaching step n unhalted
            task_loss = torch.zeros(B, device=dev)
            exp_steps = torch.zeros(B, device=dev)
            halt_probs = []
            for n in range(N):
                z = self.cell(z, inj)
                lam = torch.sigmoid(self.halt(z.mean(dim=1))).squeeze(-1)        # (B,)
                p_n = reach if n == N - 1 else lam * reach                       # remainder at last step
                logits = self.head(self.norm_f(z))                              # (B,L,V)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt,
                                     reduction="none").view(B, L).mean(dim=1)    # (B,)
                task_loss = task_loss + p_n * ce
                exp_steps = exp_steps + p_n * (n + 1)
                halt_probs.append(p_n)
                reach = reach * (1 - lam)
            p = torch.stack(halt_probs, dim=1)                                   # (B,N)
            steps_idx = torch.arange(1, N + 1, device=dev, dtype=torch.float32)
            geo = self.prior_halt * (1 - self.prior_halt) ** (steps_idx - 1)
            geo = geo / geo.sum()
            kl = (p * (torch.log(p + 1e-8) - torch.log(geo[None] + 1e-8))).sum(dim=1)
            ponder = self.ponder_weight * (exp_steps / N)      # normalized depth cost in [0,1]
            loss = (task_loss + self.kl_weight * kl + ponder).mean()
            return loss, exp_steps.mean()

        # inference: commit each sequence at its own halt step (streaming)
        reach = torch.ones(B, device=dev)
        cum_halt = torch.zeros(B, device=dev)
        committed = torch.zeros(B, dtype=torch.bool, device=dev)
        chosen = torch.zeros(B, L, self.head.out_features, device=dev)
        steps_used = torch.full((B,), N, dtype=torch.long, device=dev)
        for n in range(N):
            z = self.cell(z, inj)
            lam = torch.sigmoid(self.halt(z.mean(dim=1))).squeeze(-1)
            logits = self.head(self.norm_f(z))
            cum_halt = cum_halt + lam * reach
            newly = (~committed) & ((cum_halt >= 1 - self.ponder_eps) | (n == N - 1))
            if newly.any():
                chosen[newly] = logits[newly]
                steps_used[newly] = n + 1
                committed = committed | newly
            reach = reach * (1 - lam)
        return chosen, steps_used

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train(model, L, dev, *, steps=6000, lr=1e-3, batch=64, warmup=400, gen_prob=0.5, ND=30):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, tgt = make_noisy_batch(batch, L, gen_prob, ND, dev)
        loss, _ = model(x, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def evaluate(model, Le, dev, *, eval_iters, gen_prob=0.5, ND=30, n=1024):
    model.eval()
    x, tgt = make_noisy_batch(n, Le, gen_prob, ND, dev)
    logits, steps = model(x, max_iters=eval_iters)
    acc = (logits.argmax(-1)[:, -1] == tgt[:, -1]).float().mean().item()
    return acc, steps.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-L", type=int, default=32)
    ap.add_argument("--eval-Ls", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--train-iters", type=int, default=48)
    ap.add_argument("--eval-iters", type=int, default=192)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--ND", type=int, default=30)
    ap.add_argument("--ponder-cost", dest="ponder_cost", type=float, default=0.0,
                    help="explicit cost on expected depth (0 = none; collapses to never-halt)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 120 + args.ND
    print(f"HALTING-SETTLE | train_L={args.train_L} train_iters={args.train_iters} "
          f"eval_iters={args.eval_iters} steps={args.steps} d={args.d} ponder_cost={args.ponder_cost} dev={dev}", flush=True)
    torch.manual_seed(0)
    m = HaltingSettlingLM(vocab, d=args.d, max_iters=args.train_iters, ponder_weight=args.ponder_cost)
    print(f"params={m.num_params():,}", flush=True)
    train(m, args.train_L, dev, steps=args.steps, ND=args.ND)
    print(f"{'L_eval':>7} | {'halting acc':>11} | {'mean depth':>10}", flush=True)
    for Le in args.eval_Ls:
        acc, depth = evaluate(m, Le, dev, eval_iters=args.eval_iters, ND=args.ND)
        print(f"{Le:>7} | {acc:>11.3f} | {depth:>10.1f}", flush=True)
    print("reference (fixed iters=L, 6000 steps): L32 0.996 L48 0.943 L64 0.866 L96 0.639 L128 0.413", flush=True)
    print("HALTING-SETTLE DONE", flush=True)


if __name__ == "__main__":
    main()
