"""Language test, step 1 — is the settling formulation a viable LANGUAGE model?

A weight-tied iterated PhaseBlock (the oscillator-SSM mixer settled to depth,
Universal-Transformer style). Trained as a byte-level LM on text8, compared on
bits-per-char to the transformer (1.350) and plain 6-layer PhaseSSM (1.377). This
is the viability gate: does settling-as-depth do language at all? The deeper test
(does the state-tracking CAPABILITY help on a language-format task transformers
fail) is the follow-up.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PhaseSSMConfig, PhaseBlock, RMSNorm
from .data import ByteData, get_text8, stats


class SettlingByteLM(nn.Module):
    def __init__(self, cfg: PhaseSSMConfig, iters: int = 6):
        super().__init__()
        self.cfg = cfg
        self.iters = iters
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.block = PhaseBlock(cfg)                 # ONE block, weight-tied across iterations
        self.damp = nn.Parameter(torch.tensor(-2.0)) # sigmoid(-2)~0.12 damped settling step
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)                       # std=0.02 init (the bug: default emb std=1 blew up logits)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight       # re-tie AFTER init

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, targets=None):
        x = self.emb(ids)
        a = torch.sigmoid(self.damp)
        for _ in range(self.iters):
            x = x + a * (self.block(x) - x)          # damped contraction toward a fixed point
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p)); total += p.numel()
        return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/settling_lm")
    ap.add_argument("--d-model", type=int, default=640)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--state-dim", type=int, default=96)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PhaseSSMConfig(vocab_size=256, d_model=args.d_model, n_layers=1, state_dim=args.state_dim,
                         expand=1, d_ff_mult=3, use_phase_memory=False)
    model = SettlingByteLM(cfg, iters=args.iters).to(dev)
    print(f"[settling-LM] iters={args.iters} d_model={args.d_model} params={model.num_params():,} "
          f"(weight-tied; baselines were 11.6M)", flush=True)
    data = ByteData(get_text8("data"))
    print(f"[data] {stats(data)}", flush=True)
    from .train_core import train_model
    train_model(model, data, name="settling-lm", out=Path(args.out), steps=args.steps,
                batch=args.batch, seq=args.seq, lr=args.lr, min_lr=1e-4, warmup=400,
                weight_decay=0.1, grad_clip=1.0, grad_accum=1, eval_interval=1000, eval_iters=40,
                train_log_interval=0, skip_initial_eval=False, device=dev, amp_dtype=torch.bfloat16,
                compile=False, model_cfg=cfg.__dict__, seed=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
