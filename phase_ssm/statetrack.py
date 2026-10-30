"""S_5 permutation state-tracking — the TC0-hard capability-separation probe.

The canonical task from "The Illusion of State in State-Space Models" (Merrill,
Petty, Sabharwal 2024): given a sequence of group elements, output the running
composition. S_5 is non-solvable, so its word problem is NC1-complete — OUTSIDE
TC0. Fixed-depth transformers AND diagonal SSMs are in TC0, so they provably
cannot track this for growing depth; a genuine non-diagonal sequential recurrence
(GRU) can. This is the discriminator: if transformer + diagonal-SSM degrade with
length while the GRU holds, the task can DETECT a capability separation — the
prerequisite for testing whether a nonlinear *settling* model escapes TC0 too.

Run: python -m phase_ssm.statetrack
"""
from __future__ import annotations
import itertools, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerConfig, TransformerLM
from .model import PhaseSSMConfig, PhaseSSMLM

# --- S_5 group tables ---
PERMS = list(itertools.permutations(range(5)))          # 120 elements as tuples
IDX = {p: i for i, p in enumerate(PERMS)}
def _compose(a, b):                                     # apply a THEN b: (a;b)[i] = b[a[i]]
    return tuple(b[a[i]] for i in range(5))
MUL = torch.tensor([[IDX[_compose(PERMS[i], PERMS[j])] for j in range(120)] for i in range(120)])
IDENTITY = IDX[tuple(range(5))]
S = IDX[(1, 0, 2, 3, 4)]                                # transposition (0 1)
C = IDX[(1, 2, 3, 4, 0)]                                # 5-cycle
GENS = [IDENTITY, S, C]                                 # {id, (01), 5-cycle} generate S_5
VOCAB = 120


def make_batch(B, L, device):
    g = torch.tensor(GENS)
    ids = g[torch.randint(0, len(GENS), (B, L))]        # (B,L) input generator ids
    mul = MUL.to(ids.device)
    targets = torch.empty(B, L, dtype=torch.long)
    state = torch.full((B,), IDENTITY, dtype=torch.long)
    for t in range(L):
        state = mul[state, ids[:, t]]                   # running composition
        targets[:, t] = state
    return ids.to(device), targets.to(device)


class GRULM(nn.Module):
    """Non-diagonal nonlinear recurrence — the reference true state-tracker (escapes TC0)."""
    def __init__(self, vocab, d, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.gru = nn.GRU(d, d, layers, batch_first=True)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids, targets=None):
        x, _ = self.gru(self.emb(ids))
        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_eval(model, L, device, steps=6000, lr=1e-3, batch=64, warmup=400):
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
    model.eval()
    with torch.no_grad():
        ids, tgt = make_batch(1024, L, device)
        pred = model(ids)[0].argmax(-1)
        last = (pred[:, -1] == tgt[:, -1]).float().mean().item()   # full running state
        allp = (pred == tgt).float().mean().item()
    return last, allp


def build(kind, L):
    if kind == "transformer":
        return TransformerLM(TransformerConfig(vocab_size=VOCAB, d_model=128, n_layers=4, n_heads=4,
                                               d_ff_mult=2, block_size=max(L, 8)))
    if kind == "diag-ssm":
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=VOCAB, d_model=128, n_layers=4, state_dim=64,
                                         expand=1, d_ff_mult=2, use_phase_memory=False))
    return GRULM(VOCAB, 128, 4)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lengths = [8, 16, 32, 64]
    kinds = ["transformer", "diag-ssm", "GRU"]
    print(f"S_5 state-tracking | device={dev} | last-token = full running-composition accuracy (chance {1/120:.3f})", flush=True)
    print(f"{'L':>4} | " + " | ".join(f"{k:>22}" for k in kinds), flush=True)
    for L in lengths:
        cells = []
        for kind in kinds:
            torch.manual_seed(0)
            m = build(kind, L)
            last, allp = train_eval(m, L, dev)
            cells.append(f"last {last:.2f} all {allp:.2f}")
        print(f"{L:>4} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
