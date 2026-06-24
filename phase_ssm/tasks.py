"""MQAR — multi-query associative recall — the SSM Achilles' heel.

A sequence presents key->value pairs, then queries keys; the model must emit the
right value. Pure SSMs compress to a fixed state and fail this at small state
size; attention solves it; the phase-binding memory is built to solve it too.
This is the task that shows the differentiator earns its place.

    seq = [k1 v1 k2 v2 ... kP vP  q1 q2 ... qQ]
    target at each query position qi = value bound to that key (else -100)
"""
from __future__ import annotations

import torch


def make_mqar(batch: int, n_pairs: int, n_queries: int, n_keys: int, n_values: int,
              device: str = "cpu", seed: int | None = None):
    """Returns (x, y, vocab_size). Keys in [1, n_keys]; values in [n_keys+1, n_keys+n_values]."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    KEY0, VAL0 = 1, 1 + n_keys
    L = 2 * n_pairs + n_queries
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    for b in range(batch):
        keys = torch.randperm(n_keys, generator=g)[:n_pairs] + KEY0
        # unique values per sequence -> no modal-value shortcut; only true binding wins
        vals = torch.randperm(n_values, generator=g)[:n_pairs] + VAL0
        seq = torch.empty(2 * n_pairs, dtype=torch.long)
        seq[0::2] = keys
        seq[1::2] = vals
        x[b, : 2 * n_pairs] = seq
        qidx = torch.randint(0, n_pairs, (n_queries,), generator=g)
        for j, qi in enumerate(qidx):
            pos = 2 * n_pairs + j
            x[b, pos] = keys[qi]
            y[b, pos] = vals[qi]           # must recall the bound value
    vocab = 1 + n_keys + n_values
    if device.startswith("cuda"):
        x, y = x.to(device), y.to(device)
    return x, y, vocab


def recall_accuracy(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        mask = y != -100
        return (logits.argmax(-1)[mask] == y[mask]).float().mean().item()


def train_on_mqar(model, *, steps, n_pairs, n_queries, n_keys, n_values, batch=64,
                  lr=3e-3, device="cpu", log_every=0, warmup=None):
    import math
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(device)
    warmup = warmup if warmup is not None else min(1000, steps // 10)
    for step in range(steps):
        # warmup then cosine decay to 10% — induction heads need stable optimization
        if step < warmup:
            cur_lr = lr * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            cur_lr = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        x, y, _ = make_mqar(batch, n_pairs, n_queries, n_keys, n_values, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if log_every and step % log_every == 0:
            xv, yv, _ = make_mqar(256, n_pairs, n_queries, n_keys, n_values, device, seed=12345)
            print(f"    step {step:4d}  loss {loss.item():.3f}  recall {recall_accuracy(model, xv, yv):.3f}")
    xv, yv, _ = make_mqar(512, n_pairs, n_queries, n_keys, n_values, device, seed=99)
    return recall_accuracy(model, xv, yv)


if __name__ == "__main__":
    import torch
    from .model import PhaseSSMConfig, PhaseSSMLM
    from .transformer import TransformerConfig, TransformerLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P, Q, NK, NV = 8, 4, 32, 16
    vocab = 1 + NK + NV
    steps = 2500
    torch.manual_seed(0)
    print(f"device={dev}")

    def ssm(use_mem):
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=160, n_layers=3, state_dim=64,
                                         expand=1, d_ff_mult=2, use_phase_memory=use_mem,
                                         mem_heads=4, mem_head_dim=32))
    configs = {
        "SSM-only (oscillatory)": ssm(False),
        "OURS (SSM + phase-memory)": ssm(True),
        "Transformer": TransformerLM(TransformerConfig(vocab_size=vocab, d_model=160, n_layers=3,
                                                       n_heads=4, d_ff_mult=2, block_size=128)),
    }
    print(f"MQAR: {P} pairs, {Q} queries, {NK} keys, {NV} values  (chance ~{1/NV:.3f})")
    results = {}
    for name, m in configs.items():
        torch.manual_seed(0)
        acc = train_on_mqar(m, steps=steps, n_pairs=P, n_queries=Q, n_keys=NK, n_values=NV,
                            batch=64, lr=3e-3, device=dev, log_every=500)
        results[name] = (acc, m.num_params())
        print(f"  {name:30s}  params={m.num_params()/1e6:.2f}M  recall={acc:.3f}")
    print("\n=== MQAR recall (higher = better associative memory) ===")
    for name, (acc, p) in results.items():
        print(f"  {name:30s}  {acc:.3f}")
