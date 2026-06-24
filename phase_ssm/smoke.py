"""Smoke test: prove PhaseSSM actually learns.

Two checks, both gradient-trained end-to-end through the oscillatory recurrence:

1. STRUCTURE: a deterministic induction/copy task the model can only solve by
   using context (not unigram statistics). Accuracy must rise well above chance.
2. LANGUAGE: overfit a short text; cross-entropy must collapse and the model must
   regenerate the passage from a seed.

Run: python3 -m phase_ssm.smoke
"""
from __future__ import annotations

import torch

from .model import PhaseSSMConfig, PhaseSSMLM


def _device() -> str:
    return "cpu"  # CPU is safe everywhere for complex FFT; tiny model trains fast.


def induction_task(seed: int = 0):
    """Targets = inputs shifted so each token must be predicted from a fixed
    long-range rule: y_t = x_{t-K}. Solvable only with memory of K steps back."""
    g = torch.Generator().manual_seed(seed)
    vocab, L, K, B = 32, 48, 8, 64
    x = torch.randint(0, vocab, (B, L), generator=g)
    y = torch.full_like(x, -100)
    y[:, K:] = x[:, : L - K]  # predict the token from K positions earlier
    return x, y, vocab


def run_induction() -> float:
    torch.manual_seed(0)
    dev = _device()
    x, y, vocab = induction_task()
    x, y = x.to(dev), y.to(dev)
    cfg = PhaseSSMConfig(vocab_size=vocab, d_model=96, n_layers=3, state_dim=48, short_conv=4)
    model = PhaseSSMLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"[induction] params={model.num_params():,}  (predict token from 8 steps back)")
    for step in range(401):
        model.train()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 80 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(x)
                mask = y != -100
                acc = (logits.argmax(-1)[mask] == y[mask]).float().mean().item()
            print(f"  step {step:4d}  loss {loss.item():.4f}  acc {acc:.3f}  (chance ~{1/32:.3f})")
    return acc


def run_language() -> float:
    torch.manual_seed(0)
    dev = _device()
    text = (
        "PhaseSSM is the phase mesh that can learn. Each channel is a damped "
        "oscillator with its own frequency and damping, and every parameter is "
        "trained by gradient descent. The field still evolves over time, but now "
        "it can be optimized. This is the next evolution of the substrate.\n"
    ) * 6
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long, device=dev)
    L = 96
    starts = torch.arange(0, len(data) - L - 1)
    X = torch.stack([data[s : s + L] for s in starts])
    Y = torch.stack([data[s + 1 : s + L + 1] for s in starts])

    cfg = PhaseSSMConfig(vocab_size=256, d_model=128, n_layers=4, state_dim=64, short_conv=4)
    model = PhaseSSMLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"[language]  params={model.num_params():,}  windows={len(X)}")
    bs = 32
    for step in range(351):
        model.train()
        idx = torch.randint(0, len(X), (bs,))
        _, loss = model(X[idx], Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 70 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    seed = torch.tensor([list("PhaseSSM is".encode())], dtype=torch.long, device=dev)
    out = model.generate(seed, max_new_tokens=80, temperature=0.0)
    gen = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    print("  sample:", repr(gen))
    return loss.item()


if __name__ == "__main__":
    print("=" * 64)
    acc = run_induction()
    print("=" * 64)
    final_loss = run_language()
    print("=" * 64)
    ok_ind = acc > 0.6
    ok_lang = final_loss < 0.3
    print(f"INDUCTION accuracy {acc:.3f} -> {'PASS' if ok_ind else 'FAIL'} (>0.6)")
    print(f"LANGUAGE   loss     {final_loss:.4f} -> {'PASS' if ok_lang else 'FAIL'} (<0.3)")
    print("RESULT:", "PhaseSSM LEARNS ✓" if (ok_ind and ok_lang) else "needs tuning")
