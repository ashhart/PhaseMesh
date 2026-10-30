"""Transfer test: does serial-depth length-extrapolation appear on REAL text?

Train byte-level LMs on text8 at a short context, then measure bits-per-char at
LONGER contexts than trained. The make-or-break question for the whole serial-
depth bet: does the settling model (and the SSM class) extrapolate to longer
sequences where a transformer degrades? settling vs PhaseSSM (diag-SSM) vs
transformer (RoPE). Lower bpc = better; the story is in how bpc changes past the
training length.
"""
from __future__ import annotations
import argparse, math, time
import torch

from .data import ByteData, get_text8, stats
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM
from .settling_lm import SettlingByteLM


def train_model(model, data, dev, *, steps, seq, batch, lr, warmup=400):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.to(dev)
    for step in range(steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
            lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, y = data.get_batch("train", batch, seq, dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = ByteData(get_text8("data"))
    maxseq = max(args.eval_seqs)
    print(f"LANG LENGTH-GEN | {stats(data)} | train_seq={args.train_seq} eval={args.eval_seqs} "
          f"steps={args.steps} d={args.d} iters={args.iters} dev={dev}", flush=True)

    def settling():
        cfg = PhaseSSMConfig(vocab_size=256, d_model=args.d, n_layers=1, state_dim=64,
                             expand=2, d_ff_mult=4, use_phase_memory=False)
        return SettlingByteLM(cfg, iters=args.iters)

    def diagssm():
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=256, d_model=args.d, n_layers=4, state_dim=64,
                                         expand=2, d_ff_mult=4, use_phase_memory=False))

    def xf():
        return TransformerLM(TransformerConfig(vocab_size=256, d_model=args.d, n_layers=4,
                                               n_heads=max(1, args.d // 64), d_ff_mult=4,
                                               block_size=maxseq, pos_enc="rope"))

    results = {}
    for name, mk, lr in [("settling", settling, args.lr), ("diag-SSM", diagssm, args.lr),
                         ("transformer", xf, args.lr)]:
        torch.manual_seed(0)
        m = mk(); p = m.num_params()
        t0 = time.time()
        train_model(m, data, dev, steps=args.steps, seq=args.train_seq, batch=args.batch, lr=lr)
        row = {s: data.eval_bpc(m, "val", args.eval_batch, s, dev, iters=40) for s in args.eval_seqs}
        results[name] = (row, p)
        print(f"  {name:<12} ({p/1e6:.1f}M, {time.time()-t0:.0f}s) | "
              + "  ".join(f"L{s}:{row[s]:.3f}" for s in args.eval_seqs), flush=True)

    print(f"\n{'model':<12} | " + " | ".join(f"bpc@{s}" for s in args.eval_seqs)
          + " | delta(train->max)", flush=True)
    base = args.train_seq
    for name, (row, p) in results.items():
        delta = row[maxseq] - row[base]
        print(f"{name:<12} | " + " | ".join(f"{row[s]:>6.3f} " for s in args.eval_seqs)
              + f"| {delta:+.3f}", flush=True)
    print("(lower bpc better; positive delta = degrades when extrapolating past training length)", flush=True)
    print("LANG-LENGTHGEN DONE", flush=True)


if __name__ == "__main__":
    main()
