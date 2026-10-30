"""EXPERIMENT 3 — ternary parity at the block: does a full-ternary PhaseSSM block
match fp within noise on bpc?

This is the hardware-investment pillar. The whole "compute economics compound on
custom silicon" thesis rides on ternary being a NATIVE, load-bearing precision —
not a compression-afterthought. If a ternary block matches fp at small scale
(BitNet's claim), the hw boys have a real bet: silicon that does 1.58-bit math
natively gets ~10x the FLOPs/$ of GPU-bound fp, and the architecture is designed
around that from the ground up.

This is half-done in AGI-WorldLLM/DeltaLM. Here we measure it cleanly on the
PhaseSSM block: same config, same data, same steps, same seeds — only the weight
precision changes. A clean tie (within noise) is the green result.

ARMS
  fp      : PhaseSSMLM with all nn.Linear (standard fp32/bf16).
  ternary : same block but every Linear in the FFN and the mixer projections is
            TernaryLinear (BitNet 1.58, STE). Embeddings/head stay fp (standard
            BitNet practice; IO tables are a tiny fraction of params).

KILL CONDITION
  ternary bpc > fp bpc + TERNARY_GAP by a clear margin at convergence, multi-seed.
  A small gap is expected and acceptable (BitNet reports ~1% at scale); a large
  one means ternary is not load-bearing at this width and the hw pitch weakens.

Run on the GB10:
    python -m phase_ssm.ternary_parity_gate --steps 6000 --seeds 3 --device cuda
Smoke:
    python -m phase_ssm.ternary_parity_gate --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PhaseSSMConfig, RMSNorm, OscillatorySSM, SwiGLU
from .ternary import TernaryLinear
from .poc_common import D_MODEL, N_LAYERS, STATE_DIM, EXPAND, D_FF_MULT, HEAD_DIM, MEM_HEADS


# --------------------------------------------------------------------------- #
# A PhaseSSM block whose FFN + projections are ternary. The SSM poles and the
# associative memory stay fp (they're tiny and not the matmul-heavy part the hw
# bet targets). This isolates the ternary question to the matmul bulk.
# --------------------------------------------------------------------------- #
class _TernaryPhaseTimeMixer(nn.Module):
    """PhaseTimeMixer with ternary in/gate/out projections."""
    def __init__(self, cfg: PhaseSSMConfig):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.use_gate = cfg.use_gate
        self.in_proj = TernaryLinear(cfg.d_model, d_inner)
        self.gate_proj = TernaryLinear(cfg.d_model, d_inner) if cfg.use_gate else None
        self.short_conv = (
            nn.Conv1d(d_inner, d_inner, cfg.short_conv, groups=d_inner, padding=cfg.short_conv - 1)
            if cfg.short_conv > 0 else None
        )
        self.ssm = OscillatorySSM(d_inner, cfg.state_dim, cfg.dt_min, cfg.dt_max,
                                  backend=cfg.ssm_backend, chunk=cfg.ssm_chunk,
                                  auto_threshold=cfg.ssm_auto_threshold)
        self.out_proj = TernaryLinear(d_inner, cfg.d_model)

    def forward(self, x):
        u = self.in_proj(x)
        if self.short_conv is not None:
            L = u.shape[1]
            u = self.short_conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        y = self.ssm(u)
        if self.gate_proj is not None:
            y = y * F.silu(self.gate_proj(x))
        return self.out_proj(y)


class _TernarySwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_in = TernaryLinear(d_model, 2 * hidden)
        self.w_out = TernaryLinear(hidden, d_model)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


def make_block(cfg: PhaseSSMConfig, ternary: bool) -> nn.Module:
    """Return a PhaseBlock with the FFN + mixer projections optionally ternary."""
    if not ternary:
        from .model import PhaseBlock
        return PhaseBlock(cfg)

    # Build a ternary variant by patching the standard block's submodules.
    from .model import PhaseBlock
    blk = PhaseBlock(cfg)
    # Replace the mixer and ffn with ternary versions (keep norm, keep memory fp).
    blk.mixer = _TernaryPhaseTimeMixer(cfg)
    if blk.ffn is not None:
        blk.ffn = _TernarySwiGLU(cfg.d_model, cfg.d_ff_mult * cfg.d_model)
    return blk


class TernaryPhaseSSMLM(nn.Module):
    """PhaseSSMLM with ternary FFN+projections when ternary=True. Else identical to PhaseSSMLM."""
    def __init__(self, cfg: PhaseSSMConfig, ternary: bool = True):
        super().__init__()
        self.cfg = cfg
        self.ternary = ternary
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(make_block(cfg, ternary) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p)); total += p.numel()
        return total

    def ternary_params(self) -> int:
        n = 0
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                n += m.weight.numel()
        return n


# --------------------------------------------------------------------------- #
# Training on a char corpus.
# --------------------------------------------------------------------------- #
def train_char(model, *, x_tr, y_tr, x_val, y_val, steps, lr, device, batch, seed=0):
    from .poc_common import char_bpc
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.to(device); x_tr = x_tr.to(device); y_tr = y_tr.to(device)
    x_val = x_val.to(device); y_val = y_val.to(device)
    N = x_tr.shape[0]
    warmup = min(1000, steps // 10)
    torch.manual_seed(seed)
    best = float("inf")
    for step in range(steps):
        if step < warmup:
            lr_c = lr * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            lr_c = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr_c
        model.train()
        idx = torch.randint(0, N, (batch,), device=device)
        _, loss = model(x_tr[idx], y_tr[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                _, vl = model(x_val, y_val)
            best = min(best, vl.item())
    model.eval()
    with torch.no_grad():
        _, vl = model(x_val, y_val)
    return {"val_bpc": char_bpc(vl.item()), "best_bpc": char_bpc(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corpus", default=None, help="path to a text file (else synthetic fallback)")
    ap.add_argument("--out", default="results/poc_ternary_parity.json")
    ap.add_argument("--markdown", default="results/poc_ternary_parity.md")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps = 20; args.seeds = 1; args.batch = 8; args.seq = 32
        args.out = "results/poc_ternary_smoke.json"
        args.markdown = "results/poc_ternary_smoke.md"
        import phase_ssm.poc_common as _pc
        _pc.D_MODEL = 48
        _pc.N_LAYERS = 1
        _pc.STATE_DIM = 16
        print("[SMOKE] tiny run to verify the pipeline. NOT a real result.")

    from .poc_common import build_char_dataset
    print("TERNARY PARITY AT THE BLOCK (EXPERIMENT 3)")
    print(f"  config: d={D_MODEL} L={N_LAYERS} state={STATE_DIM}; steps={args.steps} "
          f"seeds={args.seeds} device={args.device} corpus={args.corpus or 'synthetic'}")

    x, y, _, _, V = build_char_dataset(args.corpus, seq_len=args.seq, vocab=None)
    # train/val split (deterministic)
    n_tr = int(0.9 * x.shape[0])
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_val, y_val = x[n_tr:], y[n_tr:]
    print(f"  vocab={V} chars; {n_tr} train / {x.shape[0]-n_tr} val sequences")

    def cfg():
        return PhaseSSMConfig(vocab_size=V, d_model=D_MODEL, n_layers=N_LAYERS,
                              state_dim=STATE_DIM, expand=EXPAND, d_ff_mult=D_FF_MULT,
                              mem_heads=MEM_HEADS, mem_head_dim=HEAD_DIM,
                              use_phase_memory=False)

    results = {"fp": [], "ternary": []}
    TERNARY_GAP = 0.05   # bpc; acceptable ternary overhead
    t0 = time.time()
    for s in range(args.seeds):
        for ternary in (False, True):
            torch.manual_seed(s)
            m = TernaryPhaseSSMLM(cfg(), ternary=ternary)
            r = train_char(m, x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val,
                           steps=args.steps, lr=args.lr, device=args.device,
                           batch=args.batch, seed=s)
            tag = "ternary" if ternary else "fp"
            r["seed"] = s
            r["params_M"] = m.num_params() / 1e6
            r["ternary_param_frac"] = (m.ternary_params() / m.num_params()) if ternary else 0.0
            results[tag].append(r)
            print(f"  seed{s} {tag:<8} val_bpc={r['val_bpc']:.4f} (best {r['best_bpc']:.4f}) "
                  f"{r['params_M']:.2f}M params")
        json.dump(results, open(args.out, "w"), indent=2)

    fp_mean = sum(r["val_bpc"] for r in results["fp"]) / len(results["fp"])
    ter_mean = sum(r["val_bpc"] for r in results["ternary"]) / len(results["ternary"])
    gap = ter_mean - fp_mean
    print(f"\n{'='*70}")
    print(f"fp mean val_bpc      : {fp_mean:.4f}")
    print(f"ternary mean val_bpc : {ter_mean:.4f}")
    print(f"gap (ternary - fp)   : {gap:+.4f}  (acceptable <= {TERNARY_GAP})")
    if gap <= TERNARY_GAP:
        print("VERDICT: GREEN — ternary matches fp within noise. The hw bet holds.")
    elif gap <= TERNARY_GAP * 2:
        print("VERDICT: PARTIAL — small ternary gap. Likely fine at scale; widen/trains-longer.")
    else:
        print("VERDICT: RED — ternary loses meaningful quality. The hw pitch weakens.")
    print(f"(elapsed {time.time()-t0:.0f}s)")
    print(f"{'='*70}")

    lines = [
        "# POC Experiment 3 — Ternary Parity at the Block",
        "",
        f"fp vs full-ternary FFN+projections (BitNet 1.58, STE), matched config. "
        f"d={D_MODEL}, L={N_LAYERS}, steps={args.steps}, seeds={args.seeds}.",
        "",
        "| seed | fp val_bpc | ternary val_bpc | gap |",
        "|---:|---:|---:|---:|",
    ]
    for s in range(args.seeds):
        f = results["fp"][s]; t = results["ternary"][s]
        lines.append(f"| {s} | {f['val_bpc']:.4f} | {t['val_bpc']:.4f} | {t['val_bpc']-f['val_bpc']:+.4f} |")
    lines += ["", f"**mean gap: {gap:+.4f} bpc (acceptable <= {TERNARY_GAP})**"]
    open(args.markdown, "w").write("\n".join(lines))
    print(f"wrote {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
