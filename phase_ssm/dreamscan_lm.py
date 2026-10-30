"""DreamScan LM — the post-transformer composition.

The architecture this program was building toward: a language model whose only
token-mixers are (1) a non-abelian parallel scan (temporal composition, outside
the TC0 cell that bounds every diagonal SSM, every linear attention, and BDH) and
(2) a phase-binding recall memory (associative content-addressable recall,
attention-class but at fixed O(L) state with no KV cache). The two share one
normalized input and contribute in parallel; the block is iterated to a damped
fixed point.

Why this is the empty chair, not a remix:

  * Every "post-attention" attempt replaces attention with ONE mechanism and
    inherits that mechanism's blind spot. Mamba = diagonal SSM -> collapses on
    recall (MQAR). RetNet/GLA/BDH = linear attention -> commutative -> cannot
    track non-solvable structure. Both failures are measured in this repo.
  * This composition uses TWO mechanisms, each validated separately for the job
    the other can't do:
      - recall memory: attention-class retrieval at fixed state (TARGET_ARCH.md's
        go/no-go gate; the |<q,k>|^2 retrieval weight is provably separable into
        a linear recurrence, so no O(L^2) tax for the recall that hybrids pay).
      - non-abelian scan: non-commutative composition at O(log L) parallel depth,
        the one object outside TC0 (Rung A; the thing BDH structurally cannot do,
        3-seed confirmed).
  * The parallel layout is exactly PhaseBlock's target architecture
    (model.py:257-264), with the temporal mixer swapped from the oscillatory SSM
    to the non-abelian scan. The recall half is unchanged.

The fixed-point iteration (z <- z + sigmoid(a)(block(z) - z), iterated S times)
makes the per-sweep operator state-conditional: sweep s recomputes transitions
from sweep s-1's output. This is DREAM's two-timescale thesis (settling's
adaptive depth in units of parallel sweeps) with the oscillator replaced by the
non-abelian scan and the recall memory both inside the fixed-point operator.

Baseline targets (this repo's text8 harness):
  transformer 1.350 bpc, PhaseSSM-6L 1.377 bpc.
The pure-scan DreamScan (no recall memory) landed ~1.74 at step 7000/12000 and
was losing — this composition is the attempt to close that gap by adding the
recall half the pure scan lacked.

Run:
  python -m phase_ssm.dreamscan_lm --use-memory --compile          # the composition
  python -m phase_ssm.dreamscan_lm --no-use-memory --compile       # pure-scan ablation
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonabelian_scan import NonAbelianLayer
from .data import ByteData, get_text8, stats


class RMSNorm(nn.Module):
    """RMSNorm (matches model.py's normalizer for apples-to-apples with PhaseBlock)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DreamScanBlock(nn.Module):
    """The composition: non-abelian scan || phase-binding recall, parallel norm, + FFN.

    Mirrors PhaseBlock's parallel layout (model.py:288-295): temporal and
    associative branches read the SAME normalized state h and contribute in
    parallel; the FFN runs separately behind its own norm. The temporal branch
    is the non-abelian scan (not the oscillatory SSM); the recall branch is
    PhaseBindingMemory unchanged. Both are linear-in-L recurrences with fixed
    state, so this is the layout that fuses into one scan kernel later.
    """

    def __init__(self, d_model, scan_heads, scan_head_dim, scan_n_h,
                 mem_heads, mem_head_dim, d_ff_mult=2, beta_init="identity"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        # temporal branch: non-abelian scan (Householder-product transitions)
        self.scan = NonAbelianLayer(
            d_model, H=scan_heads, d_h=scan_head_dim, n_h=scan_n_h,
            transition="householder", scan=True, beta_init=beta_init)
        # associative branch: phase-binding recall memory (unchanged, drop-in)
        from .memory import PhaseBindingMemory
        self.memory = PhaseBindingMemory(
            d_model, n_heads=mem_heads, head_dim=mem_head_dim, linear=True, chunk=64)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff_mult * d_model)

    def forward(self, x):
        h = self.norm1(x)
        # parallel: temporal scan + associative recall share the same normalized input
        mixed = self.scan(h) + self.memory(h)
        x = x + mixed
        x = x + self.ffn(self.norm2(x))
        return x


class DreamScanPureScanBlock(nn.Module):
    """Ablation: scan + FFN only, no recall memory. Matches the earlier pure DreamScan."""

    def __init__(self, d_model, scan_heads, scan_head_dim, scan_n_h,
                 d_ff_mult=2, beta_init="identity"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.scan = NonAbelianLayer(
            d_model, H=scan_heads, d_h=scan_head_dim, n_h=scan_n_h,
            transition="householder", scan=True, beta_init=beta_init)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff_mult * d_model)

    def forward(self, x):
        x = x + self.scan(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DreamScanLM(nn.Module):
    """Byte-level LM: embedding -> [damped iterated DreamScanBlock] -> head.

    The iterated block (scan || memory, parallel norm) is relaxed to a fixed
    point via z <- z + sigmoid(damp)(block(z) - z). One weight-tied block per
    n_layer; each iterated `iters` times. State cost fixed in L (scan's (H,d_h)
    state + memory's dk^2 state per head); depth cost O(S log L) parallel.
    """

    def __init__(self, vocab=256, d_model=256, n_layer=1,
                 scan_heads=8, scan_head_dim=32, scan_n_h=8,
                 mem_heads=4, mem_head_dim=32, iters=4, d_ff_mult=2,
                 damp_init=-2.0, beta_init="identity",
                 use_memory=True, tie_embeddings=True):
        super().__init__()
        self.cfg = dict(vocab=vocab, d_model=d_model, n_layer=n_layer,
                        scan_heads=scan_heads, scan_head_dim=scan_head_dim, scan_n_h=scan_n_h,
                        mem_heads=mem_heads, mem_head_dim=mem_head_dim, iters=iters,
                        d_ff_mult=d_ff_mult, damp_init=damp_init, beta_init=beta_init,
                        use_memory=use_memory, tie_embeddings=tie_embeddings)
        self.iters = iters
        self.use_memory = use_memory
        self.embed = nn.Embedding(vocab, d_model)
        if use_memory:
            self.blocks = nn.ModuleList([
                DreamScanBlock(d_model, scan_heads, scan_head_dim, scan_n_h,
                               mem_heads, mem_head_dim, d_ff_mult, beta_init)
                for _ in range(n_layer)
            ])
        else:
            self.blocks = nn.ModuleList([
                DreamScanPureScanBlock(d_model, scan_heads, scan_head_dim, scan_n_h,
                                       d_ff_mult, beta_init)
                for _ in range(n_layer)
            ])
        self.damps = nn.ParameterList([
            nn.Parameter(torch.tensor(float(damp_init))) for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self._init()
        if tie_embeddings:
            self.head.weight = self.embed.weight

    @staticmethod
    def _build_init(std):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
        return init

    def _init(self):
        self.apply(self._build_init(0.02))

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        for blk, damp in zip(self.blocks, self.damps):
            a = torch.sigmoid(damp)
            for _ in range(self.iters):
                x = x + a * (blk(x) - x)          # damped contraction toward the
                                                  # fixed point of (scan || memory)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
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
    ap.add_argument("--out", default="runs/dreamscan_lm")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--scan-heads", type=int, default=8)
    ap.add_argument("--scan-head-dim", type=int, default=32)
    ap.add_argument("--scan-n-h", type=int, default=8)
    ap.add_argument("--mem-heads", type=int, default=4)
    ap.add_argument("--mem-head-dim", type=int, default=32)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--d-ff-mult", type=int, default=2)
    ap.add_argument("--damp-init", type=float, default=-2.0)
    ap.add_argument("--beta-init", choices=["identity", "split", "reflect"], default="identity")
    ap.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True,
                    help="composition (default) vs pure-scan ablation (--no-use-memory)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = torch.bfloat16 if dev == "cuda" else torch.float32
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = DreamScanLM(
        vocab=256, d_model=args.d_model, n_layer=args.n_layer,
        scan_heads=args.scan_heads, scan_head_dim=args.scan_head_dim, scan_n_h=args.scan_n_h,
        mem_heads=args.mem_heads, mem_head_dim=args.mem_head_dim, iters=args.iters,
        d_ff_mult=args.d_ff_mult, damp_init=args.damp_init, beta_init=args.beta_init,
        use_memory=args.use_memory)
    tag = "composition(scan+mem)" if args.use_memory else "pure-scan(ablation)"
    print(f"[dreamscan-LM] {tag} n_layer={args.n_layer} iters={args.iters} d_model={args.d_model} "
          f"scan(H={args.scan_heads},d_h={args.scan_head_dim},n_h={args.scan_n_h}) "
          f"mem(H={args.mem_heads},dk={args.mem_head_dim}) dev={dev}", flush=True)
    print(f"[dreamscan-LM] params={model.num_params():,} (baselines: transformer 1.350 bpc, "
          f"PhaseSSM-6L 1.377 bpc on text8)", flush=True)
    data = ByteData(get_text8(args.data))
    print(f"[data] {stats(data)}", flush=True)

    from .train_core import train_model
    train_model(model, data, name="dreamscan-lm", out=Path(args.out), steps=args.steps,
                batch=args.batch, seq=args.seq, lr=args.lr, min_lr=1e-4, warmup=300,
                weight_decay=0.1, grad_clip=1.0, grad_accum=1, eval_interval=500, eval_iters=40,
                train_log_interval=0, skip_initial_eval=False, device=dev, amp_dtype=amp,
                compile=args.compile, model_cfg=model.cfg, seed=args.seed)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
