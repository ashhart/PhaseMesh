"""Model-agnostic trainer for the PhaseSSM vs transformer bake-off.

Trains either architecture on identical data / optimizer / schedule / budget, so
the only variable is the architecture. Logs train + val bits-per-char and
throughput, checkpoints the best model, and writes a JSON run log.

    python -m phase_ssm.train --model phasessm   --out runs/ssm   ...
    python -m phase_ssm.train --model transformer --out runs/xf   ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import ByteData, get_text8, stats
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM


def build_model(args, vocab_size: int):
    if args.model == "phasessm":
        cfg = PhaseSSMConfig(vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                             state_dim=args.state_dim, expand=args.expand, d_ff_mult=args.d_ff_mult,
                             short_conv=args.short_conv, ssm_backend=args.ssm_backend,
                             ssm_chunk=args.ssm_chunk, ssm_auto_threshold=args.ssm_auto_threshold,
                             dropout=args.dropout, use_phase_memory=args.use_phase_memory,
                             mem_heads=args.mem_heads, mem_head_dim=args.mem_head_dim,
                             mem_linear=args.mem_linear, mem_chunk=args.mem_chunk,
                             mem_layout=args.mem_layout)
        return PhaseSSMLM(cfg), cfg.__dict__
    cfg = TransformerConfig(vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                            n_heads=args.n_heads, d_ff_mult=args.d_ff_mult, block_size=args.seq,
                            dropout=args.dropout)
    return TransformerLM(cfg), cfg.__dict__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["phasessm", "transformer"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--d-model", dest="d_model", type=int, default=384)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=6)
    p.add_argument("--state-dim", dest="state_dim", type=int, default=64)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=6)
    p.add_argument("--d-ff-mult", dest="d_ff_mult", type=int, default=3)
    p.add_argument("--short-conv", dest="short_conv", type=int, default=4)
    p.add_argument("--ssm-backend", choices=["fft", "auto", "real_chunked", "fixed_triton", "skip"], default="fft",
                   help="Training backend for PhaseSSM temporal mixer.")
    p.add_argument("--ssm-chunk", type=int, default=64,
                   help="Chunk length for --ssm-backend real_chunked.")
    p.add_argument("--ssm-auto-threshold", type=int, default=32768,
                   help="For --ssm-backend auto, use FFT at or below this sequence length.")
    p.add_argument("--allow-diagnostic-backend", action="store_true",
                   help="Allow quality-degrading diagnostic backends such as fixed_triton or skip.")
    p.add_argument("--use-phase-memory", action="store_true",
                   help="Add the phase-binding associative-memory branch to each PhaseSSM block.")
    p.add_argument("--mem-heads", type=int, default=4,
                   help="Phase-binding memory heads when --use-phase-memory is enabled.")
    p.add_argument("--mem-head-dim", type=int, default=32,
                   help="Per-head phase dimension for --use-phase-memory.")
    p.add_argument("--mem-linear", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the O(L) linear phase memory (default on); --no-mem-linear for the O(L^2) reference form.")
    p.add_argument("--mem-chunk", type=int, default=64,
                   help="Chunk size for the linear phase-memory hybrid form.")
    p.add_argument("--mem-layout", choices=["parallel", "sequential"], default="parallel",
                   help="Block layout: parallel (SSM ∥ memory, target) or sequential (mixer→memory→ffn, ablation).")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", dest="min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--eval-interval", dest="eval_interval", type=int, default=500)
    p.add_argument("--eval-iters", dest="eval_iters", type=int, default=50)
    p.add_argument("--train-log-interval", dest="train_log_interval", type=int, default=0,
                   help="Print train-only throughput every N optimization steps. 0 disables.")
    p.add_argument("--skip-initial-eval", action="store_true",
                   help="Do not run the expensive step-0 eval/checkpoint pass.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    diagnostic_backends = {"fixed_triton", "skip"}
    if args.model == "phasessm" and args.ssm_backend in diagnostic_backends and not args.allow_diagnostic_backend:
        raise SystemExit(
            f"--ssm-backend {args.ssm_backend!r} is diagnostic and can degrade training quality. "
            "Use --allow-diagnostic-backend only for timing/probing runs. "
            "Use --ssm-backend fft for quality training."
        )
    if (
        args.model == "phasessm"
        and args.ssm_backend == "auto"
        and args.seq > args.ssm_auto_threshold
        and not args.allow_diagnostic_backend
    ):
        raise SystemExit(
            "--ssm-backend 'auto' would route this sequence length through the frozen recurrent path. "
            "Use --ssm-backend fft for quality training, lower --seq, raise --ssm-auto-threshold, "
            "or pass --allow-diagnostic-backend for timing/probing only."
        )

    torch.manual_seed(args.seed)
    dev = args.device
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    data = ByteData(get_text8(args.data_root))
    print(f"[data] {stats(data)}")
    model, model_cfg = build_model(args, data.vocab_size)
    print(f"[model] {args.model}  params={model.num_params():,}  cfg={model_cfg}")

    # The training loop is shared with phase_ssm.bakeoff via train_core, so a
    # single-model run here and the bake-off comparison there use identical code.
    from .train_core import train_model
    train_model(
        model, data, name=args.model, out=Path(args.out), steps=args.steps,
        batch=args.batch, seq=args.seq, lr=args.lr, min_lr=args.min_lr,
        warmup=args.warmup, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        grad_accum=args.grad_accum, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, train_log_interval=args.train_log_interval,
        skip_initial_eval=args.skip_initial_eval, device=dev, amp_dtype=amp_dtype,
        compile=args.compile, model_cfg=model_cfg, seed=args.seed,
    )


if __name__ == "__main__":
    main()
