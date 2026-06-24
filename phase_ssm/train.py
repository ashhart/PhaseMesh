"""Model-agnostic trainer for the PhaseSSM vs transformer bake-off.

Trains either architecture on identical data / optimizer / schedule / budget, so
the only variable is the architecture. Logs train + val bits-per-char and
throughput, checkpoints the best model, and writes a JSON run log.

    python -m phase_ssm.train --model phasessm   --out runs/ssm   ...
    python -m phase_ssm.train --model transformer --out runs/xf   ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
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
                             dropout=args.dropout)
        return PhaseSSMLM(cfg), cfg.__dict__
    cfg = TransformerConfig(vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                            n_heads=args.n_heads, d_ff_mult=args.d_ff_mult, block_size=args.seq,
                            dropout=args.dropout)
    return TransformerLM(cfg), cfg.__dict__


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.steps:
        return args.min_lr
    ratio = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * ratio))


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
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    data = ByteData(get_text8(args.data_root))
    print(f"[data] {stats(data)}")
    model, model_cfg = build_model(args, data.vocab_size)
    model = model.to(dev)
    n_params = model.num_params()
    print(f"[model] {args.model}  params={n_params:,}  cfg={model_cfg}")
    if args.compile:
        model = torch.compile(model)

    decay, no_decay = [], []
    for pn, pp in model.named_parameters():
        (decay if pp.dim() >= 2 else no_decay).append(pp)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    log = {"model": args.model, "params": n_params, "cfg": model_cfg, "args": vars(args), "history": []}
    best_val = float("inf")
    t0 = time.time()
    tok_seen = 0
    run_t = time.time()
    train_log_tokens = 0
    train_log_t = time.time()

    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        should_eval = (
            args.eval_interval > 0
            and step % args.eval_interval == 0
            and not (step == 0 and args.skip_initial_eval)
        )
        if should_eval:
            eval_t0 = time.time()
            val_bpc = data.eval_bpc(model, "val", args.batch, args.seq, dev, args.eval_iters)
            train_bpc = data.eval_bpc(model, "train", args.batch, args.seq, dev, max(10, args.eval_iters // 5))
            eval_s = time.time() - eval_t0
            dt = time.time() - run_t; tps = tok_seen / dt if dt > 0 else 0
            print(f"[{args.model}] step {step:6d}  train_bpc {train_bpc:.4f}  val_bpc {val_bpc:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  interval_tok/s {tps/1e3:.0f}k  "
                  f"eval_s {eval_s:.1f}  elapsed_s {time.time()-t0:.0f}")
            log["history"].append({"step": step, "train_bpc": train_bpc, "val_bpc": val_bpc,
                                   "lr": opt.param_groups[0]["lr"], "tok_per_s": tps,
                                   "eval_s": eval_s, "elapsed_s": time.time() - t0})
            (out / "log.json").write_text(json.dumps(log, indent=2))
            if val_bpc < best_val:
                best_val = val_bpc
                torch.save({"model_state": getattr(model, "_orig_mod", model).state_dict(),
                            "cfg": model_cfg, "model_type": args.model, "step": step, "val_bpc": val_bpc},
                           out / "best.pt")
            tok_seen = 0; run_t = time.time()

        if step == args.steps:
            break

        model.train()
        opt.zero_grad(set_to_none=True)
        train_step_t0 = time.time()
        for _ in range(args.grad_accum):
            x, y = data.get_batch("train", args.batch, args.seq, dev)
            with torch.autocast(device_type=dev.split(":")[0], dtype=amp_dtype, enabled=(amp_dtype == torch.bfloat16)):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            tok_seen += x.numel()
            train_log_tokens += x.numel()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        train_step_s = time.time() - train_step_t0

        if args.train_log_interval > 0 and (step + 1) % args.train_log_interval == 0:
            train_log_dt = time.time() - train_log_t
            train_tps = train_log_tokens / train_log_dt if train_log_dt > 0 else 0
            print(f"[{args.model}] train_step {step + 1:6d}  loss {loss.item() * args.grad_accum:.4f}  "
                  f"train_tok/s {train_tps/1e3:.1f}k  step_s {train_step_s:.2f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  elapsed_s {time.time()-t0:.0f}")
            log.setdefault("train_history", []).append({
                "step": step + 1,
                "loss": loss.item() * args.grad_accum,
                "train_tok_per_s": train_tps,
                "step_s": train_step_s,
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": time.time() - t0,
            })
            (out / "log.json").write_text(json.dumps(log, indent=2))
            train_log_tokens = 0
            train_log_t = time.time()

    log["best_val_bpc"] = best_val
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[{args.model}] DONE  best_val_bpc={best_val:.4f}  params={n_params:,}  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
