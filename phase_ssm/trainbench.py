"""Training-step microbenchmark for PhaseSSM backends.

This isolates the optimization-step hot path: random byte batches, forward,
cross-entropy, backward, gradient clipping, optimizer step. It is intentionally
separate from ``phase_ssm.train`` so eval/checkpoint/data timing cannot distort
kernel work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .model import PhaseSSMConfig, PhaseSSMLM


def build_model(args: argparse.Namespace) -> PhaseSSMLM:
    cfg = PhaseSSMConfig(
        vocab_size=256,
        d_model=args.d_model,
        n_layers=args.n_layers,
        state_dim=args.state_dim,
        expand=args.expand,
        d_ff_mult=args.d_ff_mult,
        short_conv=args.short_conv,
        ssm_backend=args.ssm_backend,
        ssm_chunk=args.ssm_chunk,
        ssm_auto_threshold=args.ssm_auto_threshold,
        use_mixer=not args.no_mixer,
        use_ffn=not args.no_ffn,
        use_gate=not args.no_gate,
    )
    return PhaseSSMLM(cfg)


def one_batch(args: argparse.Namespace, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 256, (args.batch, args.seq), device=device)
    y = torch.randint(0, 256, (args.batch, args.seq), device=device)
    return x, y


def measure(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = args.device
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = build_model(args).to(device)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    tokens_per_step = args.batch * args.seq
    losses: list[float] = []
    step_times: list[float] = []

    for step in range(args.warmup + args.steps):
        x, y = one_batch(args, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=(amp_dtype == torch.bfloat16)):
            _, loss = model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= args.warmup:
            losses.append(float(loss.item()))
            step_times.append(dt)
        print(
            f"[trainbench] backend={args.ssm_backend} step={step + 1:4d} "
            f"loss={loss.item():.4f} step_s={dt:.3f}",
            flush=True,
        )

    mean_step_s = sum(step_times) / max(1, len(step_times))
    tok_s = tokens_per_step / mean_step_s if mean_step_s > 0 else 0.0
    result: dict[str, object] = {
        "backend": args.ssm_backend,
        "params": getattr(model, "_orig_mod", model).num_params(),
        "cfg": getattr(model, "_orig_mod", model).cfg.__dict__,
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": tokens_per_step,
        "mean_step_s": mean_step_s,
        "tok_per_s": tok_s,
        "loss_mean": sum(losses) / max(1, len(losses)),
        "step_times": step_times,
        "device": device,
        "dtype": args.dtype,
        "compile": args.compile,
    }
    if device.startswith("cuda"):
        result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PhaseSSM training-step throughput.")
    parser.add_argument("--ssm-backend", choices=["fft", "auto", "real_chunked", "fixed_triton", "skip"], default="fft")
    parser.add_argument("--ssm-chunk", type=int, default=64)
    parser.add_argument("--ssm-auto-threshold", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-ff-mult", type=int, default=3)
    parser.add_argument("--short-conv", type=int, default=4)
    parser.add_argument("--no-mixer", action="store_true", help="Ablation: remove the temporal mixer block.")
    parser.add_argument("--no-ffn", action="store_true", help="Ablation: remove the feed-forward block.")
    parser.add_argument("--no-gate", action="store_true", help="Ablation: remove the mixer gate projection.")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = measure(args)
    print(json.dumps(result, indent=2), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
