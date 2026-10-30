"""Shared training core for PhaseSSM and the matched-transformer bake-off.

Both ``phase_ssm.train`` (single-model runs) and ``phase_ssm.bakeoff`` (the
side-by-side comparison) run the *same* training loop on the *same* byte data
with the *same* optimizer/LR schedule. The only thing that differs is the model.
Keeping the loop here means the bake-off is apples-to-apples by construction,
not by careful manual alignment of two copies.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .data import ByteData


def lr_at(step: int, lr: float, min_lr: float, warmup: int, steps: int) -> float:
    """Linear warmup then cosine decay to ``min_lr``."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= steps:
        return min_lr
    ratio = (step - warmup) / max(1, steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def train_model(
    model: torch.nn.Module,
    data: ByteData,
    *,
    name: str,
    out: Path,
    steps: int,
    batch: int,
    seq: int,
    lr: float,
    min_lr: float = 1e-4,
    warmup: int = 300,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    eval_interval: int = 500,
    eval_iters: int = 50,
    train_log_interval: int = 0,
    skip_initial_eval: bool = False,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    compile: bool = False,
    model_cfg: dict | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Train one model on ``data`` and return its training log.

    The loop is identical for every architecture: AdamW with separate weight-decay
    groups for matrices vs vectors, cosine LR, bf16 autocast, bpc eval on the val
    split, ``best.pt`` checkpointing. ``model_cfg`` is stored in the checkpoint so
    ``generate``/``chat`` can rebuild the architecture exactly.
    """
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    n_params = model.num_params()
    if compile:
        model = torch.compile(model)
    print(f"[{name}] params={n_params:,}", flush=True)

    decay_params, no_decay_params = [], []
    for pn, pp in model.named_parameters():
        (decay_params if pp.dim() >= 2 else no_decay_params).append(pp)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )

    log: dict[str, Any] = {
        "name": name, "params": n_params, "cfg": model_cfg,
        "lr": lr, "min_lr": min_lr, "warmup": warmup, "steps": steps,
        "batch": batch, "seq": seq, "history": [],
    }
    best_val = float("inf")
    t0 = time.time()
    tok_seen = 0
    run_t = time.time()
    train_log_tokens = 0
    train_log_t = time.time()

    use_amp = amp_dtype == torch.bfloat16
    autocast_dev = device.split(":")[0]

    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr, min_lr, warmup, steps)

        should_eval = (
            eval_interval > 0
            and step % eval_interval == 0
            and not (step == 0 and skip_initial_eval)
        )
        if should_eval:
            eval_t0 = time.time()
            val_bpc = data.eval_bpc(model, "val", batch, seq, device, eval_iters)
            train_bpc = data.eval_bpc(model, "train", batch, seq, device, max(10, eval_iters // 5))
            eval_s = time.time() - eval_t0
            dt = time.time() - run_t
            tps = tok_seen / dt if dt > 0 else 0
            print(f"[{name}] step {step:6d}  train_bpc {train_bpc:.4f}  val_bpc {val_bpc:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  interval_tok/s {tps/1e3:.0f}k  "
                  f"eval_s {eval_s:.1f}  elapsed_s {time.time()-t0:.0f}", flush=True)
            log["history"].append({"step": step, "train_bpc": train_bpc, "val_bpc": val_bpc,
                                   "lr": opt.param_groups[0]["lr"], "tok_per_s": tps,
                                   "eval_s": eval_s, "elapsed_s": time.time() - t0})
            (out / "log.json").write_text(json.dumps(log, indent=2))
            if val_bpc < best_val:
                best_val = val_bpc
                torch.save(
                    {"model_state": getattr(model, "_orig_mod", model).state_dict(),
                     "cfg": model_cfg, "model_type": name, "step": step, "val_bpc": val_bpc},
                    out / "best.pt",
                )
            tok_seen = 0
            run_t = time.time()

        if step == steps:
            break

        model.train()
        opt.zero_grad(set_to_none=True)
        train_step_t0 = time.time()
        loss = torch.tensor(0.0, device=device)
        for _ in range(grad_accum):
            x, y = data.get_batch("train", batch, seq, device)
            with torch.autocast(device_type=autocast_dev, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            tok_seen += x.numel()
            train_log_tokens += x.numel()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        train_step_s = time.time() - train_step_t0

        if train_log_interval > 0 and (step + 1) % train_log_interval == 0:
            train_log_dt = time.time() - train_log_t
            train_tps = train_log_tokens / train_log_dt if train_log_dt > 0 else 0
            print(f"[{name}] train_step {step + 1:6d}  loss {loss.item() * grad_accum:.4f}  "
                  f"train_tok/s {train_tps/1e3:.1f}k  step_s {train_step_s:.2f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  elapsed_s {time.time()-t0:.0f}", flush=True)
            log.setdefault("train_history", []).append({
                "step": step + 1,
                "loss": loss.item() * grad_accum,
                "train_tok_per_s": train_tps,
                "step_s": train_step_s,
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": time.time() - t0,
            })
            (out / "log.json").write_text(json.dumps(log, indent=2))
            train_log_tokens = 0
            train_log_t = time.time()

    log["best_val_bpc"] = best_val
    log["final_val_bpc"] = (log["history"][-1]["val_bpc"] if log["history"] else best_val)
    log["total_s"] = time.time() - t0
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[{name}] DONE  best_val_bpc={best_val:.4f}  params={n_params:,}  "
          f"total={time.time()-t0:.0f}s", flush=True)
    return log
