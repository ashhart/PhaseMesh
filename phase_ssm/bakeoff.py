"""Matched-parameter bake-off: PhaseSSM+memory vs transformer.

This is the decisive small-scale quality gate. It builds a PhaseSSM target
(parallel + linear phase-memory) and a transformer, matched as closely as
possible on parameters, then trains *both* on identical byte data with an
identical optimizer/LR schedule and total token budget. The output is one bpc
table side by side.

Apples-to-apples guarantees, enforced by construction:

* Same corpus (text8 bytes), same split, same vocab (256 bytes), same tokenizer
  (none — raw bytes). No tokenizer confound.
* Same optimizer (AdamW, wd groups, betas), same cosine LR schedule, same
  warmup, same grad-clip.
* Same total training tokens (``batch * seq * steps``), which is the FLOP proxy.
  Steps are adjusted per model so the token budget matches exactly.
* Parameters are matched to within ``--param-tol`` (default 5%). The harness
  reports both param counts and refuses to write a verdict if they drift past
  the tolerance — a bake-off where one side has 2x the params is meaningless.
* The transformer uses RoPE (modern default), not learned position embeddings.

It does not claim scaling parity. A small-model bpc tie is necessary but not
sufficient for the post-transformer thesis; see docs/TARGET_ARCH.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import ByteData, get_text8, stats
from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM
from .train_core import train_model


def build_ssm(args, vocab_size: int) -> PhaseSSMLM:
    cfg = PhaseSSMConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        state_dim=args.state_dim, expand=args.expand, d_ff_mult=args.d_ff_mult,
        short_conv=args.short_conv, ssm_backend="fft",
        dropout=args.dropout, use_phase_memory=True, mem_heads=args.mem_heads,
        mem_head_dim=args.mem_head_dim, mem_linear=True, mem_chunk=args.mem_chunk,
        mem_layout=args.mem_layout,
    )
    return PhaseSSMLM(cfg), cfg


def build_transformer(args, vocab_size: int) -> TransformerLM:
    cfg = TransformerConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.xf_layers,
        n_heads=args.xf_heads, d_ff_mult=args.xf_d_ff_mult, block_size=args.seq,
        dropout=args.dropout, pos_enc="rope",
    )
    return TransformerLM(cfg), cfg


def _params(model: torch.nn.Module) -> int:
    return model.num_params()


def matched_steps(args, ssm_params: int, xf_params: int) -> tuple[int, int]:
    """Same total training tokens for both. Steps scaled so batch*seq*steps is equal.

    If params differ, the model with fewer params gets slightly more steps would
    be a *compute* match; a *FLOP* match would give the bigger model fewer steps.
    We hold total *tokens* equal (the standard small-scale convention) and let
    per-token FLOP differ — that is reported, not silently absorbed.
    """
    tokens = args.tokens
    per_step = args.batch * args.seq
    steps = max(1, tokens // per_step)
    return steps, steps


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# PhaseSSM vs Transformer — Matched Bake-off",
        "",
        f"Date: {payload['date']}",
        f"Corpus: text8 bytes ({payload['train_bytes']:,} train / "
        f"{payload['val_bytes']:,} val)",
        "",
        "## Match",
        "",
        "| Model | Params | Total tokens | Steps | val_bpc (best) | val_bpc (final) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["models"]:
        lines.append(
            f"| {row['name']} | {row['params']:,} | {row['tokens']:,} | {row['steps']} | "
            f"{row['best_val_bpc']:.4f} | {row['final_val_bpc']:.4f} |"
        )
    # Verdict.
    m = {r["name"]: r for r in payload["models"]}
    ssm = m.get("phasessm")
    xf = m.get("transformer")
    lines += ["", "## Verdict", ""]
    if ssm and xf:
        param_ratio = max(ssm["params"], xf["params"]) / max(1, min(ssm["params"], xf["params"]))
        delta = ssm["best_val_bpc"] - xf["best_val_bpc"]
        lines.append(f"- Param ratio (max/min): {param_ratio:.3f} "
                     f"(tolerance {payload['param_tol']:.2f})")
        lines.append(f"- bpc delta (ssm - xf): {delta:+.4f}  "
                     f"({'SSM better' if delta < 0 else 'transformer better'})")
        if param_ratio > payload["param_tol"]:
            lines.append(f"- **NO VERDICT**: param counts differ by more than the "
                         f"{payload['param_tol']:.0%} tolerance. Re-tune model widths.")
        else:
            if abs(delta) < payload["bpc_deadband"]:
                lines.append(f"- Result: **TIE** within ±{payload['bpc_deadband']:.3f} bpc deadband. "
                             "Quality parity at matched compute at this scale.")
            elif delta < 0:
                lines.append("- Result: **SSM wins** on bpc at matched params/tokens.")
            else:
                lines.append("- Result: **Transformer wins** on bpc at matched params/tokens.")
    lines += [
        "",
        "## Read",
        "",
        "This is the small-scale quality gate only. A tie or win here is necessary "
        "but not sufficient for the post-transformer thesis — scaling behavior is "
        "settled at ~1B. Per-token FLOP is *not* matched here; only total tokens. "
        "See `docs/TARGET_ARCH.md` for the full claim boundary.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev = args.device
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    data = ByteData(get_text8(args.data_root))
    print(f"[data] {stats(data)}", flush=True)
    vocab = data.vocab_size

    ssm, ssm_cfg = build_ssm(args, vocab)
    xf, xf_cfg = build_transformer(args, vocab)
    ssm_params = _params(ssm)
    xf_params = _params(xf)
    print(f"[match] phasessm params={ssm_params:,}  transformer params={xf_params:,}", flush=True)

    param_ratio = max(ssm_params, xf_params) / max(1, min(ssm_params, xf_params))
    if param_ratio > args.param_tol:
        print(f"[match] WARNING: param ratio {param_ratio:.3f} exceeds tolerance "
              f"{args.param_tol:.2f}. The harness will run both but refuse a verdict. "
              "Adjust --d-model / --xf-layers / --xf-d-ff-mult to match params.", flush=True)

    steps_ssm, steps_xf = matched_steps(args, ssm_params, xf_params)
    total_tokens = args.batch * args.seq * steps_ssm

    models_payload: list[dict[str, Any]] = []

    for name, model, cfg, steps in [("phasessm", ssm, ssm_cfg.__dict__, steps_ssm),
                                    ("transformer", xf, xf_cfg.__dict__, steps_xf)]:
        run_dir = Path(args.out) / name
        lr = args.lr if name == "phasessm" else args.xf_lr
        log = train_model(
            model, data, name=name, out=run_dir, steps=steps, batch=args.batch,
            seq=args.seq, lr=lr, min_lr=args.min_lr, warmup=args.warmup,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            grad_accum=args.grad_accum, eval_interval=args.eval_interval,
            eval_iters=args.eval_iters, train_log_interval=args.train_log_interval,
            skip_initial_eval=args.skip_initial_eval, device=dev, amp_dtype=amp_dtype,
            compile=args.compile, model_cfg=cfg, seed=args.seed,
        )
        models_payload.append({
            "name": name,
            "params": log["params"],
            "tokens": total_tokens,
            "steps": steps,
            "best_val_bpc": log["best_val_bpc"],
            "final_val_bpc": log["final_val_bpc"],
            "total_s": log["total_s"],
        })

    payload = {
        "status": "ok",
        "date": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "corpus": "text8-bytes",
        "train_bytes": len(data.train),
        "val_bytes": len(data.val),
        "batch": args.batch,
        "seq": args.seq,
        "tokens": total_tokens,
        "param_tol": args.param_tol,
        "bpc_deadband": args.bpc_deadband,
        "models": models_payload,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched PhaseSSM vs transformer bpc bake-off.")
    p.add_argument("--out", type=str, required=True, help="Output dir (writes per-model logs + report).")
    p.add_argument("--data-root", type=str, default="data")
    # Shared match budget.
    p.add_argument("--tokens", type=int, default=512_000_000,
                   help="Total training tokens per model (FLOP proxy). Steps derived from this.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=1024)
    # PhaseSSM architecture.
    p.add_argument("--d-model", dest="d_model", type=int, default=512)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=8)
    p.add_argument("--state-dim", dest="state_dim", type=int, default=64)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-ff-mult", dest="d_ff_mult", type=int, default=2)
    p.add_argument("--short-conv", dest="short_conv", type=int, default=4)
    p.add_argument("--mem-heads", dest="mem_heads", type=int, default=4)
    p.add_argument("--mem-head-dim", dest="mem_head_dim", type=int, default=32)
    p.add_argument("--mem-chunk", dest="mem_chunk", type=int, default=64)
    p.add_argument("--mem-layout", dest="mem_layout", choices=["parallel", "sequential"], default="parallel")
    # Transformer architecture (separate knobs so params can be matched independently).
    p.add_argument("--xf-layers", dest="xf_layers", type=int, default=8)
    p.add_argument("--xf-heads", dest="xf_heads", type=int, default=8)
    p.add_argument("--xf-d-ff-mult", dest="xf_d_ff_mult", type=int, default=3)
    # Optimization (shared schedule, per-model peak LR since the architectures differ).
    p.add_argument("--lr", type=float, default=3e-3, help="PhaseSSM peak LR.")
    p.add_argument("--xf-lr", dest="xf_lr", type=float, default=3e-4, help="Transformer peak LR.")
    p.add_argument("--min-lr", dest="min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--eval-interval", dest="eval_interval", type=int, default=500)
    p.add_argument("--eval-iters", dest="eval_iters", type=int, default=50)
    p.add_argument("--train-log-interval", dest="train_log_interval", type=int, default=0)
    p.add_argument("--skip-initial-eval", dest="skip_initial_eval", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    # Verdict control.
    p.add_argument("--param-tol", dest="param_tol", type=float, default=1.05,
                   help="Refuse a verdict if param ratio (max/min) exceeds this.")
    p.add_argument("--bpc-deadband", dest="bpc_deadband", type=float, default=0.02,
                   help="bpc difference below this is reported as a TIE.")
    # Runtime.
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--report", type=str, default=None,
                   help="Path for the markdown report (default: <out>/report.md).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    (out / "bakeoff.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = Path(args.report) if args.report else out / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out),
                      "report": str(report_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
