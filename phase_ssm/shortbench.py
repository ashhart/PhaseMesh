"""Short-context PhaseSSM backend sweep.

This bench exists for the below-64k gap. It compares the parallel FFT prefill
path against the recurrent Triton path and the hybrid auto route, then writes a
threshold recommendation from the measured rows. If FFT never beats recurrent
Triton, the recommended threshold is ``None`` and ``auto`` should stay on
Triton for speed benches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .effbench import AttnStack, SSMStack, measure
from .triton_scan import triton_status


def recommend_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_length: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_length.setdefault(int(row["length"]), {})[str(row["backend"])] = row

    fft_wins: list[int] = []
    best_rows: list[dict[str, Any]] = []
    first_ssm_win: int | None = None
    for length in sorted(by_length):
        group = by_length[length]
        measured = [row for row in group.values() if row.get("ssm_tok_s")]
        if not measured:
            continue
        best = max(measured, key=lambda item: float(item["ssm_tok_s"]))
        best_rows.append({
            "length": length,
            "best_backend": best["backend"],
            "best_ssm_tok_s": best["ssm_tok_s"],
            "attn_tok_s": best.get("attn_tok_s"),
            "best_ratio": best.get("ratio"),
        })
        if first_ssm_win is None and best.get("ratio") is not None and float(best["ratio"]) >= 1.0:
            first_ssm_win = length

        fft = group.get("fft")
        triton = group.get("triton")
        if fft and triton and fft.get("ssm_tok_s") and triton.get("ssm_tok_s"):
            if float(fft["ssm_tok_s"]) >= float(triton["ssm_tok_s"]):
                fft_wins.append(length)

    threshold = max(fft_wins) if fft_wins else None
    return {
        "recommended_auto_threshold": threshold,
        "first_best_ssm_win_vs_attention": first_ssm_win,
        "best_by_length": best_rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PhaseSSM Short-Context Backend Sweep",
        "",
        "This sweep measures the below-64k gap: FFT short prefill, recurrent Triton, hybrid auto, and flash attention.",
        "",
        f"- `recommended_auto_threshold`: `{payload['recommendation']['recommended_auto_threshold']}`",
        f"- `first_best_ssm_win_vs_attention`: `{payload['recommendation']['first_best_ssm_win_vs_attention']}`",
        "",
        "| Length | Backend | SSM tok/s | Attention tok/s | SSM GB | Attention GB | Ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {length} | `{backend}` | {ssm_tok_s} | {attn_tok_s} | {ssm_mem_gb} | {attn_mem_gb} | {ratio} |".format(
                length=row["length"],
                backend=row["backend"],
                ssm_tok_s=_fmt_tok(row.get("ssm_tok_s")),
                attn_tok_s=_fmt_tok(row.get("attn_tok_s")),
                ssm_mem_gb=_fmt_float(row.get("ssm_mem_gb")),
                attn_mem_gb=_fmt_float(row.get("attn_mem_gb")),
                ratio=_fmt_ratio(row.get("ratio")),
            )
        )
    lines.extend([
        "",
        "Read: `fft` is the candidate short-prefill path. `triton` is the recurrent speed path. "
        "If `recommended_auto_threshold` is `None`, FFT did not win at any measured length and `auto` should stay on Triton.",
        "The result is a speed benchmark only; it does not measure language quality.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("phase_ssm.shortbench requires CUDA; run it on NVIDIA GB10 or another CUDA host.")
    ok, reason = triton_status()
    if not ok:
        raise SystemExit(f"Triton unavailable: {reason}")

    torch.manual_seed(args.seed)
    device = "cuda"
    backends = tuple(item.strip() for item in args.backends.split(",") if item.strip())
    lengths = list(args.lengths)
    rows: list[dict[str, Any]] = []

    for length in lengths:
        attn = AttnStack(args.d_model, args.heads, args.layers).to(device)
        attn_tok_s, attn_mem = measure(
            attn,
            length,
            device,
            args.d_model,
            batch=args.batch,
            reps=args.reps,
            dtype="bfloat16",
        )
        del attn
        torch.cuda.empty_cache()

        for backend in backends:
            ssm = SSMStack(
                args.d_model,
                args.state_dim,
                args.layers,
                chunk=args.chunk,
                backend=backend,
                state_block=args.state_block,
                auto_threshold=args.auto_threshold,
            ).to(device)
            try:
                ssm_tok_s, ssm_mem = measure(
                    ssm,
                    length,
                    device,
                    args.d_model,
                    batch=args.batch,
                    reps=args.reps,
                    dtype=args.ssm_dtype,
                )
                error = ""
            except RuntimeError as exc:
                ssm_tok_s, ssm_mem, error = None, None, str(exc)
                torch.cuda.empty_cache()
            used_backend = ssm.backend_for_length(length)
            del ssm
            torch.cuda.empty_cache()
            row = {
                "length": length,
                "backend": backend,
                "used_backend": used_backend,
                "ssm_tok_s": ssm_tok_s,
                "attn_tok_s": attn_tok_s,
                "ssm_mem_gb": ssm_mem,
                "attn_mem_gb": attn_mem,
                "ratio": None if ssm_tok_s is None else ssm_tok_s / attn_tok_s,
                "error": error,
            }
            rows.append(row)
            print(
                f"{length:8d} {backend:>8s}->{used_backend:<6s} "
                f"ssm={_fmt_tok(ssm_tok_s):>8s} attn={_fmt_tok(attn_tok_s):>8s} "
                f"ratio={_fmt_ratio(row['ratio']):>6s}",
                flush=True,
            )

    payload = {
        "status": "ok",
        "config": {
            "d_model": args.d_model,
            "layers": args.layers,
            "state_dim": args.state_dim,
            "heads": args.heads,
            "batch": args.batch,
            "reps": args.reps,
            "ssm_dtype": args.ssm_dtype,
            "auto_threshold": args.auto_threshold,
            "backends": backends,
            "lengths": lengths,
        },
        "rows": rows,
        "recommendation": recommend_threshold(rows),
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep PhaseSSM short-context backends against flash attention.")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--backends", default="fft,triton,auto")
    parser.add_argument("--auto-threshold", type=int, default=0, help="Use FFT at or below this length for backend=auto. 0 disables FFT routing.")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--state-block", type=int, default=16)
    parser.add_argument("--ssm-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/short_prefill_sweep.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/short_prefill_sweep.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "ok", "recommendation": payload["recommendation"]}, indent=2), flush=True)
    return 0


def _fmt_tok(value: Any) -> str:
    if value is None:
        return "OOM"
    return f"{float(value) / 1000:.0f}k"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "OOM"
    return f"{float(value):.2f}"


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}x"


if __name__ == "__main__":
    raise SystemExit(main())
