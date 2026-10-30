from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench import compare, test_adaptive, test_context


REFERENCE = {
    "name": "Llama-3-8B INT4 reference anchor",
    "dense_flops_per_token": 16_000_000_000,
    "disk_bytes": 5_000_000_000,
    "kv_cache_bytes_at_32k": 8_000_000_000,
    "kv_context_tokens": 32_768,
    "easy_accuracy_anchor": 0.99,
    "hard_accuracy_anchor": 0.92,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase-mesh frontier-comparison harness.")
    parser.add_argument("--out", type=Path, default=Path("benchmark/frontier_comparison/out"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.25)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--easy-budget", type=int, default=40)
    parser.add_argument("--hard-budget", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--flops-per-cell-step", type=float, default=52.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out
    bench_dir = out_dir / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_before = peak_rss_bytes()
    context = test_context.run(
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        backend=args.backend,
        pin_strength=args.pin_strength,
        residual_carry=args.residual_carry,
        out=bench_dir,
    )
    adaptive = test_adaptive.run(
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        backend=args.backend,
        pin_strength=args.pin_strength,
        residual_carry=args.residual_carry,
        easy_budget=args.easy_budget,
        hard_budget=args.hard_budget,
        temperature=args.temperature,
        out=bench_dir,
    )
    footprint = compare.run(
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        backend=args.backend,
        pin_strength=args.pin_strength,
        residual_carry=args.residual_carry,
        out=bench_dir,
    )
    peak_after = peak_rss_bytes()

    easy = query_resource_summary(adaptive["easy"], args.size, args.flops_per_cell_step)
    hard = query_resource_summary(adaptive["hard"], args.size, args.flops_per_cell_step)
    context_curve = ram_context_curve(
        mesh_bytes=max(footprint["state_q8_bytes"], footprint["state_full_bytes"]),
        context_tokens=[1024, 4096, 8192, 16384, 32768],
    )
    plots = write_plots(out_dir, easy, hard, context_curve)

    payload = {
        "experiment": "frontier_comparison",
        "config": {
            "size": args.size,
            "steps": args.steps,
            "seed": args.seed,
            "backend": args.backend,
            "pin_strength": args.pin_strength,
            "residual_carry": args.residual_carry,
            "easy_budget": args.easy_budget,
            "hard_budget": args.hard_budget,
            "temperature": args.temperature,
            "flops_per_cell_step": args.flops_per_cell_step,
        },
        "context_retention": {
            "gradient": context["gradient"],
            "target_gradient": context["target_gradient"],
            "passed": context["passed"],
        },
        "adaptive_compute": {
            "easy": easy,
            "hard": hard,
            "hard_to_easy_steps": adaptive["adaptive_speedup"]["hard_to_easy_ratio"],
            "prediction_accuracy_proxy": adaptive["prediction_accuracy"],
            "basin_persistence": adaptive["basin_persistence"],
            "passed": adaptive["passed"],
        },
        "footprint": {
            "state_full_bytes": footprint["state_full_bytes"],
            "state_q8_bytes": footprint["state_q8_bytes"],
            "compression_ratio": footprint["compression_ratio"],
            "process_peak_rss_before_bytes": peak_before,
            "process_peak_rss_after_bytes": peak_after,
        },
        "reference_anchor": REFERENCE,
        "comparison_table": comparison_table(context, adaptive, footprint, easy, hard),
        "ram_context_curve": context_curve,
        "plots": plots,
        "notes": [
            "Prediction accuracy proxy is 1 - mean_prediction_error, not task answer accuracy.",
            "Reference Llama values are anchors from the comparison brief, not local inference measurements.",
            "FLOPs are transparent estimates for field updates, not hardware-profiler measurements.",
        ],
    }
    payload["passed"] = bool(context["passed"] and adaptive["passed"])
    metrics_path = out_dir / "frontier_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["output_path"] = str(metrics_path)
    return payload


def query_resource_summary(run: dict[str, Any], size: int, flops_per_cell_step: float) -> dict[str, Any]:
    steps_used = int(run["steps_used"])
    prompt_tokens = max(1, len(run["prompt"].split()))
    estimated_flops = steps_used * size * size * flops_per_cell_step
    mean_error = float(run["mean_prediction_error"])
    return {
        "steps_used": steps_used,
        "prompt_tokens": prompt_tokens,
        "estimated_flops_per_query": estimated_flops,
        "estimated_flops_per_token": estimated_flops / prompt_tokens,
        "mean_prediction_error": mean_error,
        "prediction_accuracy_proxy": max(0.0, 1.0 - mean_error),
        "verifier_passed": bool(run["verifier"]["passed"]),
        "verifier_checker": run["verifier"]["checker"],
        "exhausted": bool(run["exhausted"]),
    }


def ram_context_curve(mesh_bytes: int, context_tokens: list[int]) -> list[dict[str, Any]]:
    llama_per_token = REFERENCE["kv_cache_bytes_at_32k"] / REFERENCE["kv_context_tokens"]
    return [
        {
            "context_tokens": tokens,
            "phase_mesh_bytes": mesh_bytes,
            "llama3_8b_int4_kv_bytes": int(llama_per_token * tokens),
        }
        for tokens in context_tokens
    ]


def comparison_table(
    context: dict[str, Any],
    adaptive: dict[str, Any],
    footprint: dict[str, Any],
    easy: dict[str, Any],
    hard: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "compute_scaling",
            "phase_mesh": f"{adaptive['adaptive_speedup']['hard_to_easy_ratio']:.1f}x hard/easy steps",
            "reference_anchor": "fixed dense forward, approx 1.0x hard/easy",
            "interpretation": "native adaptive test-time budget",
        },
        {
            "metric": "context_retention",
            "phase_mesh": f"gradient {context['gradient']:.4f}",
            "reference_anchor": "KV cache preserves tokens but grows with context",
            "interpretation": "phase pinning holds salient context under diffusion",
        },
        {
            "metric": "state_footprint",
            "phase_mesh": f"Q8 state {format_bytes(footprint['state_q8_bytes'])}",
            "reference_anchor": f"INT4 weights {format_bytes(REFERENCE['disk_bytes'])}",
            "interpretation": "local topology state is tiny in this prototype",
        },
        {
            "metric": "prediction_accuracy_proxy",
            "phase_mesh": f"{easy['prediction_accuracy_proxy']:.3f} easy / {hard['prediction_accuracy_proxy']:.3f} hard",
            "reference_anchor": f"{REFERENCE['easy_accuracy_anchor']:.2f} easy / {REFERENCE['hard_accuracy_anchor']:.2f} hard",
            "interpretation": "proxy only; answer accuracy still needs external evals",
        },
    ]


def write_plots(
    out_dir: Path,
    easy: dict[str, Any],
    hard: dict[str, Any],
    context_curve: list[dict[str, Any]],
) -> dict[str, str]:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = plots_dir / "accuracy_vs_compute.png"
    plt.figure(figsize=(6, 4))
    plt.loglog(
        [easy["estimated_flops_per_query"], hard["estimated_flops_per_query"]],
        [easy["prediction_accuracy_proxy"], hard["prediction_accuracy_proxy"]],
        marker="o",
        label="Phase mesh proxy",
    )
    plt.loglog(
        [REFERENCE["dense_flops_per_token"], REFERENCE["dense_flops_per_token"]],
        [REFERENCE["easy_accuracy_anchor"], REFERENCE["hard_accuracy_anchor"]],
        marker="s",
        label="Dense LLM anchor",
    )
    plt.xlabel("Estimated FLOPs/query or dense FLOPs/token")
    plt.ylabel("Accuracy / prediction proxy")
    plt.ylim(0.85, 1.01)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=180)
    plt.close()

    ram_path = plots_dir / "ram_vs_context.png"
    contexts = [row["context_tokens"] for row in context_curve]
    phase_ram = [row["phase_mesh_bytes"] for row in context_curve]
    llama_ram = [row["llama3_8b_int4_kv_bytes"] for row in context_curve]
    plt.figure(figsize=(6, 4))
    plt.plot(contexts, [value / 1_000_000 for value in phase_ram], marker="o", label="Phase mesh state")
    plt.plot(contexts, [value / 1_000_000 for value in llama_ram], marker="s", label="Dense LLM KV anchor")
    plt.xlabel("Context tokens")
    plt.ylabel("RAM / state MB")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ram_path, dpi=180)
    plt.close()

    hist_path = plots_dir / "steps_used_histogram.png"
    plt.figure(figsize=(6, 4))
    plt.bar(["easy", "hard"], [easy["steps_used"], hard["steps_used"]], color=["#2474a6", "#ba4a3a"])
    plt.ylabel("Steps used")
    plt.title("Adaptive budget allocation")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()

    return {
        "accuracy_vs_compute": str(accuracy_path),
        "ram_vs_context": str(ram_path),
        "steps_used_histogram": str(hist_path),
    }


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def format_bytes(value: int | float) -> str:
    value = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while value >= 1000 and index < len(units) - 1:
        value /= 1000
        index += 1
    if index == 0:
        return f"{value:.0f} {units[index]}"
    return f"{value:.2f} {units[index]}"


if __name__ == "__main__":
    raise SystemExit(main())
