from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import MeshConfig
from .runtime import CognitiveMeshRuntime
from .viz import save_phase_image


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve(args)
    if args.command == "bench":
        return bench(args)
    if args.command == "model-train":
        return model_train(args)
    if args.command == "model-eval":
        return model_eval(args)
    if args.command == "generate":
        return generate(args)
    if args.command == "probe-arithmetic":
        return probe_arithmetic(args)
    if args.command == "probe-arithmetic-result":
        return probe_arithmetic_result(args)
    if args.command == "fit-arithmetic-readout":
        return fit_arithmetic_readout(args)
    if args.command == "solve-arithmetic":
        return solve_arithmetic(args)

    config = MeshConfig(
        width=args.size,
        height=args.size,
        max_steps=args.steps,
        seed=args.seed,
        laplacian_backend=args.backend,
        phase_pin_strength=args.pin_strength,
        phase_residual_carry=args.residual_carry,
    )
    runtime = CognitiveMeshRuntime(config)

    if args.command == "demo":
        return demo(runtime, args)
    if args.command == "run":
        return run_once(runtime, args)
    if args.command == "think":
        return think(runtime, args)
    if args.command == "route":
        return route(runtime, args)
    if args.command == "learn":
        return learn(runtime, args)

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a laptop-scale phase-field cognitive mesh.")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--size", type=int, default=128, help="Square mesh size.")
        command.add_argument("--steps", type=int, default=320, help="Maximum evolution steps.")
        command.add_argument("--seed", type=int, default=7, help="Seed for the natural landscape.")
        command.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"], help="Laplacian backend.")
        command.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0, help="Phase pinning strength for salient packets.")
        command.add_argument("--residual-carry", type=float, default=0.08, help="Previous-phase carry blended into each step.")
        command.add_argument("--out", type=Path, default=None, help="Output directory for image/state files.")

    demo_parser = subparsers.add_parser("demo", help="Run a built-in demo.")
    add_common(demo_parser)

    run_parser = subparsers.add_parser("run", help="Inject text and decode resonance.")
    add_common(run_parser)
    run_parser.add_argument("text", nargs="+", help="Prompt text.")
    run_parser.add_argument("--expect", default=None, help="Expected value for verifier feedback.")
    run_parser.add_argument("--learn", action="store_true", help="Apply one feedback update after resonance.")

    think_parser = subparsers.add_parser("think", help="Run predictive adaptive compute.")
    add_common(think_parser)
    think_parser.add_argument("text", nargs="+", help="Prompt text.")
    think_parser.add_argument("--expect", default=None, help="Expected value for verifier feedback.")
    think_parser.add_argument("--learn", action="store_true", help="Apply feedback after adaptive thinking.")
    think_parser.add_argument("--max-budget", type=int, default=200, help="Maximum adaptive compute steps.")
    think_parser.add_argument("--min-steps", type=int, default=None, help="Minimum adaptive compute steps.")
    think_parser.add_argument("--temperature", type=float, default=0.0, help="Uncertainty-scaled phase noise.")
    think_parser.add_argument("--verifier-control", action="store_true", help="Spend remaining budget when a resonant state fails verification.")

    route_parser = subparsers.add_parser("route", help="Return a compact terminal/tool dispatch decision.")
    add_common(route_parser)
    route_parser.add_argument("text", nargs="+", help="Prompt text.")
    route_parser.add_argument("--expect", default=None, help="Expected value for verifier feedback.")

    learn_parser = subparsers.add_parser("learn", help="Run repeated verifier feedback rounds.")
    add_common(learn_parser)
    learn_parser.add_argument("text", nargs="+", help="Prompt text.")
    learn_parser.add_argument("--expect", default=None, help="Expected value for verifier feedback.")
    learn_parser.add_argument("--rounds", type=int, default=4, help="Learning rounds.")

    train_parser = subparsers.add_parser("model-train", help="Train the experimental basin-to-token model layer.")
    train_parser.add_argument("data", type=Path, help="UTF-8 text file with one training chunk per line.")
    train_parser.add_argument("--out", type=Path, default=Path("runs/phase-model"), help="Output directory.")
    train_parser.add_argument("--load-model-dir", type=Path, default=None, help="Warm-start from a saved PhaseModel directory.")
    train_parser.add_argument("--chunks", type=int, default=None, help="Maximum chunks to train.")
    train_parser.add_argument("--max-steps", dest="chunks", type=int, help="Alias for --chunks.")
    train_parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint interval in chunks.")
    train_parser.add_argument("--steps-per-chunk", type=int, default=20, help="Field steps per text chunk.")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Decoder-head batch size.")
    train_parser.add_argument(
        "--mode",
        choices=[
            "next-token",
            "contrastive",
            "ranking",
            "structural",
            "structural-anchor",
            "structural-repulsion",
            "computational-distillation",
            "guided-evolution",
            "phase-geometry",
            "delta-geometry",
            "delta-geometry-frozen",
            "residual-tunnel",
            "push-pull",
            "prototype-decoder",
        ],
        default="next-token",
        help="Training objective. `ranking` trains candidate verification; `structural-anchor` hard-collapses equivalent basins.",
    )
    train_parser.add_argument("--context-tokens", type=int, default=8, help="Context tokens before each next-token target.")
    train_parser.add_argument("--windows-per-chunk", type=int, default=4, help="Maximum next-token windows sampled per chunk.")
    train_parser.add_argument("--window-stride", type=int, default=1, help="Stride between candidate next-token windows.")
    train_parser.add_argument(
        "--ablation",
        choices=["full", "no-interference", "random-walk", "static-topology", "uniform-init"],
        default="full",
        help="Physics/topology ablation mode.",
    )
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Decoder-head learning rate.")
    train_parser.add_argument("--size", type=int, default=128, help="Square mesh size.")
    train_parser.add_argument("--seed", type=int, default=7, help="Seed for the natural landscape.")
    train_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"], help="Laplacian backend.")
    train_parser.add_argument("--encoder-mode", choices=["text", "structured"], default="text")
    train_parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    train_parser.add_argument("--structured-feature-strength", type=float, default=2.0)
    train_parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.25)
    train_parser.add_argument("--residual-carry", type=float, default=0.08)
    train_parser.add_argument("--vocab-capacity", type=int, default=4096)
    train_parser.add_argument("--basin-dim", type=int, default=256)
    train_parser.add_argument("--hidden", type=int, default=128)
    train_parser.add_argument("--no-decoder", action="store_true", help="Carve topology only; skip decoder-head training.")
    train_parser.add_argument("--no-topology", action="store_true", help="Train decoder only; skip basin reinforcement.")
    train_parser.add_argument("--freeze-omega", action="store_true", help="Restore omega after each decoder observation.")
    train_parser.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder during structural/anchor passes.")
    train_parser.add_argument("--unfreeze-decoder", action="store_true", help="Also sync decoder on positive ranking candidates.")
    train_parser.add_argument("--consolidate-interval", type=int, default=0, help="Run consolidation every N chunks.")
    train_parser.add_argument("--consolidate-cycles", type=int, default=8, help="Consolidation cycles when interval fires.")
    train_parser.add_argument("--structural-weight", type=float, default=0.5, help="Weight for structural alignment term.")
    train_parser.add_argument("--topology-gain", type=float, default=0.025, help="Gradient-free basin bridge gain for structural mode.")
    train_parser.add_argument("--prototype-alpha", type=float, default=0.10, help="EMA rate for structural-anchor prototypes.")
    train_parser.add_argument("--readout-temperature", type=float, default=0.1, help="Prototype readout inverse-distance softmax temperature.")
    train_parser.add_argument("--readout-direct-scale", type=float, default=8.0, help="Direct nearest-prototype target logit scale.")
    train_parser.add_argument("--result-attract-gain", type=float, default=0.10, help="Prototype feature pull toward the active result basin.")
    train_parser.add_argument("--repulsion-strength", type=float, default=0.40, help="Strength for pushing active basins away from wrong result prototypes.")
    train_parser.add_argument("--distill-gain", type=float, default=0.10, help="Computational teacher distillation gain.")
    train_parser.add_argument("--strength", type=float, default=None, help="Alias for computational-distillation prototype/landscape strength.")
    train_parser.add_argument("--teacher-result-weight", type=float, default=2.0, help="Weight of the correct result prototype inside teacher basins.")
    train_parser.add_argument("--coupling", type=float, default=0.30, help="Teacher coupling for guided phase evolution.")
    train_parser.add_argument("--guided-success-threshold", type=float, default=0.10, help="MSE threshold before guided paths are carved into topology.")
    train_parser.add_argument("--patch-size", type=int, default=None, help="Odd local patch size for phase-geometry teacher injection.")
    train_parser.add_argument("--geometry-strength", type=float, default=0.05, help="Persistent phase patch strength for phase-geometry mode.")
    train_parser.add_argument("--delta-success-distance", type=float, default=1.0, help="Target-distance threshold before delta paths are carved.")
    train_parser.add_argument("--tunnel-strength", type=float, default=0.05, help="Residual landscape tunnel strength.")
    train_parser.add_argument("--push-pull-strength", type=float, default=0.05, help="Local feature pull strength for push-pull mode.")
    train_parser.add_argument("--wrong-strength", type=float, default=0.5, help="Wrong-result push multiplier for push-pull mode.")

    eval_parser = subparsers.add_parser("model-eval", help="Evaluate decoder loss/perplexity on held-out text.")
    eval_parser.add_argument("data", type=Path, help="UTF-8 held-out text file with one chunk per line.")
    eval_parser.add_argument("--model-dir", type=Path, required=True, help="Directory produced by model-train.")
    eval_parser.add_argument("--chunks", type=int, default=None, help="Maximum chunks to evaluate.")
    eval_parser.add_argument("--steps-per-chunk", type=int, default=20)
    eval_parser.add_argument("--context-tokens", type=int, default=8)
    eval_parser.add_argument("--windows-per-chunk", type=int, default=4)
    eval_parser.add_argument("--window-stride", type=int, default=1)
    eval_parser.add_argument("--teacher-result-weight", type=float, default=2.0)
    eval_parser.add_argument(
        "--mode",
        choices=[
            "next-token",
            "ranking",
            "structural",
            "structural-anchor",
            "structural-repulsion",
            "computational-distillation",
            "guided-evolution",
            "phase-geometry",
            "delta-geometry",
            "delta-geometry-frozen",
            "residual-tunnel",
            "push-pull",
        ],
        default="next-token",
    )
    eval_parser.add_argument("--coupling", type=float, default=0.30)
    eval_parser.add_argument("--patch-size", type=int, default=None)

    generate_parser = subparsers.add_parser("generate", help="Generate with a trained experimental PhaseModel.")
    generate_parser.add_argument("text", nargs="+", help="Prompt text.")
    generate_parser.add_argument("--model-dir", type=Path, required=True, help="Directory produced by model-train.")
    generate_parser.add_argument("--max-tokens", "--max-len", dest="max_tokens", type=int, default=32)
    generate_parser.add_argument("--steps-per-token", type=int, default=15)
    generate_parser.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.8)
    generate_parser.add_argument("--temperature-decay", type=float, default=1.0)
    generate_parser.add_argument("--min-temperature", type=float, default=0.05)
    generate_parser.add_argument("--top-k", type=int, default=16)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Penalty for recently generated tokens; 1.0 disables it.")
    generate_parser.add_argument("--repeat-window", type=int, default=10, help="Number of recent generated tokens to penalize.")
    generate_parser.add_argument("--rerank", action="store_true", help="Pick a candidate with the trained basin verifier head.")
    generate_parser.add_argument("--rerank-k", type=int, default=5, help="Number of decoder candidates to rerank.")
    generate_parser.add_argument("--anneal", action="store_true", help="Use longer decaying-noise phase settling for math-like prompts.")
    generate_parser.add_argument("--anneal-steps", type=int, default=30, help="Minimum settling steps when --anneal is enabled.")
    generate_parser.add_argument(
        "--rerank-candidates",
        default=None,
        help="Comma-separated candidate override for verifier auditing.",
    )

    probe_parser = subparsers.add_parser("probe-arithmetic", help="Probe whether arithmetic factors survive in basin features.")
    probe_parser.add_argument("--encoder", choices=["text", "structured"], default="structured")
    probe_parser.add_argument("--compare-text", action="store_true", help="Run both text and structured encoders side by side.")
    probe_parser.add_argument("--max-value", type=int, default=20)
    probe_parser.add_argument("--min-value", type=int, default=0)
    probe_parser.add_argument("--ops", default="add,mul", help="Comma-separated operations: add,sub,mul.")
    probe_parser.add_argument("--size", type=int, default=64)
    probe_parser.add_argument("--basin-dim", type=int, default=128)
    probe_parser.add_argument("--hidden", type=int, default=64)
    probe_parser.add_argument("--steps-per-chunk", type=int, default=12)
    probe_parser.add_argument("--seed", type=int, default=7)
    probe_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    probe_parser.add_argument("--train-fraction", type=float, default=0.7)
    probe_parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    probe_parser.add_argument("--structured-feature-strength", type=float, default=2.0)
    probe_parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")

    result_probe_parser = subparsers.add_parser(
        "probe-arithmetic-result",
        help="Probe exact arithmetic readout from decoded basin factors.",
    )
    result_probe_parser.add_argument("--encoder", choices=["text", "structured"], default="structured")
    result_probe_parser.add_argument("--compare-text", action="store_true", help="Run both text and structured encoders side by side.")
    result_probe_parser.add_argument("--max-value", type=int, default=20)
    result_probe_parser.add_argument("--min-value", type=int, default=0)
    result_probe_parser.add_argument("--ops", default="add,mul", help="Comma-separated operations: add,sub,mul.")
    result_probe_parser.add_argument("--size", type=int, default=64)
    result_probe_parser.add_argument("--basin-dim", type=int, default=128)
    result_probe_parser.add_argument("--hidden", type=int, default=64)
    result_probe_parser.add_argument("--steps-per-chunk", type=int, default=12)
    result_probe_parser.add_argument("--seed", type=int, default=7)
    result_probe_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    result_probe_parser.add_argument("--train-fraction", type=float, default=0.7)
    result_probe_parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    result_probe_parser.add_argument("--structured-feature-strength", type=float, default=2.0)
    result_probe_parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")

    fit_readout_parser = subparsers.add_parser(
        "fit-arithmetic-readout",
        help="Fit and save a reusable structured arithmetic factor readout.",
    )
    fit_readout_parser.add_argument("--out", type=Path, default=Path("runs/arithmetic-readout"), help="Output directory.")
    fit_readout_parser.add_argument("--max-value", type=int, default=20)
    fit_readout_parser.add_argument("--min-value", type=int, default=0)
    fit_readout_parser.add_argument("--ops", default="add,mul", help="Comma-separated operations: add,sub,mul.")
    fit_readout_parser.add_argument("--size", type=int, default=64)
    fit_readout_parser.add_argument("--basin-dim", type=int, default=128)
    fit_readout_parser.add_argument("--hidden", type=int, default=64)
    fit_readout_parser.add_argument("--steps-per-chunk", type=int, default=12)
    fit_readout_parser.add_argument("--seed", type=int, default=7)
    fit_readout_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    fit_readout_parser.add_argument("--train-fraction", type=float, default=0.7)
    fit_readout_parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    fit_readout_parser.add_argument("--structured-feature-strength", type=float, default=2.0)

    solve_parser = subparsers.add_parser(
        "solve-arithmetic",
        help="Solve one arithmetic prompt through the structured basin factor readout.",
    )
    solve_parser.add_argument("text", nargs="+", help="Arithmetic prompt, for example: 8 plus 9")
    solve_parser.add_argument("--readout-dir", type=Path, default=None, help="Optional directory or readout.json produced by fit-arithmetic-readout.")
    solve_parser.add_argument("--max-value", type=int, default=20)
    solve_parser.add_argument("--min-value", type=int, default=0)
    solve_parser.add_argument("--ops", default="add,mul", help="Comma-separated operations: add,sub,mul.")
    solve_parser.add_argument("--size", type=int, default=64)
    solve_parser.add_argument("--basin-dim", type=int, default=128)
    solve_parser.add_argument("--hidden", type=int, default=64)
    solve_parser.add_argument("--steps-per-chunk", type=int, default=12)
    solve_parser.add_argument("--seed", type=int, default=7)
    solve_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    solve_parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    solve_parser.add_argument("--structured-feature-strength", type=float, default=2.0)
    solve_parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--size", type=int, default=128)
    serve_parser.add_argument("--steps", type=int, default=320)
    serve_parser.add_argument("--seed", type=int, default=7)
    serve_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    serve_parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.25)
    serve_parser.add_argument("--residual-carry", type=float, default=0.08)
    serve_parser.add_argument("--state-dir", type=Path, default=Path("runs/service-state"))
    serve_parser.add_argument("--no-persist", action="store_true")

    bench_parser = subparsers.add_parser("bench", help="Run the benchmark suite.")
    bench_parser.add_argument("--trials", type=int, default=50)
    bench_parser.add_argument("--facts", type=int, default=10)
    bench_parser.add_argument("--math-count", type=int, default=50)
    bench_parser.add_argument("--size", type=int, default=64)
    bench_parser.add_argument("--steps", type=int, default=180)
    bench_parser.add_argument("--seed", type=int, default=7)
    bench_parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    bench_parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    bench_parser.add_argument("--residual-carry", type=float, default=0.08)
    bench_parser.add_argument("--out", type=Path, default=Path("runs/bench"))

    return parser


def demo(runtime: CognitiveMeshRuntime, args: argparse.Namespace) -> int:
    prompts = [
        ("check 17 * 19 = 323 and route the result", None),
        ("def add(a, b):\n    return a + b", None),
        ("compress this local workflow into a stable reusable route", None),
    ]
    results: list[dict[str, Any]] = []
    for prompt, expected in prompts:
        run = runtime.resonate(prompt, expected=expected, learn=True)
        results.append(run.to_dict())
    runtime.mesh.consolidate(cycles=16)
    payload = {"demo": results, "final_metrics": runtime.mesh.metrics().to_dict()}
    emit(payload, runtime, args.out, "demo")
    return 0


def run_once(runtime: CognitiveMeshRuntime, args: argparse.Namespace) -> int:
    prompt = " ".join(args.text)
    run = runtime.resonate(prompt, expected=args.expect, learn=args.learn)
    emit(run.to_dict(), runtime, args.out, "run")
    return 0


def think(runtime: CognitiveMeshRuntime, args: argparse.Namespace) -> int:
    prompt = " ".join(args.text)
    run = runtime.think(
        prompt,
        max_budget=args.max_budget,
        min_steps=args.min_steps,
        temperature=args.temperature,
        expected=args.expect,
        learn=args.learn,
        verifier_control=args.verifier_control,
    )
    emit(run.to_dict(), runtime, args.out, "think")
    return 0


def route(runtime: CognitiveMeshRuntime, args: argparse.Namespace) -> int:
    prompt = " ".join(args.text)
    run = runtime.resonate(prompt, expected=args.expect, learn=False)
    verifier = run.verifier
    dispatch = {
        "route": run.decoded.route,
        "tool": tool_for(run.decoded.route, verifier.checker),
        "signature": run.decoded.signature,
        "confidence": run.decoded.confidence,
        "resonant": run.metrics.resonant,
        "verifier": verifier.to_dict(),
    }
    print(json.dumps(dispatch, indent=2))
    return 0


def learn(runtime: CognitiveMeshRuntime, args: argparse.Namespace) -> int:
    prompt = " ".join(args.text)
    payload = runtime.learn(prompt, expected=args.expect, rounds=args.rounds, steps=args.steps)
    emit(payload, runtime, args.out, "learn")
    return 0


def model_train(args: argparse.Namespace) -> int:
    try:
        from .model import PhaseModel
        from .trainer import (
            iter_ranking_jsonl,
            iter_repulsion_jsonl,
            iter_structural_jsonl,
            iter_text_file,
            stream_computational_distillation_train,
            stream_delta_geometry_train,
            stream_guided_evolution_train,
            stream_phase_geometry_train,
            stream_push_pull_train,
            stream_residual_tunnel_train,
            stream_ranking_train,
            stream_prototype_decoder_train,
            stream_structural_repulsion_train,
            stream_structural_train,
            stream_train,
        )
        if args.load_model_dir is not None:
            model = PhaseModel.load(args.load_model_dir, load_decoder=not args.no_decoder)
            if not args.no_decoder:
                model.reset_optimizer(args.lr)
        else:
            model = PhaseModel(
                grid_size=args.size,
                vocab_capacity=args.vocab_capacity,
                basin_dim=args.basin_dim,
                hidden=args.hidden,
                seed=args.seed,
                backend=args.backend,
                pin_strength=args.pin_strength,
                residual_carry=args.residual_carry,
                learning_rate=args.lr,
                encoder_mode=args.encoder_mode,
                structured_result_hint=args.structured_result_hint,
                structured_feature_strength=args.structured_feature_strength,
                create_decoder=not args.no_decoder,
            )
        if args.mode == "ranking":
            payload = stream_ranking_train(
                model,
                iter_ranking_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                batch_size=args.batch_size,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                train_decoder=args.unfreeze_decoder,
            )
        elif args.mode == "prototype-decoder":
            payload = stream_prototype_decoder_train(
                model,
                iter_structural_jsonl(args.data),
                batch_size=args.batch_size,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                readout_temperature=args.readout_temperature,
                readout_direct_scale=args.readout_direct_scale,
            )
        elif args.mode == "structural-repulsion":
            payload = stream_structural_repulsion_train(
                model,
                iter_repulsion_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                attract_gain=args.result_attract_gain,
                repulsion_strength=args.repulsion_strength,
                topology_gain=args.topology_gain,
            )
        elif args.mode == "computational-distillation":
            distill_strength = args.strength if args.strength is not None else args.distill_gain
            topology_strength = args.strength if args.strength is not None else args.topology_gain
            payload = stream_computational_distillation_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                distill_gain=distill_strength,
                repulsion_strength=args.repulsion_strength,
                topology_gain=topology_strength,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode == "guided-evolution":
            payload = stream_guided_evolution_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                coupling=args.coupling,
                success_mse=args.guided_success_threshold,
                distill_gain=args.distill_gain,
                repulsion_strength=args.repulsion_strength,
                topology_gain=args.topology_gain,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode == "phase-geometry":
            payload = stream_phase_geometry_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                coupling=args.coupling,
                success_mse=args.guided_success_threshold,
                geometry_strength=args.geometry_strength,
                patch_size=args.patch_size,
                distill_gain=args.distill_gain,
                repulsion_strength=args.repulsion_strength,
                topology_gain=args.topology_gain,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode in {"delta-geometry", "delta-geometry-frozen"}:
            payload = stream_delta_geometry_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                coupling=args.coupling,
                success_distance=args.delta_success_distance,
                geometry_strength=args.geometry_strength,
                patch_size=args.patch_size,
                topology_gain=args.topology_gain,
                result_weight=args.teacher_result_weight,
                freeze_targets=args.mode == "delta-geometry-frozen",
            )
        elif args.mode == "residual-tunnel":
            payload = stream_residual_tunnel_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                tunnel_strength=args.tunnel_strength,
            )
        elif args.mode == "push-pull":
            payload = stream_push_pull_train(
                model,
                iter_repulsion_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                push_pull_strength=args.push_pull_strength,
                wrong_strength=args.wrong_strength,
            )
        elif args.mode in {"structural", "structural-anchor"}:
            payload = stream_structural_train(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                batch_size=args.batch_size,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_rows=args.chunks,
                structural_weight=args.structural_weight,
                topology_gain=args.topology_gain,
                anchor=args.mode == "structural-anchor",
                freeze_decoder=args.freeze_decoder,
                prototype_alpha=args.prototype_alpha,
            )
        else:
            payload = stream_train(
                model,
                iter_text_file(args.data),
                steps_per_chunk=args.steps_per_chunk,
                batch_size=args.batch_size,
                context_tokens=args.context_tokens,
                windows_per_chunk=args.windows_per_chunk,
                window_stride=args.window_stride,
                save_interval=args.save_interval,
                out_dir=args.out,
                max_chunks=args.chunks,
                train_decoder=not args.no_decoder,
                train_topology=not args.no_topology,
                freeze_omega=args.freeze_omega,
                train_mode=args.mode,
                ablation_mode=args.ablation,
                consolidate_interval=args.consolidate_interval,
                consolidate_cycles=args.consolidate_cycles,
            )
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    print(json.dumps(payload, indent=2))
    return 0


def model_eval(args: argparse.Namespace) -> int:
    try:
        from .model import PhaseModel
        from .trainer import (
            iter_ranking_jsonl,
            iter_repulsion_jsonl,
            iter_text_file,
            stream_computational_distillation_evaluate,
            stream_delta_geometry_evaluate,
            stream_evaluate,
            stream_guided_evolution_evaluate,
            stream_phase_geometry_evaluate,
            stream_push_pull_evaluate,
            stream_residual_tunnel_evaluate,
            stream_ranking_evaluate,
            stream_ranking_group_evaluate,
            iter_structural_jsonl,
            stream_repulsion_evaluate,
            stream_structural_evaluate,
        )
        model = PhaseModel.load(args.model_dir)
        if args.mode == "ranking":
            payload = {
                "threshold": stream_ranking_evaluate(
                    model,
                    iter_ranking_jsonl(args.data),
                    steps_per_chunk=args.steps_per_chunk,
                    max_rows=args.chunks,
                ),
                "group_top1": stream_ranking_group_evaluate(
                    model,
                    iter_ranking_jsonl(args.data),
                    steps_per_chunk=args.steps_per_chunk,
                    max_groups=args.chunks,
                ),
            }
        elif args.mode in {"structural", "structural-anchor"}:
            payload = stream_structural_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
            )
        elif args.mode == "structural-repulsion":
            payload = stream_repulsion_evaluate(
                model,
                iter_repulsion_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
            )
        elif args.mode == "computational-distillation":
            payload = stream_computational_distillation_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode == "guided-evolution":
            payload = stream_guided_evolution_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
                coupling=args.coupling,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode == "phase-geometry":
            payload = stream_phase_geometry_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
                coupling=args.coupling,
                patch_size=args.patch_size,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode in {"delta-geometry", "delta-geometry-frozen"}:
            payload = stream_delta_geometry_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
                coupling=args.coupling,
                patch_size=args.patch_size,
                result_weight=args.teacher_result_weight,
            )
        elif args.mode == "residual-tunnel":
            payload = stream_residual_tunnel_evaluate(
                model,
                iter_structural_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
            )
        elif args.mode == "push-pull":
            payload = stream_push_pull_evaluate(
                model,
                iter_repulsion_jsonl(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_rows=args.chunks,
            )
        else:
            payload = stream_evaluate(
                model,
                iter_text_file(args.data),
                steps_per_chunk=args.steps_per_chunk,
                max_chunks=args.chunks,
                context_tokens=args.context_tokens,
                windows_per_chunk=args.windows_per_chunk,
                window_stride=args.window_stride,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    print(json.dumps(payload, indent=2))
    return 0


def generate(args: argparse.Namespace) -> int:
    try:
        from .model import PhaseModel
        model = PhaseModel.load(args.model_dir)
        if args.rerank:
            candidates = None
            if args.rerank_candidates:
                candidates = [item.strip() for item in args.rerank_candidates.split(",") if item.strip()]
            result = model.rerank(
                " ".join(args.text),
                candidates=candidates,
                k=args.rerank_k,
                steps_per_chunk=args.steps_per_token,
                anneal=args.anneal,
                anneal_steps=args.anneal_steps,
            )
            best = result.get("best") or {}
            payload = {
                "status": "ok",
                "text": best.get("candidate", ""),
                "best": best,
                "candidates": result.get("candidates", []),
                "note": "Experimental basin verifier rerank; use candidate overrides to audit arithmetic choices.",
            }
            print(json.dumps(payload, indent=2))
            return 0
        steps = model.generate_steps(
            " ".join(args.text),
            max_tokens=args.max_tokens,
            steps_per_token=args.steps_per_token,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature_decay=args.temperature_decay,
            min_temperature=args.min_temperature,
            repeat_penalty=args.repeat_penalty,
            repeat_window=args.repeat_window,
            anneal=args.anneal,
            anneal_steps=args.anneal_steps,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    tokens = [step.token for step in steps]
    payload = {
        "status": "ok",
        "text": " ".join(token for token in tokens if token not in {"<eos>", "<unk>"}),
        "tokens": tokens,
        "steps": [step.to_dict() for step in steps],
        "note": "Experimental basin decoder output; quality depends on the saved decoder/topology.",
    }
    print(json.dumps(payload, indent=2))
    return 0


def probe_arithmetic(args: argparse.Namespace) -> int:
    from .probes import run_arithmetic_representation_probe, save_probe_result

    ops = tuple(item.strip() for item in str(args.ops).split(",") if item.strip())
    encoder_modes = ("text", "structured") if args.compare_text else (args.encoder,)
    results = []
    for encoder_mode in encoder_modes:
        result = run_arithmetic_representation_probe(
            encoder_mode=encoder_mode,
            max_value=args.max_value,
            min_value=args.min_value,
            ops=ops,
            grid_size=args.size,
            basin_dim=args.basin_dim,
            hidden=args.hidden,
            steps_per_chunk=args.steps_per_chunk,
            seed=args.seed,
            backend=args.backend,
            train_fraction=args.train_fraction,
            structured_result_hint=args.structured_result_hint,
            structured_feature_strength=args.structured_feature_strength,
        )
        results.append(result)
    payload: dict[str, Any]
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {
            "comparison": results,
            "best_by_pair_probe": max(
                results,
                key=lambda item: float(item["probes"]["pair"]["accuracy"]),
            )["encoder_mode"],
        }
    if args.out is not None:
        save_probe_result(payload, args.out)
    print(json.dumps(payload, indent=2))
    return 0


def probe_arithmetic_result(args: argparse.Namespace) -> int:
    from .probes import run_arithmetic_result_readout_probe, save_probe_result

    ops = tuple(item.strip() for item in str(args.ops).split(",") if item.strip())
    encoder_modes = ("text", "structured") if args.compare_text else (args.encoder,)
    results = []
    for encoder_mode in encoder_modes:
        result = run_arithmetic_result_readout_probe(
            encoder_mode=encoder_mode,
            max_value=args.max_value,
            min_value=args.min_value,
            ops=ops,
            grid_size=args.size,
            basin_dim=args.basin_dim,
            hidden=args.hidden,
            steps_per_chunk=args.steps_per_chunk,
            seed=args.seed,
            backend=args.backend,
            train_fraction=args.train_fraction,
            structured_result_hint=args.structured_result_hint,
            structured_feature_strength=args.structured_feature_strength,
        )
        results.append(result)
    payload: dict[str, Any]
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {
            "comparison": results,
            "best_by_factorized_result": max(
                results,
                key=lambda item: float(item["factorized_result"]["accuracy"]),
            )["encoder_mode"],
        }
    if args.out is not None:
        save_probe_result(payload, args.out)
    print(json.dumps(payload, indent=2))
    return 0


def fit_arithmetic_readout(args: argparse.Namespace) -> int:
    from .probes import fit_save_arithmetic_factor_readout

    ops = tuple(item.strip() for item in str(args.ops).split(",") if item.strip())
    payload = fit_save_arithmetic_factor_readout(
        args.out,
        max_value=args.max_value,
        min_value=args.min_value,
        ops=ops,
        grid_size=args.size,
        basin_dim=args.basin_dim,
        hidden=args.hidden,
        steps_per_chunk=args.steps_per_chunk,
        seed=args.seed,
        backend=args.backend,
        train_fraction=args.train_fraction,
        structured_result_hint=args.structured_result_hint,
        structured_feature_strength=args.structured_feature_strength,
    )
    print(json.dumps(payload, indent=2))
    return 0


def solve_arithmetic(args: argparse.Namespace) -> int:
    from .probes import ArithmeticFactorReadout, save_probe_result, solve_arithmetic_with_factor_readout

    ops = tuple(item.strip() for item in str(args.ops).split(",") if item.strip())
    if args.readout_dir is not None:
        readout = ArithmeticFactorReadout.load(args.readout_dir)
        payload = readout.solve(" ".join(args.text))
    else:
        payload = solve_arithmetic_with_factor_readout(
            " ".join(args.text),
            max_value=args.max_value,
            min_value=args.min_value,
            ops=ops,
            grid_size=args.size,
            basin_dim=args.basin_dim,
            hidden=args.hidden,
            steps_per_chunk=args.steps_per_chunk,
            seed=args.seed,
            backend=args.backend,
            structured_result_hint=args.structured_result_hint,
            structured_feature_strength=args.structured_feature_strength,
        )
    if args.out is not None:
        save_probe_result(payload, args.out)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 2


def tool_for(route_name: str, checker: str) -> str:
    if checker.startswith("arithmetic"):
        return "python-eval"
    if checker == "python-compile":
        return "python-compile"
    if checker == "json":
        return "json-parser"
    route_tools = {
        "calculate": "python-eval",
        "code": "python-repl",
        "verify": "verifier",
        "search": "web-or-docs-search",
        "write": "editor",
        "compress": "note-compressor",
        "plan": "planner",
        "act": "shell",
    }
    return route_tools.get(route_name, "router")


def serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    os.environ["PHASE_MESH_SIZE"] = str(args.size)
    os.environ["PHASE_MESH_STEPS"] = str(args.steps)
    os.environ["PHASE_MESH_SEED"] = str(args.seed)
    os.environ["PHASE_MESH_BACKEND"] = args.backend
    os.environ["PHASE_MESH_PIN"] = str(args.pin_strength)
    os.environ["PHASE_MESH_RESIDUAL_CARRY"] = str(args.residual_carry)
    os.environ["PHASE_MESH_STATE_DIR"] = str(args.state_dir)
    if args.no_persist:
        os.environ["PHASE_MESH_PERSIST"] = "0"

    uvicorn.run(
        "phase_mesh.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def bench(args: argparse.Namespace) -> int:
    from bench.run_all import run

    payload = run(
        trials=args.trials,
        facts=args.facts,
        math_count=args.math_count,
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        backend=args.backend,
        pin_strength=args.pin_strength,
        residual_carry=args.residual_carry,
        out=args.out,
    )
    print(json.dumps(payload, indent=2))
    return 0


def emit(payload: dict[str, Any], runtime: CognitiveMeshRuntime, out_dir: Path | None, stem: str) -> None:
    print(json.dumps(payload, indent=2))
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    runtime.mesh.save(out_dir / f"{stem}.npz")
    runtime.mesh.save_quantized(out_dir / f"{stem}.q8.npz")
    save_phase_image(runtime.mesh, out_dir / f"{stem}.png", title=f"phase mesh: {stem}")


if __name__ == "__main__":
    raise SystemExit(main())
