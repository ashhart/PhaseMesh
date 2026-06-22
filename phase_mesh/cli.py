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
