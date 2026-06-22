from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import make_runtime, write_result


def run(
    *,
    token_count: int = 50,
    target_index: int = 5,
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
    out: str | Path = "runs/bench",
) -> dict[str, Any]:
    runtime = make_runtime(
        size=size,
        steps=steps,
        seed=seed,
        backend=backend,
        pin_strength=pin_strength,
        residual_carry=residual_carry,
    )
    tokens = [f"ctx_{index:03d}" for index in range(token_count)]
    target = tokens[target_index]
    prompt = " ".join(tokens) + f"\nquery: stabilize around early token {target}"
    run_result = runtime.resonate(prompt)
    gradient = run_result.metrics.gradient
    payload = {
        "experiment": "context_retention",
        "token_count": token_count,
        "target_index": target_index,
        "target_token": target,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "gradient": gradient,
        "coherence": run_result.metrics.coherence,
        "target_gradient": 0.05,
        "passed": gradient < 0.05,
        "run": run_result.to_dict(),
    }
    payload["output_path"] = str(write_result(out, "context", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run context-retention benchmark.")
    parser.add_argument("--token-count", type=int, default=50)
    parser.add_argument("--target-index", type=int, default=5)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args(argv)
    print_json(run(**vars(args)))
    return 0


def print_json(payload: dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
