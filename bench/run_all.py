from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import compare, test_adaptive, test_context, test_correction, test_memory, test_stability, train_math
from .common import write_result


def run(
    *,
    trials: int = 50,
    facts: int = 10,
    math_count: int = 50,
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
    out: str | Path = "runs/bench",
) -> dict[str, Any]:
    out_path = Path(out)
    results = {
        "stability": test_stability.run(
            trials=trials,
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "correction": test_correction.run(
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "context": test_context.run(
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "memory": test_memory.run(
            facts=facts,
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "math_logic": train_math.run(
            count=math_count,
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "compare": compare.run(
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
        "adaptive": test_adaptive.run(
            size=size,
            steps=steps,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
            out=out_path,
        ),
    }
    payload = {
        "suite": "phase_mesh_bench",
        "size": size,
        "steps": steps,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "passed": {name: result.get("passed") for name, result in results.items()},
        "results": results,
    }
    payload["output_path"] = str(write_result(out_path, "results", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phase-mesh benchmark suite.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--facts", type=int, default=10)
    parser.add_argument("--math-count", type=int, default=50)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args(argv)
    print(json.dumps(run(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
