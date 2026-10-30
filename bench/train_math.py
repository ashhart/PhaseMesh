from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import make_runtime, summarize_numbers, timed, write_result


def run(
    *,
    count: int = 50,
    rounds: int = 2,
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
    records = []
    for index in range(count):
        a = 11 + (index * 7) % 89
        b = 13 + (index * 11) % 83
        expression = f"{a} * {b}"
        expected = str(a * b)
        result, elapsed = timed(lambda expression=expression, expected=expected: runtime.learn(
            expression,
            expected=expected,
            rounds=rounds,
            steps=steps,
        ))
        final = result["final"]
        records.append(
            {
                "expression": expression,
                "expected": expected,
                "passed": final["verifier"]["passed"],
                "resonance_step": final["metrics"]["step"],
                "coherence": final["metrics"]["coherence"],
                "elapsed_s": elapsed,
            }
        )

    pass_rate = sum(1 for item in records if item["passed"]) / max(1, count)
    payload = {
        "experiment": "math_logic_training",
        "count": count,
        "rounds": rounds,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "pass_rate": pass_rate,
        "resonance_steps": summarize_numbers([float(item["resonance_step"]) for item in records]),
        "elapsed_s": summarize_numbers([float(item["elapsed_s"]) for item in records]),
        "records": records,
    }
    payload["output_path"] = str(write_result(out, "math_logic", payload))
    runtime.mesh.save(Path(out) / "math_logic_state.npz")
    runtime.mesh.save_quantized(Path(out) / "math_logic_state_q8.npz")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train on generated math examples.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
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
