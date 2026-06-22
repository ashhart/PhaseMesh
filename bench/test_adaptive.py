from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import make_runtime, write_result


def run(
    *,
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
    easy_budget: int = 40,
    hard_budget: int = 240,
    temperature: float = 0.35,
    basin_repeats: int = 4,
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
    easy = runtime.think(
        "check 17 * 19 = 323",
        max_budget=easy_budget,
        temperature=0.0,
        expected=None,
        learn=True,
    )
    repeat_runs = [
        runtime.think(
            "check 17 * 19 = 323",
            max_budget=easy_budget,
            temperature=0.0,
            learn=False,
        ).to_dict()
        for _ in range(max(0, basin_repeats - 1))
    ]

    hard_prompt = (
        "ctx_000 ctx_001 ctx_002 ctx_003 ctx_004 ctx_005 ctx_006 ctx_007 "
        "ctx_008 ctx_009 ctx_010 ctx_011 ctx_012 ctx_013 ctx_014 ctx_015 "
        "route a contradiction: 17 * 19 = 320 but earlier arithmetic says 323; "
        "stabilize the correct basin and preserve ctx_005"
    )
    hard = runtime.think(
        hard_prompt,
        max_budget=hard_budget,
        temperature=temperature,
        learn=True,
        verifier_control=True,
    )
    basin_state = runtime.discover_basins()

    easy_accuracy = max(0.0, 1.0 - easy.mean_prediction_error)
    hard_accuracy = max(0.0, 1.0 - hard.mean_prediction_error)
    step_ratio = hard.steps_used / max(1, easy.steps_used)
    persistent_basins = [
        basin for basin in basin_state["basins"] if basin["persistence"] >= 0.70
    ]

    payload = {
        "experiment": "adaptive_predictive_compute",
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "easy": easy.to_dict(),
        "hard": hard.to_dict(),
        "basin_repeat_runs": repeat_runs,
        "prediction_accuracy": {
            "easy": easy_accuracy,
            "hard": hard_accuracy,
            "target": 0.85,
            "passed": easy_accuracy >= 0.85 and hard_accuracy >= 0.85,
        },
        "adaptive_speedup": {
            "easy_steps": easy.steps_used,
            "hard_steps": hard.steps_used,
            "hard_to_easy_ratio": step_ratio,
            "target": 1.5,
            "passed": step_ratio >= 1.5,
        },
        "basin_persistence": {
            "basin_count": len(basin_state["basins"]),
            "persistent_count": len(persistent_basins),
            "target": 1,
            "passed": len(persistent_basins) >= 1,
            "basins": basin_state["basins"],
        },
    }
    payload["passed"] = bool(
        payload["prediction_accuracy"]["passed"]
        and payload["adaptive_speedup"]["passed"]
        and payload["basin_persistence"]["passed"]
    )
    payload["output_path"] = str(write_result(out, "adaptive", payload))
    runtime.basin_tracker.save(Path(out) / "basins.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run predictive adaptive-compute benchmark.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--easy-budget", type=int, default=40)
    parser.add_argument("--hard-budget", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--basin-repeats", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args(argv)
    print_json(run(**vars(args)))
    return 0


def print_json(payload: dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
