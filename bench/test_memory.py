from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import make_runtime, write_result


def run(
    *,
    facts: int = 10,
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
    expected: dict[str, str] = {}
    remembers = []
    for index in range(facts):
        key = f"mesh_fact_{index:02d}"
        value = f"value_{(index * 37 + 11) % 997:03d}"
        expected[key] = value
        remembers.append(runtime.remember(key, value, steps=steps))

    runtime.mesh.consolidate(cycles=24)
    recall_records = []
    correct = 0
    for key, value in expected.items():
        recalled = runtime.recall(key, steps=steps)
        got = recalled["recall"]["value"]
        is_correct = got == value
        correct += int(is_correct)
        recall_records.append(
            {
                "key": key,
                "expected": value,
                "got": got,
                "correct": is_correct,
                "score": recalled["recall"]["score"],
            }
        )

    recall_rate = correct / max(1, facts)
    payload = {
        "experiment": "topology_memory",
        "facts": facts,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "recall_rate": recall_rate,
        "target": 0.85,
        "passed": recall_rate >= 0.85,
        "remembers": remembers,
        "recalls": recall_records,
    }
    payload["output_path"] = str(write_result(out, "memory", payload))
    runtime.memory.save(Path(out) / "memory_store.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run topology-memory benchmark.")
    parser.add_argument("--facts", type=int, default=10)
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
