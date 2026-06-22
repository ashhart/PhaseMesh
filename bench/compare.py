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
    out: str | Path = "runs/bench",
) -> dict[str, Any]:
    out_path = Path(out)
    runtime = make_runtime(
        size=size,
        steps=steps,
        seed=seed,
        backend=backend,
        pin_strength=pin_strength,
        residual_carry=residual_carry,
    )
    runtime.learn("17 * 19", expected="323", rounds=2, steps=steps)
    full_state = out_path / "compare_state_full.npz"
    q_state = out_path / "compare_state_q8.npz"
    runtime.mesh.save(full_state)
    runtime.mesh.save_quantized(q_state)
    full_bytes = full_state.stat().st_size
    q_bytes = q_state.stat().st_size
    payload = {
        "experiment": "disk_footprint",
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "state_full_bytes": full_bytes,
        "state_q8_bytes": q_bytes,
        "compression_ratio": full_bytes / max(1, q_bytes),
        "reference_7b_fp16_bytes": 14_000_000_000,
        "reference_note": "7B fp16 value is a size reference, not a local measured baseline.",
    }
    payload["output_path"] = str(write_result(out_path, "compare", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight footprint comparison.")
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
