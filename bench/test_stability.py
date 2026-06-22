from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phase_mesh import MeshConfig, PhaseFieldMesh, TextPhaseEncoder
from phase_mesh.encoding import PhaseDecoder

from .common import timed, write_result


def run(
    *,
    prompt: str = "check 17 * 19 = 323 and route the result",
    trials: int = 50,
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
    phase_noise: float = 0.25,
    out: str | Path = "runs/bench",
) -> dict[str, Any]:
    decoder = PhaseDecoder()
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    def one_trial(index: int) -> dict[str, Any]:
        config = MeshConfig(
            width=size,
            height=size,
            max_steps=steps,
            seed=seed + index,
            laplacian_backend=backend,
            phase_pin_strength=pin_strength,
            phase_residual_carry=residual_carry,
        )
        mesh = PhaseFieldMesh(config)
        encoder = TextPhaseEncoder(size, size)
        for packet in encoder.encode(prompt):
            shifted = replace(
                packet,
                phase=(packet.phase + rng.uniform(-phase_noise, phase_noise)) % (2.0 * math.pi),
            )
            mesh.inject_packet(shifted)
        history = mesh.run_until_resonance()
        metrics = history[-1] if history else mesh.metrics()
        decoded = decoder.decode(mesh.theta, coherence=metrics.coherence)
        return {
            "trial": index + 1,
            "sector": decoded.dominant_sector,
            "route": decoded.route,
            "signature": decoded.signature,
            "confidence": decoded.confidence,
            "metrics": metrics.to_dict(),
        }

    for index in range(trials):
        record, elapsed = timed(lambda index=index: one_trial(index))
        record["elapsed_s"] = elapsed
        records.append(record)

    basin_counts = Counter((item["sector"], item["route"]) for item in records)
    mode_basin, mode_count = basin_counts.most_common(1)[0]
    mode_ratio = mode_count / max(1, trials)
    payload = {
        "experiment": "phase_lock_stability",
        "prompt": prompt,
        "trials": trials,
        "size": size,
        "steps": steps,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "phase_noise": phase_noise,
        "mode_basin": {"sector": mode_basin[0], "route": mode_basin[1]},
        "mode_ratio": mode_ratio,
        "target": 0.90,
        "passed": mode_ratio >= 0.90,
        "records": records,
    }
    payload["output_path"] = str(write_result(out, "stability", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phase-lock stability benchmark.")
    parser.add_argument("--prompt", default="check 17 * 19 = 323 and route the result")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--phase-noise", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args(argv)
    print_json(run(**vars(args)))
    return 0


def print_json(payload: dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
