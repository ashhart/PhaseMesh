from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from phase_mesh import MeshConfig, PhaseFieldMesh, TextPhaseEncoder
from phase_mesh.encoding import PhaseDecoder
from phase_mesh.verifier import VerifierRouter

from .common import write_result


def run(
    *,
    prompt: str = "check 17 * 19 = 320",
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
    correction_steps: int = 3,
    out: str | Path = "runs/bench",
) -> dict[str, Any]:
    config = MeshConfig(
        width=size,
        height=size,
        max_steps=steps,
        seed=seed,
        laplacian_backend=backend,
        phase_pin_strength=pin_strength,
        phase_residual_carry=residual_carry,
    )
    mesh = PhaseFieldMesh(config)
    encoder = TextPhaseEncoder(size, size)
    decoder = PhaseDecoder()
    verifier = VerifierRouter()

    mesh.inject_text(prompt, encoder)
    history = mesh.run_until_resonance()
    before_metrics = history[-1] if history else mesh.metrics()
    before_decoded = decoder.decode(mesh.theta, coherence=before_metrics.coherence)
    verification = verifier.verify(prompt, candidate=before_decoded.route, coherence=before_metrics.coherence)

    mesh.apply_feedback(success=verification.passed, message=verification.message, encoder=encoder)
    after_records = []
    for index in range(correction_steps):
        metrics = mesh.step()
        decoded = decoder.decode(mesh.theta, coherence=metrics.coherence)
        after_records.append(
            {
                "step_after_feedback": index + 1,
                "decoded": decoded.to_dict(),
                "metrics": metrics.to_dict(),
            }
        )

    final = after_records[-1]
    destabilized = (
        final["decoded"]["signature"] != before_decoded.signature
        or final["decoded"]["dominant_sector"] != before_decoded.dominant_sector
        or final["metrics"]["coherence"] < before_metrics.coherence - 0.01
    )
    payload = {
        "experiment": "verifier_correction",
        "prompt": prompt,
        "backend": backend,
        "pin_strength": pin_strength,
        "residual_carry": residual_carry,
        "before": {
            "decoded": before_decoded.to_dict(),
            "metrics": before_metrics.to_dict(),
            "verifier": verification.to_dict(),
        },
        "after": after_records,
        "target": f"error basin destabilizes within {correction_steps} steps",
        "passed": bool((not verification.passed) and destabilized),
    }
    payload["output_path"] = str(write_result(out, "correction", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run verifier-correction benchmark.")
    parser.add_argument("--prompt", default="check 17 * 19 = 320")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.0)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--correction-steps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args(argv)
    print_json(run(**vars(args)))
    return 0


def print_json(payload: dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
