from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh.config import MeshConfig
from phase_mesh.field import PhaseFieldMesh, laplacian


def custom_laplacian(theta: np.ndarray) -> np.ndarray:
    """Drop-in place to experiment with CUDA, Triton, or alternate kernels."""

    return laplacian(theta, backend="scipy")


def main() -> None:
    mesh = PhaseFieldMesh(MeshConfig(width=64, height=64, max_steps=80))
    mesh.inject_text("custom backend smoke test")
    for _ in range(20):
        lap = custom_laplacian(mesh.theta)
        metrics = mesh.step(external_force=0.001 * lap)
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
