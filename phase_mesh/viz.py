from __future__ import annotations

from pathlib import Path

from .field import PhaseFieldMesh


def save_phase_image(mesh: PhaseFieldMesh, path: str | Path, *, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    phase = axes[0].imshow(mesh.theta, cmap="twilight", origin="lower")
    axes[0].set_title(title or "Phase field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(phase, ax=axes[0], fraction=0.046, pad=0.04)

    landscape = axes[1].imshow(mesh.landscape, cmap="coolwarm", origin="lower")
    axes[1].set_title("Potential landscape")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(landscape, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=150)
    plt.close(fig)

