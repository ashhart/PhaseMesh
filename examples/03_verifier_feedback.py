from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh import CognitiveMeshRuntime, MeshConfig


def main() -> None:
    out_dir = Path("runs/example-feedback")
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = CognitiveMeshRuntime(
        MeshConfig(
            width=128,
            height=128,
            max_steps=180,
            phase_pin_strength=0.25,
        )
    )
    rounds = []
    for _ in range(4):
        run = runtime.resonate("17 * 19", expected="323", learn=True)
        rounds.append(run.to_dict())

    runtime.mesh.consolidate(cycles=12)
    runtime.mesh.save_quantized(out_dir / "topology.q8.npz")
    (out_dir / "summary.json").write_text(json.dumps({"rounds": rounds}, indent=2) + "\n")
    print(f"wrote {out_dir / 'topology.q8.npz'}")


if __name__ == "__main__":
    main()
