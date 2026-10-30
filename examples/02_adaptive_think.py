from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh import CognitiveMeshRuntime, MeshConfig


def main() -> None:
    runtime = CognitiveMeshRuntime(
        MeshConfig(
            width=128,
            height=128,
            max_steps=180,
            phase_pin_strength=0.25,
        )
    )
    run = runtime.think(
        "check whether 17 * 19 = 323 and stabilize the answer",
        expected="323",
        max_budget=120,
        temperature=0.2,
        verifier_control=True,
    )
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
