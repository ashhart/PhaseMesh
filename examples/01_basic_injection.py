from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh import CognitiveMeshRuntime, MeshConfig


def main() -> None:
    runtime = CognitiveMeshRuntime(MeshConfig(width=128, height=128, max_steps=180))
    run = runtime.resonate(
        "check 17 * 19 = 323 and route the result",
        expected="323",
        learn=False,
    )
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
