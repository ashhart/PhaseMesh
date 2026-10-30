#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_mesh.model import PhaseModel


def extract_prototypes(model_dir: str | Path) -> dict[str, np.ndarray]:
    model = PhaseModel.load(model_dir)
    prototypes = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in sorted(model.structural_prototypes.items())
    }
    print(f"Extracted {len(prototypes)} prototypes from {model_dir}")
    if prototypes:
        print("Sample keys:", ", ".join(list(prototypes)[:8]))
    return prototypes


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract PhaseMesh structural prototypes for decoder sync.")
    parser.add_argument("--model-dir", type=Path, default=Path("runs/anchored-model/final"))
    parser.add_argument("--out", type=Path, default=Path("runs/prototypes.pt"))
    args = parser.parse_args()

    prototypes = extract_prototypes(args.model_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except Exception as exc:
        raise SystemExit("PyTorch is required to save .pt prototype artifacts.") from exc
    torch.save(prototypes, args.out)
    print(f"Saved prototypes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
