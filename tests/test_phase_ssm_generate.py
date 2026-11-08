from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class PhaseSSMGenerateTests(unittest.TestCase):
    def test_checkpoint_generate_cli_loads_and_prints_json(self) -> None:
        from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pt"
            cfg = PhaseSSMConfig(vocab_size=256, d_model=16, n_layers=1, state_dim=4, expand=1, d_ff_mult=1)
            model = PhaseSSMLM(cfg)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "model_type": "phasessm",
                    "step": 0,
                    "val_bpc": 9.0,
                },
                checkpoint,
            )

            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_ssm.generate",
                    "--checkpoint",
                    str(checkpoint),
                    "--max-tokens",
                    "2",
                    "--temperature",
                    "0",
                    "--json",
                    "hi",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn('"prompt": "hi"', run.stdout)
        self.assertIn('"tokens_generated": 2', run.stdout)
        self.assertIn('"model_type": "phasessm"', run.stdout)
        self.assertIn('"tokens_per_s"', run.stdout)
        self.assertIn('"limits"', run.stdout)

    def test_load_checkpoint_refuses_pickled_code(self) -> None:
        import pickle

        from phase_ssm.generate import load_checkpoint

        class _OpensMarkerWhenUnpickled:
            def __init__(self, marker: Path) -> None:
                self.marker = marker

            def __reduce__(self):
                return (open, (str(self.marker), "w"))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "unpickled-code-ran"
            torch.save(_OpensMarkerWhenUnpickled(marker), root / "best.pt")
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(root, "cpu")
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
