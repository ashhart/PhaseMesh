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


if __name__ == "__main__":
    unittest.main()
