from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class PhaseWeightPourTests(unittest.TestCase):
    def test_weight_pour_uses_tensor_values_and_writes_manifest(self) -> None:
        from phase_mesh.weight_pour import PhaseWeightPourConfig, pour_arrays_to_phase

        arrays = {
            "model.embed_tokens.weight": np.arange(24, dtype=np.float32).reshape(6, 4) / 10.0,
            "model.layers.0.self_attn.q_proj.weight": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4),
        }
        changed = {
            "model.embed_tokens.weight": arrays["model.embed_tokens.weight"].copy(),
            "model.layers.0.self_attn.q_proj.weight": arrays["model.layers.0.self_attn.q_proj.weight"].copy(),
        }
        changed["model.layers.0.self_attn.q_proj.weight"][0, 0] += 1.0

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = pour_arrays_to_phase(
                arrays,
                out_dir=root / "first",
                config=PhaseWeightPourConfig(phase_cells=128, token_cells=8, chunk_size=7),
            )
            second = pour_arrays_to_phase(
                changed,
                out_dir=root / "second",
                config=PhaseWeightPourConfig(phase_cells=128, token_cells=8, chunk_size=7),
            )
            bank_first = np.load(root / "first" / "phase_weight_bank.npz")
            bank_second = np.load(root / "second" / "phase_weight_bank.npz")
            stats = (root / "first" / "tensor_stats.jsonl").read_text(encoding="utf-8")

        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["elements_seen"], 56)
        self.assertEqual(first["tensors"], 2)
        self.assertTrue(first["token_signature_files"])
        self.assertIn("q_proj", stats)
        self.assertNotEqual(first["phase_bank_norm"], second["phase_bank_norm"])
        self.assertFalse(np.allclose(bank_first["real"], bank_second["real"]))

    def test_weight_reader_ranks_and_generates_from_artifact(self) -> None:
        from phase_mesh.weight_reader import PhaseWeightReader, PhaseWeightReadoutConfig, _position_phasor

        class TinyTokenizer:
            bos_token_id = 0
            eos_token_id = 1
            pad_token_id = 2
            unk_token_id = None
            all_special_ids = [0, 1, 2]

            def encode(self, text, add_special_tokens=False):
                return [3 if "alpha" in str(text).lower() else 5]

            def decode(self, ids, skip_special_tokens=True):
                vocab = {3: " alpha", 4: " beta", 5: " gamma"}
                return "".join(vocab.get(int(item), "") for item in ids)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            cells = 8
            signatures = np.zeros((8, cells), dtype=np.complex64)
            base = np.ones(cells, dtype=np.complex64)
            base /= np.linalg.norm(base)
            signatures[3] = base
            signatures[4] = base * _position_phasor(0, cells, 7)
            signatures[5] = -signatures[4]
            np.savez_compressed(root / "token_signatures_model.embed_tokens.weight.npz", real=signatures.real, imag=signatures.imag)
            np.savez_compressed(root / "phase_weight_bank.npz", real=np.zeros(cells, dtype=np.float32), imag=np.zeros(cells, dtype=np.float32))
            (root / "manifest.json").write_text(
                """{
  "status": "ok",
  "type": "phase-mesh-weight-pour",
  "source": "tiny",
  "config": {"seed": 7},
  "elements_seen": 24,
  "tensors": 1,
  "phase_bank": "phase_weight_bank.npz",
  "phase_bank_norm": 0.0,
  "token_signature_files": ["token_signatures_model.embed_tokens.weight.npz"]
}
""",
                encoding="utf-8",
            )

            reader = PhaseWeightReader(
                root,
                tokenizer=TinyTokenizer(),
                config=PhaseWeightReadoutConfig(phase_mix=0.0, context_tokens=4, seed=7),
            )
            ranks = reader.rank_tokens("alpha", top_k=2)
            generated = reader.generate("alpha", max_tokens=1, top_k=2, temperature=0.0)

        self.assertEqual(ranks[0]["id"], 4)
        self.assertIn("beta", generated["completion"])


if __name__ == "__main__":
    unittest.main()
