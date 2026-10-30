from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PhaseDistillationTests(unittest.TestCase):
    def test_prompt_file_ignores_blank_lines_and_comments(self) -> None:
        from phase_mesh.distill import read_distill_prompts

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.txt"
            path.write_text("\n# skip me\nPhaseMesh learns\n\nPhaseMesh answers\n", encoding="utf-8")
            prompts = read_distill_prompts(path)

        self.assertEqual(prompts, ["PhaseMesh learns", "PhaseMesh answers"])

    def test_train_phase_lm_from_teacher_texts(self) -> None:
        from phase_mesh.distill import train_phase_lm_from_texts
        from phase_mesh.language_model import PhaseLMConfig, PhaseLanguageModel

        texts = [
            "phase mesh learns from a teacher model . phase mesh writes answers .",
            "phase mesh learns from a teacher model . phase mesh routes prompts .",
        ] * 4
        with TemporaryDirectory() as tmp:
            payload = train_phase_lm_from_texts(
                texts,
                out_dir=tmp,
                config=PhaseLMConfig(order=3, phase_cells=256, vocab_size=128),
            )
            model = PhaseLanguageModel.load(tmp)
            generated = model.generate("phase mesh", max_tokens=8, temperature=0.0, top_k=8)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("learns", generated["text"])
        self.assertGreater(model.summary()["training_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
