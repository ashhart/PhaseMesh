from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PhaseLanguageModelTests(unittest.TestCase):
    def _corpus(self) -> str:
        return "\n".join([
            "phase mesh learns language . phase mesh learns memory . phase mesh writes answers .",
            "phase mesh learns language . phase mesh learns memory . phase mesh routes prompts .",
            "phase mesh learns language . phase mesh writes answers . phase mesh routes prompts .",
        ] * 8)

    def test_phase_language_model_trains_scores_generates_and_loads(self) -> None:
        from phase_mesh.language_model import PhaseLanguageModel, PhaseLMConfig

        model = PhaseLanguageModel(PhaseLMConfig(order=4, phase_cells=512, vocab_size=128, seed=3))
        summary = model.train_text(self._corpus())
        scores = model.next_scores(["phase", "mesh"], top_k=3)
        generated = model.generate("phase mesh", max_tokens=16, temperature=0.0, top_k=8)
        evaluation = model.evaluate_text(self._corpus())

        self.assertGreater(summary["windows"], 0)
        self.assertEqual(scores[0][0], "learns")
        self.assertIn("phase mesh learns", generated["text"])
        self.assertNotRegex(generated["text"], r"[^\w\s]{2,}")
        self.assertFalse(_has_repeated_ngram(generated["tokens"], 3), generated["tokens"])
        self.assertGreater(evaluation["top1"], 0.70)
        self.assertLess(evaluation["perplexity"], 10.0)
        self.assertGreater(model.summary()["phase_memory_norm"], 0.0)

        with TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = PhaseLanguageModel.load(tmp)
            loaded_scores = loaded.next_scores(["phase", "mesh"], top_k=1)
            loaded_generated = loaded.generate("phase mesh", max_tokens=4, temperature=0.0, top_k=5)

        self.assertEqual(loaded_scores[0][0], "learns")
        self.assertIn("phase mesh learns", loaded_generated["text"])

    def test_soft_distribution_training_pushes_teacher_candidate(self) -> None:
        from phase_mesh.language_model import PhaseLanguageModel, PhaseLMConfig

        model = PhaseLanguageModel(PhaseLMConfig(order=3, phase_cells=256, vocab_size=128, seed=9))
        model.train_text("phase mesh answers locally .")
        before = model.next_scores(["phase", "mesh"], top_k=3)
        payload = model.train_next_distribution(
            "phase mesh",
            [("absorbs", 0.92), ("ignores", 0.08)],
            weight_scale=8.0,
        )
        after = model.next_scores(["phase", "mesh"], top_k=3)

        self.assertEqual(payload["candidates_added"], 2)
        self.assertNotEqual(before[0][0], "absorbs")
        self.assertEqual(after[0][0], "absorbs")

    def test_training_populates_all_backoff_suffix_counts(self) -> None:
        from phase_mesh.language_model import PhaseLanguageModel, PhaseLMConfig

        model = PhaseLanguageModel(PhaseLMConfig(order=3, phase_cells=128, vocab_size=128, seed=13))
        model.train_text("alpha beta gamma delta")
        alpha = model.token_to_id["alpha"]
        beta = model.token_to_id["beta"]
        gamma = model.token_to_id["gamma"]
        delta = model.token_to_id["delta"]

        self.assertGreater(model.context_counts[()][alpha], 0)
        self.assertGreater(model.context_counts[(alpha,)][beta], 0)
        self.assertGreater(model.context_counts[(alpha, beta)][gamma], 0)
        self.assertGreater(model.context_counts[(alpha, beta, gamma)][delta], 0)

    def test_phase_language_model_cli_train_generate_eval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.txt"
            model_dir = root / "model"
            corpus.write_text(self._corpus(), encoding="utf-8")

            train = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-train",
                    str(corpus),
                    "--out",
                    str(model_dir),
                    "--order",
                    "2",
                    "--phase-cells",
                    "256",
                    "--vocab-size",
                    "128",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-generate",
                    "phase mesh",
                    "--model-dir",
                    str(model_dir),
                    "--max-tokens",
                    "4",
                    "--temperature",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            evaluated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-eval",
                    str(corpus),
                    "--model-dir",
                    str(model_dir),
                    "--max-tokens",
                    "40",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn('"status": "ok"', train.stdout)
        self.assertIn("phase mesh learns", generated.stdout)
        self.assertIn('"perplexity"', evaluated.stdout)

    def test_lm_train_whole_file_preserves_cross_line_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.txt"
            model_dir = root / "model"
            corpus.write_text("alpha beta\ngamma delta\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-train",
                    str(corpus),
                    "--out",
                    str(model_dir),
                    "--order",
                    "2",
                    "--phase-cells",
                    "128",
                    "--vocab-size",
                    "64",
                    "--whole-file",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

            from phase_mesh.language_model import PhaseLanguageModel

            model = PhaseLanguageModel.load(model_dir)

        beta = model.token_to_id["beta"]
        gamma = model.token_to_id["gamma"]
        self.assertGreater(model.context_counts[(beta,)][gamma], 0)


def _has_repeated_ngram(tokens: list[str], width: int) -> bool:
    seen = set()
    for index in range(0, len(tokens) - width + 1):
        ngram = tuple(tokens[index : index + width])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


if __name__ == "__main__":
    unittest.main()
