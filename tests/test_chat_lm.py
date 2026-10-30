from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PhaseChatModelTests(unittest.TestCase):
    def test_chat_model_retrieves_teacher_response_and_persists(self) -> None:
        from phase_mesh.chat_lm import PhaseChatConfig, PhaseChatModel

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "teacher_samples.jsonl"
            samples.write_text(
                "\n".join([
                    json.dumps({
                        "prompt": "Write a Python function that adds two numbers",
                        "text": "def add(a, b):\n    return a + b",
                    }),
                    json.dumps({
                        "prompt": "Explain why a failing unit test is useful",
                        "text": "A failing unit test is useful because it gives reproducible evidence.",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            model = PhaseChatModel.from_teacher_samples(samples, config=PhaseChatConfig(signature_cells=512, retrieval_threshold=0.1))
            answer = model.answer("Please write a Python function that adds two numbers", threshold=0.1)
            model.save(root / "model")
            loaded = PhaseChatModel.load(root / "model")
            loaded_answer = loaded.answer("Why is a failing unit test useful?", threshold=0.1)

        self.assertEqual(answer["mode"], "phase-chat-retrieval")
        self.assertGreater(answer["score"], 0.0)
        self.assertIn("return a + b", answer["completion"])
        self.assertEqual(loaded_answer["mode"], "phase-chat-retrieval")
        self.assertIn("reproducible evidence", loaded_answer["completion"])

    def test_chat_cli_build_and_query(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "teacher_samples.jsonl"
            model_dir = root / "chat"
            samples.write_text(
                json.dumps({
                    "prompt": "Find the bug in add",
                    "text": "The bug is subtraction; use return a + b.",
                }) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-chat-build",
                    str(samples),
                    "--out",
                    str(model_dir),
                    "--signature-cells",
                    "512",
                    "--retrieval-threshold",
                    "0.1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "lm-chat",
                    "Find the bug in add",
                    "--model-dir",
                    str(model_dir),
                    "--threshold",
                    "0.1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("return a + b", result.stdout)

    def test_chat_model_separates_similar_code_prompts(self) -> None:
        from phase_mesh.chat_lm import PhaseChatConfig, PhaseChatModel

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "teacher_samples.jsonl"
            samples.write_text(
                "\n".join([
                    json.dumps({
                        "prompt": "Write a concise Python function that adds two numbers and explain it in one sentence.",
                        "text": "user\nWrite a concise Python function that adds two numbers and explain it in one sentence.\nassistant\n```python\ndef add(a, b):\n    return a + b\n```",
                    }),
                    json.dumps({
                        "prompt": "Write a Python function that filters even numbers from a list.",
                        "text": "user\nWrite a Python function that filters even numbers from a list.\nassistant\ndef filter_even_numbers(numbers):\n    return [n for n in numbers if n % 2 == 0]",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )

            model = PhaseChatModel.from_teacher_samples(samples, config=PhaseChatConfig(signature_cells=512, retrieval_threshold=0.1))
            answer = model.answer("Write a Python function that adds two numbers", threshold=0.1)

        self.assertEqual(answer["mode"], "phase-chat-retrieval")
        self.assertIn("return a + b", answer["completion"])
        self.assertNotIn("filter_even_numbers", answer["completion"])

    def test_chat_model_rejects_topic_mismatched_lookalike(self) -> None:
        from phase_mesh.chat_lm import PhaseChatConfig, PhaseChatModel

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "teacher_samples.jsonl"
            samples.write_text(
                json.dumps({
                    "prompt": "Write a Python function that counts words in a string.",
                    "text": "def count_words(text):\n    return len(text.split())",
                }) + "\n",
                encoding="utf-8",
            )

            model = PhaseChatModel.from_teacher_samples(samples, config=PhaseChatConfig(signature_cells=512, retrieval_threshold=0.1))
            answer = model.answer("Write a Python function that reverses a string.", threshold=0.1, max_tokens=1, allow_fallback=False)

        self.assertEqual(answer["mode"], "phase-chat-abstain")
        self.assertLess(answer["confidence"]["topic_coverage"], answer["confidence"]["topic_coverage_threshold"])


if __name__ == "__main__":
    unittest.main()
