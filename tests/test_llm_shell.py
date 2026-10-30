from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REQUIRED_ORGANS = {
    "memory_retrieval",
    "binding",
    "reasoning",
    "generation",
    "learning",
    "control",
}


class PhaseMeshLLMShellContractTests(unittest.TestCase):
    def _shell_class(self):
        try:
            module = importlib.import_module("phase_mesh.llm_shell")
        except ModuleNotFoundError as exc:
            raise AssertionError("expected module phase_mesh.llm_shell to exist") from exc
        try:
            return module.PhaseMeshLLMShell
        except AttributeError as exc:
            raise AssertionError("phase_mesh.llm_shell must expose PhaseMeshLLMShell") from exc

    def _new_shell(self, artifact_dir: str | Path):
        return self._shell_class()(artifact_dir=artifact_dir)

    def _assert_run_dict(self, result: dict, *, route: str | None = None) -> None:
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "ok")
        if route is not None:
            self.assertEqual(result.get("route"), route)

        answer = result.get("answer")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 12)
        self.assertRegex(answer, r"[A-Za-z]")

        trace = result.get("trace")
        self.assertIsInstance(trace, list)
        self.assertTrue(trace, "run() must return a non-empty trace")
        self.assertTrue(all(isinstance(step, dict) for step in trace))

        organs = result.get("organs")
        self.assertIsInstance(organs, dict)
        self.assertTrue(REQUIRED_ORGANS.issubset(organs), organs)
        for organ in REQUIRED_ORGANS:
            self.assertTrue(organs[organ], f"{organ} must include evidence of presence")

    def _assert_trace_records_control(self, result: dict) -> None:
        trace = result["trace"]
        control_steps = [step for step in trace if step.get("organ") == "control"]
        self.assertTrue(control_steps, trace)
        self.assertTrue(
            any("route" in f"{step.get('step', '')} {step.get('action', '')}" for step in control_steps),
            control_steps,
        )

    def test_learns_recalls_from_noisy_query_and_persists(self) -> None:
        with TemporaryDirectory() as tmp:
            shell = self._new_shell(tmp)

            learned = shell.run("Remember: PhaseMesh project codename is Azure Compass.")
            self._assert_run_dict(learned, route="memory")
            self.assertTrue(learned["organs"]["learning"])
            self._assert_trace_records_control(learned)

            recalled = shell.run(
                "Noise: ticket 41 mentions blueprints and old names. "
                "What codename did I ask you to remember for the PhaseMesh project?"
            )
            self._assert_run_dict(recalled, route="memory")
            self.assertIn("azure compass", recalled["answer"].lower())
            self.assertTrue(recalled["organs"]["memory_retrieval"])
            self._assert_trace_records_control(recalled)

            shell.save()
            loaded = self._shell_class().load(tmp)
            persisted = loaded.run(
                "Ignore unrelated deployment notes. Recall the PhaseMesh project codename."
            )

        self._assert_run_dict(persisted, route="memory")
        self.assertIn("azure compass", persisted["answer"].lower())

    def test_routes_arithmetic_code_json_and_memory_prompts(self) -> None:
        with TemporaryDirectory() as tmp:
            shell = self._new_shell(tmp)
            shell.run("Remember: the demo room is called Lantern Hall.")

            cases = [
                ("What is 7 * 6?", "arithmetic", "42"),
                ("Write Python code for a function named add_one that returns n + 1.", "code", "add_one"),
                ('{"kind": "route-check", "ok": true}', "json", "object"),
                ("With distractor text around it, what demo room should you recall?", "memory", "lantern hall"),
            ]

            for prompt, route, expected_text in cases:
                with self.subTest(route=route):
                    result = shell.run(prompt)
                    self._assert_run_dict(result, route=route)
                    self._assert_trace_records_control(result)
                    self.assertIn(expected_text, result["answer"].lower())

    def test_shell_handles_chained_arithmetic_and_obvious_add_bug(self) -> None:
        with TemporaryDirectory() as tmp:
            shell = self._new_shell(tmp)

            arithmetic = shell.run("What is 12 * 60 * 5?")
            bug = shell.run("Find the bug in this Python function: def add(a, b): return a - b")

        self._assert_run_dict(arithmetic, route="arithmetic")
        self.assertIn("3600", arithmetic["answer"])
        self.assertEqual(arithmetic["data"]["reasoning"]["mode"], "arithmetic-chain")

        self._assert_run_dict(bug, route="code")
        self.assertIn("return a + b", bug["answer"])
        self.assertEqual(bug["data"]["reasoning"]["mode"], "code-correction")

    def test_binding_query_routes_without_explicit_recall_words(self) -> None:
        with TemporaryDirectory() as tmp:
            shell = self._new_shell(tmp)
            learned = shell.run("bind mira moved copper relay near north archive -> quartz-1842")
            self._assert_run_dict(learned, route="memory")

            result = shell.run("mira moved relay archive")

        self._assert_run_dict(result, route="memory")
        self._assert_trace_records_control(result)
        self.assertIn("quartz-1842", result["answer"].lower())

    def test_generates_fluent_answer_and_reports_all_organs(self) -> None:
        with TemporaryDirectory() as tmp:
            shell = self._new_shell(tmp)
            result = shell.run("Explain in one sentence what a phase-mesh shell is doing.")

        self._assert_run_dict(result, route="generation")
        self._assert_trace_records_control(result)
        answer = result["answer"].strip()
        self.assertGreaterEqual(len(answer.split()), 8)
        self.assertIn(answer[-1], ".!?")

        organs = result["organs"]
        for organ in REQUIRED_ORGANS:
            self.assertIn(organ, organs)

    def test_generation_route_uses_phase_language_model_when_available(self) -> None:
        from phase_mesh.language_model import PhaseLanguageModel, PhaseLMConfig

        corpus = "\n".join([
            "phase accio answers with resonance . phase accio answers with memory .",
            "phase accio answers with resonance . phase accio routes language .",
            "phase accio answers with memory . phase accio routes language .",
        ] * 10)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "phase_lm"
            model = PhaseLanguageModel(PhaseLMConfig(order=2, phase_cells=256, vocab_size=128, seed=11))
            model.train_text(corpus)
            model.save(model_dir)

            shell = self._new_shell(root)
            result = shell.run("phase accio")

        self._assert_run_dict(result, route="generation")
        self._assert_trace_records_control(result)
        self.assertIn("answers", result["answer"].lower())

        reasoning = result["data"]["reasoning"]
        self.assertEqual(reasoning["mode"], "phase-language-model")
        self.assertTrue(reasoning["language_model"]["loaded"])
        self.assertGreater(reasoning["generated_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
