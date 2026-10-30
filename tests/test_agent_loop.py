from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REQUIRED_ROLES = {
    "observer",
    "memory",
    "reasoner",
    "planner",
    "critic",
    "executor",
    "recorder",
    "trainer",
}


class PhaseMeshAgentLoopTests(unittest.TestCase):
    def test_agent_loop_records_episode_and_preserves_shell_memory(self) -> None:
        from phase_mesh.agent_loop import PhaseMeshAgentLoop

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("PhaseMesh scaffold\n", encoding="utf-8")
            state_dir = root / "state"

            loop = PhaseMeshAgentLoop(workspace=workspace, state_dir=state_dir)
            learned = loop.run("Remember: PhaseMesh north star is computer-world learning.")
            recalled = loop.run("Recall the PhaseMesh north star.")

            self.assertEqual(learned["status"], "ok")
            self.assertEqual(recalled["status"], "ok")
            self.assertTrue(REQUIRED_ROLES.issubset(recalled["agents"]))
            self.assertIn("computer-world learning", recalled["answer"].lower())
            self.assertEqual(recalled["plan"]["execute"], False)
            self.assertEqual(recalled["agents"]["executor"]["output"]["executed"], False)
            self.assertEqual(recalled["agents"]["trainer"]["output"]["trained"], False)
            self.assertEqual(recalled["prediction"]["episode_count_after"], 2)
            self.assertTrue((state_dir / "episodes.jsonl").exists())

            rows = [
                json.loads(line)
                for line in (state_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["agents"]["critic"]["output"]["approved"], True)

    def test_agent_loop_cli_emits_json_episode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "calc.py").write_text("print(7 * 6)\n", encoding="utf-8")
            state_dir = root / "state"

            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_mesh",
                    "agent-loop",
                    "What is 7 * 6?",
                    "--workspace",
                    str(workspace),
                    "--state-dir",
                    str(state_dir),
                    "--json",
                ],
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(run.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["agents"]["reasoner"]["output"]["route"], "arithmetic")
            self.assertIn("42", payload["answer"])
            self.assertEqual(payload["plan"]["execute"], False)
            self.assertEqual(payload["agents"]["executor"]["output"]["executed"], False)
            self.assertTrue((state_dir / "episodes.jsonl").exists())

    def test_agent_loop_can_execute_policy_gated_inspection(self) -> None:
        from phase_mesh.agent_loop import PhaseMeshAgentLoop

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
            (workspace / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            state_dir = root / "state"

            loop = PhaseMeshAgentLoop(workspace=workspace, state_dir=state_dir, execute_readonly=True)
            result = loop.run("Write Python code for a function named add_one that returns n + 1.")

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["plan"]["action"], "inspect")
            self.assertEqual(result["plan"]["execute"], True)
            self.assertEqual(result["agents"]["critic"]["output"]["approved"], True)
            self.assertEqual(result["action_result"]["executed"], True)
            self.assertEqual(result["action_result"]["returncode"], 0)
            self.assertEqual(result["prediction_score"]["status"], "scored")
            self.assertGreaterEqual(result["prediction_score"]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
